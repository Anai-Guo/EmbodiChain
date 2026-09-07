# FEPSolver

`FEPSolver` provides Python (PyTorch) and Warp implementations of the numerical
FEP interface in [HolisticMotion's FEPKinematics](https://github.com/chase6305/HolisticMotion/blob/1b4a5ce2cc9dab9228f5de79115d7fc9f048a1ac/src/kinematics/fep/FEPKinematics.cpp).
It supports offset serial chains with exactly seven revolute or continuous
joints. Geometry is read from the selected URDF chain; fixed transforms before,
between and after moving joints are retained.

This follows the reference's **numerical facade**, not an analytical FEP
decomposition. Shoulder, elbow and wrist labels are the signs of joints 2, 4
and 6, with zero assigned +1. Joint 7 supplies a redundancy **seed**, and is free
to move during correction. Numerical branch enumeration does not guarantee
all solutions or global convergence. No HolisticMotion binary is required.

## Key features

- One `SolverCfg` / `BaseSolver` interface for Python CPU/CUDA and Warp CPU/CUDA;
  the default selects Warp.
- Double-precision damped least-squares correction; float32 public outputs.
- Convergence checked at the actual output precision, with adaptive damping near the goal.
- Conservative geometric rejection before candidate generation for targets beyond reach.
- Shared URDF FK validation, TCP handling, limits and nearest-solution ranking.
- Failed solutions preserve their corresponding original input seed.
- Batched calls through `Robot.compute_batch_ik`, including matrix and xyzquat poses.

## Configuration

```python
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.solvers import FEPSolverCfg

cfg = FEPSolverCfg(
    urdf_path=get_data_path("Franka/Panda/PandaWithHand.urdf"),
    root_link_name="base",
    end_link_name="fr3_hand_tcp",
    backend="auto",  # Warp on CPU and CUDA; set "python" for the PyTorch path
    solve_method="seeded_numerical",
    pos_eps=1e-5,
    rot_eps=1e-5,
    max_iterations=200,
    adaptive_damping=True,
)
solver = cfg.init_solver(device="cpu")
warp_solver = cfg.replace(backend="warp").init_solver(device="cpu")
# On a CUDA-capable host:
# warp_solver = cfg.replace(backend="warp").init_solver(device="cuda:0")
```

`joint_names` may be omitted and will be inferred. If supplied, the names must
match the seven joints in serial-chain order. Prismatic joints and other DOF
counts fail during initialization. URDF limits are intersected with configured
and runtime limits; empty or nonfinite effective bounds are rejected.

```{tip}
Use `seeded_numerical` with the previous joint configuration for nearby targets.
Try `all_configurations` or `nearest_redundancy` when a single seed fails. These
methods increase the number of numerical solves and can take substantially longer.
```

| Setting | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | Warp on CPU and CUDA. Explicit Python also accepts CUDA. |
| `solve_method` | `"seeded_numerical"` | Seed strategy described below. |
| `pos_eps`, `rot_eps` | `1e-5`, `1e-5` | TCP position in metres and rotation angle in radians. |
| `max_iterations` | `200` | Update cap per candidate. |
| `damp` | `0.01` | Maximum lambda in `J Jᵀ + lambda² I`, or fixed lambda when adaptation is disabled. |
| `adaptive_damping` | `True` | Reduce damping near convergence, down to 1% of `damp`; `False` uses fixed damping. |
| `step_size` | `1.0` | Multiplier on the computed joint update. |
| `max_step` | `0.35` | Maximum absolute joint update in radians; scales the whole update. |
| `redundancy_step` | `pi / 36` | Radial joint-seven seed increment. |
| `redundancy_samples` | `37` | Levels including zero; radii exceeding pi are excluded. |

For the six-component pose residual `e = [translation_error, rotation_vector]`,
adaptive damping uses `lambda² = damp² * clip(10 * ||e||, 1e-4, 1)`.
The full configured damping is retained while the combined residual norm is at
least 0.1; it decreases smoothly near the target so weak Jacobian directions
converge faster. Set `adaptive_damping=False` to retain fixed regularization.
The position and rotation acceptance tolerances are unchanged.

There are four solve methods:

1. `seeded_numerical`: correct the supplied seed.
2. `configuration`: seed joints 2, 4 and 6 with their original signs and at least
   0.35 radians magnitude, then require the solved signs to match.
3. `all_configurations`: try all eight sign combinations and rank successful candidates.
4. `nearest_redundancy`: try joint-seven seeds at zero offset and both signs of
   increasing offsets. Select the nearest result from the first successful
   radius for each target. Only unresolved targets advance to the next radius;
   successful rows stop immediately. At most two seeds per unresolved target are
   evaluated in each subsequent batch, without allocating a complete radial grid.

## Main methods

### `get_ik`

```python
success, joints = solver.get_ik(
    target_xpos,
    qpos_seed=None,
    return_all_solutions=False,
    solve_method=None,
)
```

+ `target_xpos`: TCP pose relative to `root_link_name`, shape `(4, 4)` or `(N, 4, 4)`.
+ `qpos_seed`: `(7,)`, broadcast across targets, or `(N, 7)`. Defaults to zero,
  clamped to effective limits before iteration.
+ `return_all_solutions`: forces eight-configuration enumeration, regardless of
  the selected method. Results are numerically found candidates, not an
  exhaustive analytical solution set.
+ `solve_method`: optional per-call override of the configuration.
+ Returns: boolean `(N,)` and float32 `(N, 7)`, or `(N, 8)` and `(N, 8, 7)` for
  all solutions, on the solver's device. Invalid slots contain the original seed;
  consult validity before commanding a robot. Empty batches are supported.

Candidates are sorted by squared, per-joint weighted wrapped angular distance
to the input seed. In all-solutions mode, periodic duplicates are invalidated.
Invalid slots may appear between valid slots after deduplication, so always
use the mask. Nonfinite inputs, malformed rigid transforms and shape mismatches
raise `ValueError`.

Example using both result formats:

```python
qpos = torch.tensor([[0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.2]])
target = solver.get_fk(qpos)
seed = qpos + torch.tensor([[0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.04]])
success, joints = solver.get_ik(target, seed)
assert success.all()
validity, candidates = solver.get_ik(target, seed, return_all_solutions=True)
valid_candidates = candidates[0, validity[0]]
```

### Inherited methods and configuration labels

+ `get_fk(qpos)`: shared URDF FK with the current TCP; returns `(N, 4, 4)`.
+ `set_tcp(xpos)`: sets a `(4, 4)` flange-to-TCP matrix. Both IK backends read
  the current TCP on every call.
+ `set_qpos_limits(lower, upper)`, `update_with_robot_limit(limits)` and
  `get_qpos_limits()`: shared joint-limit management.
+ `set_ik_nearest_weight(weights)`: updates the next call's candidate ranking.
+ `get_configuration(qpos)`: returns `(..., 4)` containing the signs of joints
  2, 4 and 6 followed by joint 7's angle, for `qpos` of shape `(..., 7)`.

## How it works

1. Validate target poses and seeds, generate candidate seeds, and clamp them to limits.
   Before generating candidates, remove targets whose implied last-joint position
   lies beyond the sum of the intervening joint-to-joint lengths measured from
   the first joint. The bound uses the current TCP and includes both pose
   tolerances and numerical padding. It is a necessary geometric condition;
   targets inside it can still be unreachable. Rejected rows retain their input
   seeds and batch positions, including all-solutions and radial-search calls.
2. Evaluate TCP FK and the world-frame geometric Jacobian at the float32 joint
   values the API can actually return. Updates remain in float64. Both
   iteration paths compose the seven moving frames as rotations/translations,
   avoiding full matrices for every intermediate URDF link.
3. Compute translation error and the shortest axis-angle rotation error, including
   rotations near pi. Apply bounded damped least-squares updates until convergence
   or the iteration cap. Adaptive damping reduces regularization near the goal.
   Python CPU batches with rotation errors below 60 degrees construct only the
   reference conversion's positive-real quaternion candidate, preserving its
   arithmetic and small-angle series. Larger rotations use the full conversion.
   Warp runs this entire loop in one thread per candidate.
4. Recheck the returned float32 joints using shared double-precision URDF FK,
   then reject limit violations and configuration mismatches.
5. Rank candidates, select the first successful radius where applicable, and
   mark periodic duplicates invalid in all-solutions mode.

The correction has explicit damping and step bounds rather than reproducing the
reference C++ solver's coarse solve followed by a separate refinement pass. Warp
implements batch IK; the inspected HolisticMotion CUDA backend accelerates batch
FK and candidate scoring. CUDA launches share the caller's PyTorch stream.

## Validation and benchmark

```bash
pytest tests/sim/solvers/test_fep_solver.py -q
pytest tests/sim/solvers/test_fep_solver.py --run-gpu -m gpu -q
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s fep
```

The benchmark uses the packaged Franka model with identical targets and perturbed
seeds for Python CPU, Warp CPU and, when available, Warp CUDA, with both seeded
and nearest-redundancy methods. Each is also measured with every tenth target
shifted by 20 metres, so the cost of rejecting impossible targets in a mixed
batch is visible. For these `mixed_unreachable` components, `success_rate` means
agreement with expected validity: reachable targets must solve and shifted
targets must fail. Pose-error means use all reachable rows, including failed
solves; failed shifted targets are checked for seed preservation. The shared
report contains timing/memory, success/accuracy and leaderboard tables. Model
assets must be available through `get_data_path`.

## References

The porting reference was the local HolisticMotion working tree inspected on
2026-09-05, based on commit `1b4a5ce2cc9dab9228f5de79115d7fc9f048a1ac`, with
uncommitted changes in `FEPKinematics.cpp` and `NumericalKinematics.cpp`. These
links identify the repository baseline. The port uses EmbodiChain's URDF FK
conventions; its fixed-frame geometry is also tested against independent
XML/SciPy transform composition.

- [HolisticMotion FEP implementation, baseline revision](https://github.com/chase6305/HolisticMotion/blob/1b4a5ce2cc9dab9228f5de79115d7fc9f048a1ac/src/kinematics/fep/FEPKinematics.cpp)
- [Reference numerical correction](https://github.com/chase6305/HolisticMotion/blob/1b4a5ce2cc9dab9228f5de79115d7fc9f048a1ac/src/kinematics/NumericalKinematics.cpp)
