# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.solvers import FEPSolver, FEPSolverCfg, SolverCfg

REFERENCE_QPOS = [0.2, -0.35, 0.25, -0.6, 0.3, 0.4, -0.2]
SEED_OFFSET = [0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.04]
METHODS = [
    "seeded_numerical",
    "configuration",
    "all_configurations",
    "nearest_redundancy",
]


@pytest.fixture
def urdf_path(tmp_path: Path) -> Path:
    """Offset 7R model with oblique axes, fixed frames and an offset root."""
    axes = ["0 0 1", "0 1 0", "0.6 0 0.8", "0 1 0", "1 0 0", "0 1 0", "1 0 0"]
    links = ["world", "base", "spacer", "tool"] + [f"link{i}" for i in range(1, 8)]
    elements = [f'<link name="{name}"/>' for name in links]
    elements.append(
        '<joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="1 2 3" rpy="0.3 -0.2 0.1"/></joint>'
    )
    elements.append(
        '<joint name="spacer" type="fixed"><parent link="link3"/><child link="spacer"/><origin xyz="0.04 -0.02 0.03" rpy="0.1 -0.2 0.3"/></joint>'
    )
    for i, axis in enumerate(axes, 1):
        parent = "base" if i == 1 else ("spacer" if i == 4 else f"link{i-1}")
        elements.append(f"""<joint name="joint{i}" type="revolute">
          <parent link="{parent}"/><child link="link{i}"/>
          <origin xyz="0.02 0.01 0.1" rpy="0.05 0.02 -0.03"/>
          <axis xyz="{axis}"/><limit lower="-2.8" upper="2.8" velocity="1" effort="10"/>
        </joint>""")
    elements.append(
        '<joint name="flange" type="fixed"><parent link="link7"/><child link="tool"/><origin xyz="0.05 -0.03 0.1" rpy="-0.2 0.3 0.1"/></joint>'
    )
    path = tmp_path / "offset_7r.urdf"
    path.write_text(
        '<robot name="offset_7r">' + "".join(elements) + "</robot>", encoding="utf-8"
    )
    return path


def make_solver(
    urdf_path: Path, backend: str, device: str = "cpu", **kwargs
) -> FEPSolver:
    """Build the production solver directly, without a simulation context."""
    return FEPSolverCfg(
        urdf_path=str(urdf_path),
        root_link_name="base",
        end_link_name="tool",
        backend=backend,
        **kwargs,
    ).init_solver(device=device)


@pytest.fixture(params=["python", "warp"])
def solver(urdf_path: Path, request: pytest.FixtureRequest) -> FEPSolver:
    return make_solver(urdf_path, request.param)


def assert_pose_matches(
    solver: FEPSolver, joints: torch.Tensor, target: torch.Tensor
) -> None:
    """Check FK translation and geodesic angle independently of solver success."""
    from pytorch_kinematics.transforms import matrix_to_axis_angle

    actual = solver.get_fk(joints).double()
    translation = torch.linalg.vector_norm(actual[:, :3, 3] - target[:, :3, 3], dim=-1)
    rotation = matrix_to_axis_angle(
        actual[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2).double()
    )
    assert torch.all(translation < 2e-5)
    assert torch.all(torch.linalg.vector_norm(rotation, dim=-1) < 2e-5)


def test_round_trip_and_nonexact_seeds(solver: FEPSolver) -> None:
    generator = torch.Generator().manual_seed(27)
    qpos = torch.tensor(REFERENCE_QPOS) + 0.3 * torch.rand((16, 7), generator=generator)
    targets = solver.get_fk(qpos)
    exact_valid, exact = solver.get_ik(targets, qpos)
    assert exact_valid.all()
    torch.testing.assert_close(exact, qpos)
    valid, joints = solver.get_ik(targets, qpos + torch.tensor(SEED_OFFSET))
    assert valid.shape == (16,) and joints.shape == (16, 7)
    assert valid.all()
    assert_pose_matches(solver, joints, targets)


def test_auto_backend_uses_warp_cpu(urdf_path: Path) -> None:
    """The default backend should select the faster Warp correction path."""
    auto = make_solver(urdf_path, "auto")
    warp = make_solver(urdf_path, "warp")
    assert auto.backend == "warp"
    qpos = torch.tensor([REFERENCE_QPOS])
    target = auto.get_fk(qpos)
    seed = qpos + torch.tensor(SEED_OFFSET)
    auto_valid, auto_joints = auto.get_ik(target, seed)
    warp_valid, warp_joints = warp.get_ik(target, seed)
    torch.testing.assert_close(auto_valid, warp_valid)
    torch.testing.assert_close(auto_joints, warp_joints, atol=0, rtol=0)


@pytest.mark.parametrize("large_angles", [False, True])
def test_pose_error_preserves_reference_arithmetic(large_angles: bool) -> None:
    """Fast small-angle correction must preserve the reference's numerical path."""
    from pytorch_kinematics.transforms import matrix_to_axis_angle
    from scipy.spatial.transform import Rotation

    generator = torch.Generator().manual_seed(203)
    axes = torch.randn((128, 3), generator=generator, dtype=torch.float64)
    axes /= axes.norm(dim=-1, keepdim=True)
    angles = torch.logspace(-12, -0.1, 128, dtype=torch.float64)
    if large_angles:
        angles[::4] = math.pi
        angles[1::4] = math.pi - 1e-8
        angles[2::4] = math.pi / 3 + 1e-8
    target = torch.eye(4, dtype=torch.float64).repeat(128, 1, 1)
    actual = target.clone()
    # Include float32 FK roundoff as well as exact SO(3) rotations.
    target[:, :3, :3] = (
        torch.tensor(Rotation.from_rotvec((axes * angles[:, None]).numpy()).as_matrix())
        .float()
        .double()
    )
    expected = matrix_to_axis_angle(target[:, :3, :3])
    error = FEPSolver._pose_error(target, actual)
    torch.testing.assert_close(error[:, 3:], expected, atol=0, rtol=0)
    assert torch.isfinite(error).all()


@pytest.mark.parametrize("method", METHODS + ["return_all"])
def test_outside_reach_bound_skips_correction_and_preserves_seed(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    target = torch.eye(4).repeat(3, 1, 1)
    target[:, 0, 3] = 20
    seeds = torch.tensor([REFERENCE_QPOS]).repeat(3, 1)
    seeds[1, 2] = 10  # Even an out-of-limits failed seed must remain unchanged.

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("An outside-reach target entered numerical correction")

    monkeypatch.setattr(solver, "_solve_validated", unexpected_solve)
    all_solutions = method == "return_all"
    valid, joints = solver.get_ik(
        target,
        seeds,
        solve_method="seeded_numerical" if all_solutions else method,
        return_all_solutions=all_solutions,
    )
    assert not valid.any()
    expected = seeds[:, None].expand(-1, 8, -1) if all_solutions else seeds
    torch.testing.assert_close(joints, expected, atol=0, rtol=0)


@pytest.mark.parametrize("all_solutions", [False, True])
def test_reach_filter_keeps_batch_order_and_runtime_tcp(
    solver: FEPSolver, all_solutions: bool
) -> None:
    from scipy.spatial.transform import Rotation

    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.6]).as_matrix()
    tcp[:3, 3] = [1.2, -0.7, 0.5]
    solver.set_tcp(tcp)
    qpos = torch.tensor([REFERENCE_QPOS]) + torch.arange(3)[:, None] * 0.01
    target = solver.get_fk(qpos)
    seeds = qpos + torch.tensor(SEED_OFFSET)
    expected_valid, expected = solver.get_ik(
        target, seeds, return_all_solutions=all_solutions
    )
    mixed = target.repeat_interleave(2, dim=0)
    mixed[1::2, 0, 3] += 20
    mixed_seeds = seeds.repeat_interleave(2, dim=0)
    valid, joints = solver.get_ik(
        mixed, mixed_seeds, return_all_solutions=all_solutions
    )
    torch.testing.assert_close(valid[::2], expected_valid)
    torch.testing.assert_close(joints[::2], expected, atol=0, rtol=0)
    assert not valid[1::2].any()
    failed = seeds[:, None].expand(-1, 8, -1) if all_solutions else seeds
    torch.testing.assert_close(joints[1::2], failed, atol=0, rtol=0)


def test_reach_bound_retains_random_reachable_poses(solver: FEPSolver) -> None:
    generator = torch.Generator().manual_seed(1301)
    joints = -2.8 + 5.6 * torch.rand((512, 7), generator=generator)
    for offset in ([0, 0, 0], [2.0, -1.5, 3.0]):
        tcp = np.eye(4)
        tcp[:3, 3] = offset
        solver.set_tcp(tcp)
        target = solver.get_fk(joints).double()
        assert solver._within_reach_bound(target, torch.tensor(tcp)).all()


def test_reach_bound_accepts_extended_chain_and_pose_tolerance(
    urdf_path: Path,
) -> None:
    """The outer boundary, root offset, and TCP tolerance must not cause rejection."""
    from xml.etree import ElementTree

    tree = ElementTree.parse(urdf_path)
    for joint in tree.getroot().findall("joint"):
        joint.find("origin").set("xyz", "0.1 0 0")
        joint.find("origin").set("rpy", "0 0 0")
        if joint.find("axis") is not None:
            joint.find("axis").set("xyz", "0 0 1")
    tree.write(urdf_path)
    for backend in ("python", "warp"):
        solver = make_solver(urdf_path, backend)
        tcp = np.eye(4)
        tcp[:3, 3] = [2, 0, 0]
        solver.set_tcp(tcp)
        seed = torch.zeros((1, 7))
        target = solver.get_fk(seed).double()
        target[:, 0, 3] += solver.cfg.pos_eps * 0.5
        tcp_tensor = torch.tensor(tcp)
        assert solver._within_reach_bound(target, tcp_tensor).all()
        valid, joints = solver.get_ik(target, seed)
        assert valid.all()
        torch.testing.assert_close(joints, seed, atol=0, rtol=0)
        target[:, 0, 3] += 0.1
        assert not solver._within_reach_bound(target, tcp_tensor).any()
        # A permitted rotation error with a long TCP changes the inferred wrist
        # position, so the bound must include the rotational tolerance too.
        solver.cfg.rot_eps = 1e-3
        tcp[:3, 3] = [0, 2, 0]
        solver.set_tcp(tcp)
        target = solver.get_fk(seed).double()
        angle = solver.cfg.rot_eps * 0.5
        target[0, :2, :2] = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=torch.float64,
        )
        assert solver._within_reach_bound(target, torch.tensor(tcp)).all()
        valid, joints = solver.get_ik(target, seed)
        assert valid.all()
        torch.testing.assert_close(joints, seed, atol=0, rtol=0)


@pytest.mark.parametrize("constraint", ["position", "rotation"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_float32_boundary_is_refined_before_acceptance(
    solver: FEPSolver, constraint: str, adaptive: bool
) -> None:
    """A valid float64 seed may need correction before float32 can be returned."""
    from pytorch_kinematics.transforms.rotation_conversions import (
        axis_and_angle_to_matrix_33,
        matrix_to_axis_angle,
    )

    solver.cfg.adaptive_damping = adaptive
    seed = torch.tensor([REFERENCE_QPOS], dtype=torch.float64) + 1e-8
    raw = solver._chain.forward_kinematics_tensor(seed)[-1]
    rounded = solver._chain.forward_kinematics_tensor(seed.float().double())[-1]
    target = raw.clone()
    if constraint == "position":
        difference = raw[:, :3, 3] - rounded[:, :3, 3]
        tolerance = solver.cfg.pos_eps
    else:
        difference = matrix_to_axis_angle(
            raw[:, :3, :3] @ rounded[:, :3, :3].transpose(-1, -2)
        )
        tolerance = solver.cfg.rot_eps
    magnitude = difference.norm(dim=-1)
    assert magnitude.item() > 1e-10
    displacement = (
        difference / magnitude[:, None] * (tolerance - magnitude[:, None] / 4)
    )
    if constraint == "position":
        target[:, :3, 3] += displacement
    else:
        target[:, :3, :3] = (
            axis_and_angle_to_matrix_33(
                difference / magnitude[:, None], tolerance - magnitude / 4
            )
            @ raw[:, :3, :3]
        )
    offset = slice(0, 3) if constraint == "position" else slice(3, 6)
    assert solver._pose_error(target, raw)[:, offset].norm() < tolerance
    assert solver._pose_error(target, rounded)[:, offset].norm() > tolerance
    success, joints = solver.get_ik(target, seed)
    assert success.all()
    actual = solver._chain.forward_kinematics_tensor(joints.double())[-1]
    error = solver._pose_error(target, actual)
    assert error[:, :3].norm() <= solver.cfg.pos_eps
    assert error[:, 3:].norm() <= solver.cfg.rot_eps


def test_compact_jacobian_matches_finite_differences(solver: FEPSolver) -> None:
    """Check the optimized iteration geometry against shared URDF FK derivatives."""
    from pytorch_kinematics.transforms import matrix_to_axis_angle

    qpos = torch.tensor([REFERENCE_QPOS], dtype=torch.float64)
    tcp = qpos.new_tensor(
        [[0, -1, 0, 0.1], [1, 0, 0, -0.2], [0, 0, 1, 0.05], [0, 0, 0, 1]]
    )
    jacobian, pose = solver._evaluate(qpos, tcp)
    reference = solver._chain.forward_kinematics_tensor(qpos)[-1] @ tcp
    torch.testing.assert_close(pose, reference, atol=1e-7, rtol=0)
    step = 1e-5
    perturbations = step * torch.eye(7, dtype=qpos.dtype)
    plus = solver._chain.forward_kinematics_tensor(qpos + perturbations)[-1] @ tcp
    minus = solver._chain.forward_kinematics_tensor(qpos - perturbations)[-1] @ tcp
    linear = (plus[:, :3, 3] - minus[:, :3, 3]) / (2 * step)
    angular = matrix_to_axis_angle(
        plus[:, :3, :3] @ minus[:, :3, :3].transpose(-1, -2)
    ) / (2 * step)
    expected = torch.cat((linear, angular), dim=-1).T
    torch.testing.assert_close(jacobian[0], expected, atol=2e-6, rtol=0)


def test_redundancy_search_stops_for_solved_targets(
    solver: FEPSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    qpos = torch.tensor([REFERENCE_QPOS])
    targets = solver.get_fk(qpos).expand(4, -1, -1)
    seed = (qpos + torch.tensor(SEED_OFFSET)).expand(4, -1)
    original = solver._correct_candidates
    evaluated = []

    def counted(*args):
        evaluated.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(solver, "_correct_candidates", counted)
    # A huge level count must not allocate the full redundancy grid.
    solver.cfg.redundancy_samples = 10**8
    success, joints = solver.get_ik(targets, seed, solve_method="nearest_redundancy")
    assert success.all()
    assert evaluated == [4]
    assert_pose_matches(solver, joints, targets)


def test_redundancy_frontier_matches_exhaustive_radii(solver: FEPSolver) -> None:
    """Keep zero-radius successes while recovering another row at a later radius."""
    solver.cfg.max_iterations = 4
    solver.cfg.redundancy_samples = 5
    solver.cfg.redundancy_step = 0.2
    solver.set_ik_nearest_weight(np.array([2, 1, 4, 1, 3, 1, 0.5]))
    seeds = torch.tensor(
        [
            REFERENCE_QPOS,
            [
                0.41901102662086487,
                -0.21067480742931366,
                0.043375164270401,
                -0.9100480079650879,
                -0.29940688610076904,
                0.3656739294528961,
                -0.18487268686294556,
            ],
            REFERENCE_QPOS,
        ]
    )
    target = solver.get_fk(torch.tensor([REFERENCE_QPOS])).repeat(3, 1, 1)
    target[2, 0, 3] = 20
    initial_valid, _ = solver.get_ik(target, seeds)
    assert initial_valid.tolist() == [True, False, False]
    valid, joints = solver.get_ik(target, seeds, solve_method="nearest_redundancy")
    assert valid.tolist() == [True, True, False]
    levels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
    offsets = torch.tensor([0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8])
    trials = seeds[:, None].repeat(1, len(offsets), 1)
    trials[..., 6] += offsets
    full_target = target[:, None].expand(-1, len(offsets), -1, -1).reshape(-1, 4, 4)
    full_valid, candidates = solver.get_ik(full_target, trials.reshape(-1, 7))
    full_valid = full_valid.reshape(3, -1)
    candidates = candidates.reshape(3, -1, 7)
    expected = seeds.clone()
    for row in range(3):
        if not full_valid[row].any():
            continue
        first_level = levels[full_valid[row]].min()
        eligible = full_valid[row] & (levels == first_level)
        delta = (
            torch.remainder(candidates[row] - seeds[row] + math.pi, 2 * math.pi)
            - math.pi
        )
        costs = (
            ((delta * solver.ik_nearest_weight) ** 2)
            .sum(-1)
            .masked_fill(~eligible, float("inf"))
        )
        expected[row] = candidates[row, costs.argmin()]
    torch.testing.assert_close(joints, expected, atol=2e-5, rtol=0)


def test_fixed_frame_geometry_against_independent_urdf_fk(solver: FEPSolver) -> None:
    """Verify frame composition directly from XML, without the PK/Warp model."""
    from scipy.spatial.transform import Rotation

    root = ET.parse(solver.urdf_path).getroot()
    by_parent = {
        joint.find("parent").get("link"): joint for joint in root.findall("joint")
    }

    def independent_fk(qpos: np.ndarray) -> np.ndarray:
        actual = np.eye(4)
        parent = "base"
        while parent != "tool":
            joint = by_parent[parent]
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            actual = actual @ transform
            if joint.get("type") != "fixed":
                index = int(joint.get("name").removeprefix("joint")) - 1
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    axis / np.linalg.norm(axis) * qpos[index]
                ).as_matrix()
                actual = actual @ motion
            parent = joint.find("child").get("link")
        return actual

    target = independent_fk(np.array(REFERENCE_QPOS))
    np.testing.assert_allclose(
        solver.get_fk(torch.tensor(REFERENCE_QPOS))[0], target, rtol=0, atol=1e-6
    )
    seed = torch.tensor(REFERENCE_QPOS) + torch.tensor(SEED_OFFSET)
    valid, joints = solver.get_ik(torch.from_numpy(target), seed)
    assert valid.all()
    np.testing.assert_allclose(
        independent_fk(joints[0].numpy()), target, rtol=0, atol=2e-5
    )


@pytest.mark.parametrize("method", METHODS)
def test_methods_preserve_tcp_updates_and_branch_labels(
    solver: FEPSolver, method: str
) -> None:
    tool = np.array([[0, -1, 0, 0.1], [1, 0, 0, -0.2], [0, 0, 1, 0.05], [0, 0, 0, 1]])
    qpos = torch.tensor([REFERENCE_QPOS])
    seed = qpos + torch.tensor(SEED_OFFSET)
    for tcp in (tool, np.eye(4), tool):
        solver.set_tcp(tcp)
        target = solver.get_fk(qpos)
        valid, joints = solver.get_ik(target, seed, solve_method=method)
        assert valid.all()
        assert_pose_matches(solver, joints, target)
        if method == "configuration":
            torch.testing.assert_close(
                solver.get_configuration(joints)[..., :3],
                solver.get_configuration(seed)[..., :3],
            )


def test_all_solutions_are_valid_distinct_and_sorted(solver: FEPSolver) -> None:
    qpos = torch.tensor([REFERENCE_QPOS])
    target = solver.get_fk(qpos)
    assert solver.set_ik_nearest_weight(np.array([2, 1, 4, 1, 3, 1, 0.5]))
    valid, joints = solver.get_ik(target, qpos, return_all_solutions=True)
    assert valid.shape == (1, 8) and joints.shape == (1, 8, 7)
    assert valid.any()
    selected = joints[valid]
    assert_pose_matches(solver, selected, target.expand(len(selected), -1, -1))
    delta = torch.remainder(selected - qpos + math.pi, 2 * math.pi) - math.pi
    costs = ((delta * solver.get_ik_nearest_weight()) ** 2).sum(-1)
    assert torch.all(costs[1:] >= costs[:-1] - 1e-10)
    for i in range(len(selected)):
        for j in range(i):
            difference = (
                torch.remainder(selected[i] - selected[j] + math.pi, 2 * math.pi)
                - math.pi
            )
            assert (difference**2).sum() >= 1e-12
    torch.testing.assert_close(joints[~valid], qpos.expand((~valid).sum(), -1))


def test_unreachable_targets_and_iteration_exhaustion_preserve_seed(
    solver: FEPSolver,
) -> None:
    seed = torch.tensor([REFERENCE_QPOS])
    target = solver.get_fk(seed)
    target[:, 0, 3] = 20.0
    for all_solutions in (False, True):
        valid, joints = solver.get_ik(target, seed, return_all_solutions=all_solutions)
        assert not valid.any()
        torch.testing.assert_close(
            joints.reshape(-1, 7), seed.expand(joints.numel() // 7, -1)
        )
    solver.cfg.max_iterations = 1
    target = solver.get_fk(seed + 0.5)
    valid, joints = solver.get_ik(target, seed)
    assert not valid.any()
    torch.testing.assert_close(joints, seed)


def test_runtime_limits_and_empty_intersection(solver: FEPSolver) -> None:
    qpos = torch.tensor([REFERENCE_QPOS])
    target = solver.get_fk(qpos)
    # A single allowed configuration must reject an unrelated pose.
    solver.set_qpos_limits([0.0] * 7, [0.0] * 7)
    valid, joints = solver.get_ik(target, qpos)
    assert not valid.any()
    torch.testing.assert_close(joints, qpos)
    solver.set_qpos_limits([4.0] * 7, [5.0] * 7)
    with pytest.raises(ValueError, match="effective joint limits"):
        solver.get_ik(target, qpos)


def test_shapes_empty_and_noncontiguous_inputs(solver: FEPSolver) -> None:
    qpos = torch.tensor(REFERENCE_QPOS)
    target = solver.get_fk(qpos)
    valid, joints = solver.get_ik(target[0], qpos)
    assert valid.shape == (1,) and joints.shape == (1, 7)
    storage = torch.zeros((4, 14))
    storage[:, ::2] = qpos
    targets = target.expand(8, -1, -1)[::2]
    valid, joints = solver.get_ik(targets, storage[:, ::2])
    assert valid.all()
    assert_pose_matches(solver, joints, targets)
    for all_solutions in (False, True):
        valid, joints = solver.get_ik(
            torch.empty((0, 4, 4)), return_all_solutions=all_solutions
        )
        assert valid.shape == ((0, 8) if all_solutions else (0,))
        assert joints.shape == ((0, 8, 7) if all_solutions else (0, 7))
    valid, joints = solver.get_ik(solver.get_fk(torch.zeros(7)))
    assert valid.all()


@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "nan",
        "reflection",
        "nonorthogonal",
        "last_row",
        "seed_shape",
        "seed_nan",
    ],
)
def test_invalid_inputs_are_rejected(solver: FEPSolver, invalid: str) -> None:
    seed = torch.tensor([REFERENCE_QPOS])
    target = solver.get_fk(seed)
    if invalid == "shape":
        target = target[:, :3]
    elif invalid == "nan":
        target[0, 0, 3] = float("nan")
    elif invalid == "reflection":
        target[0, :3, 0] *= -1
    elif invalid == "nonorthogonal":
        target[0, :3, 0] *= 2
    elif invalid == "last_row":
        target[0, 3, 0] = 1
    elif invalid == "seed_shape":
        seed = seed[:, :6]
    else:
        seed[0, 0] = float("nan")
    with pytest.raises(ValueError):
        solver.get_ik(target, seed)


def test_python_warp_agree_for_nonexact_seeds(urdf_path: Path) -> None:
    python = make_solver(urdf_path, "python")
    warp = make_solver(urdf_path, "warp")
    qpos = torch.tensor([REFERENCE_QPOS])
    seed = qpos + torch.tensor(SEED_OFFSET)
    target = python.get_fk(qpos)
    for method in METHODS:
        expected_valid, expected = python.get_ik(target, seed, solve_method=method)
        actual_valid, actual = warp.get_ik(target, seed, solve_method=method)
        torch.testing.assert_close(actual_valid, expected_valid)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


def test_rotation_error_at_pi_does_not_false_converge(solver: FEPSolver) -> None:
    seed = torch.tensor([REFERENCE_QPOS])
    target = solver.get_fk(seed)
    target[0, :3, :3] = target[0, :3, :3] @ torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    solver.cfg.max_iterations = 1
    valid, joints = solver.get_ik(target, seed)
    assert not valid.any()
    torch.testing.assert_close(joints, seed)


def test_robot_batch_bridge_accepts_matrix_and_xyzquat(solver: FEPSolver) -> None:
    # Exercise the actual Robot bridge; the sole simulated input is the root pose.
    root = torch.eye(4)[None]
    root[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    robot = SimpleNamespace(
        device=solver.device,
        _all_indices=[0],
        _solvers={"arm": solver},
        get_link_pose=lambda **kwargs: root,
    )
    qpos = torch.tensor([[REFERENCE_QPOS, REFERENCE_QPOS]])
    matrix = Robot.compute_batch_fk(robot, qpos, "arm", to_matrix=True)
    xyzquat = Robot.compute_batch_fk(robot, qpos, "arm", to_matrix=False)
    success_matrix, result_matrix = Robot.compute_batch_ik(robot, matrix, qpos, "arm")
    success_quat, result_quat = Robot.compute_batch_ik(robot, xyzquat, qpos, "arm")
    assert success_matrix.all() and success_quat.all()
    torch.testing.assert_close(result_matrix, result_quat, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(result_matrix, qpos, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "cuda"},
        {"solve_method": "analytic"},
        {"damp": 0},
        {"pos_eps": float("nan")},
        {"max_iterations": 0},
        {"redundancy_samples": 1.5},
    ],
)
def test_config_rejects_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FEPSolverCfg(**kwargs)


def test_configuration_factory_and_chain_validation(urdf_path: Path) -> None:
    cfg = SolverCfg.from_dict({"class_type": "FEPSolver", "backend": "warp"})
    assert isinstance(cfg, FEPSolverCfg) and cfg.backend == "warp"
    robot = RobotCfg.from_dict({"solver_cfg": {"arm": cfg.to_dict()}})
    assert isinstance(robot.solver_cfg["arm"], FEPSolverCfg)
    with pytest.raises(ValueError, match="serial-chain order"):
        make_solver(
            urdf_path, "python", joint_names=[f"joint{i}" for i in range(7, 0, -1)]
        )
    with pytest.raises(ValueError, match="seven revolute"):
        FEPSolverCfg(
            urdf_path=str(urdf_path), root_link_name="base", end_link_name="link6"
        ).init_solver()
    urdf_path.write_text(
        urdf_path.read_text().replace(
            'name="joint3" type="revolute"', 'name="joint3" type="prismatic"'
        )
    )
    with pytest.raises(ValueError, match="seven revolute"):
        make_solver(urdf_path, "python")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "method", ["seeded_numerical", "nearest_redundancy", "all_configurations"]
)
def test_fep_cuda_matches_python_on_nondefault_stream(
    urdf_path: Path, method: str
) -> None:
    reference = make_solver(urdf_path, "python")
    solver = make_solver(urdf_path, "warp", device="cuda")
    qpos = torch.tensor([REFERENCE_QPOS])
    target = reference.get_fk(qpos)
    expected_valid, expected = reference.get_ik(
        target, qpos + torch.tensor(SEED_OFFSET), solve_method=method
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        valid, joints = solver.get_ik(
            target.cuda(),
            (qpos + torch.tensor(SEED_OFFSET)).cuda(),
            solve_method=method,
        )
        actual_valid, actual = valid.cpu(), joints.cpu()
    torch.testing.assert_close(actual_valid, expected_valid)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.gpu
class BaseSolverTest:
    """Small real-robot integration; even CPU physics needs the Vulkan device."""

    sim = None

    def setup_simulation(self, sim_device: str, backend: str) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg

        self.sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device=sim_device)
        )
        cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
        cfg.solver_cfg["arm"] = FEPSolverCfg(
            root_link_name="base",
            end_link_name="fr3_hand_tcp",
            backend=backend,
        )
        self.robot = self.sim.add_robot(cfg=cfg)

    def test_robot_fk_ik_round_trip(self) -> None:
        qpos = torch.tensor(
            [[[0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.2]]], device=self.robot.device
        )
        matrix = self.robot.compute_batch_fk(qpos, "arm", to_matrix=True)
        xyzquat = self.robot.compute_batch_fk(qpos, "arm", to_matrix=False)
        valid, result = self.robot.compute_batch_ik(matrix, qpos, "arm")
        valid_quat, result_quat = self.robot.compute_batch_ik(xyzquat, qpos, "arm")
        assert valid.all() and valid_quat.all()
        torch.testing.assert_close(result, qpos, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(result_quat, result, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(
            self.robot.compute_batch_fk(result, "arm", to_matrix=True),
            matrix,
            atol=5e-3,
            rtol=5e-3,
        )
        perturbed = qpos + torch.tensor(SEED_OFFSET, device=self.robot.device)
        valid, result = self.robot.compute_batch_ik(matrix, perturbed, "arm")
        assert valid.all()
        torch.testing.assert_close(
            self.robot.compute_batch_fk(result, "arm", to_matrix=True),
            matrix,
            atol=5e-3,
            rtol=5e-3,
        )
        matrix[..., :3, 3] = 20.0
        valid, result = self.robot.compute_batch_ik(matrix, qpos, "arm")
        assert not valid.any() and result.shape == qpos.shape
        torch.testing.assert_close(result, qpos)

    def teardown_method(self) -> None:
        from embodichain.lab.sim import SimulationManager

        if self.sim is not None:
            self.sim.destroy()
            self.sim = None
        SimulationManager.flush_cleanup_queue()


class TestFEPSolver(BaseSolverTest):
    def setup_method(self) -> None:
        self.setup_simulation("cpu", "auto")


class TestFEPSolverCUDA(BaseSolverTest):
    def setup_method(self) -> None:
        self.setup_simulation("cuda", "warp")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
