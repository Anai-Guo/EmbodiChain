# Action Engine v2 Architecture

Action Engine v2 uses a task-first protocol and executes a direct
`AtomicAction` graph. The persisted graph is symbolic and coordinate-free;
simulator geometry is resolved immediately before each action executes.

The Task Engine entry point wraps that existing pipeline with three narrow
owners:

1. `TaskAgent` produces three scene-independent `TaskDraft` candidates and
   deterministically derives each `SceneRequest` and `SuccessSpec`.
2. `SceneAdapter` binds one verified candidate to a read-only existing scene,
   producing `SceneManifest`, `RoleBindings`, and a complete `BindingReport`.
3. `ActionAgent` lowers the selected `GroundedTaskPlan` to the existing
   `action_engine_seed_graph_v4`, performs executable capability preflight,
   runs it through `ProgramExecutor`, and emits a tensor-free
   `ExecutionReport`. Report v2 records the episode seed, package and Python
   versions, Git commit/dirty state when available, and structured runtime
   arguments alongside the existing plan and graph hashes.

The public CLI is `python -m embodichain.gen_sim.task_engine --mode ...` with
strict `image`, `image-edit`, `scene`, and `scene-edit` input modes. Every
invocation runs the complete workflow through real trajectory acceptance and
publishes an isolated, timestamped child under `--output-root`. Source projects
are referenced in place and integrity-hashed; Task Engine does not modify them
or copy them into a scene package store.

## Package Ownership

The cross-engine workflow is owned by Task Engine rather than nested under
Action Engine:

- `embodichain.gen_sim.task_engine` owns scene-independent interpretation,
  E1-E9 semantic ontology, `TaskDraft`, `SceneRequest`, `SuccessSpec`, and
  `TaskAgent`.
- `embodichain.gen_sim.scene_engine` remains the scene generation subsystem and
  exposes auditable image-understanding, materialization, edit-understanding,
  and edit-materialization stages.
- `embodichain.gen_sim.task_engine.scene` owns Scene Engine adaptation, the
  richer static manifest, and deterministic scene/action feasibility reports.
- `embodichain.gen_sim.action_engine.agent` owns `ActionAgent`; Action Engine's
  existing `domain`, `planning`, and `runtime` packages remain authoritative
  for graph compilation and execution.
- `embodichain.gen_sim.task_engine.orchestration` owns cross-engine contracts,
  read-only source references, scene adaptation, orchestration, and artifacts.
  `embodichain.gen_sim.task_engine.cli` owns the unified CLI.

The former `scene_bridge`, `collaboration`, and
`action_engine.collaboration` namespaces were removed; there are no import
bridges or fallback entry points.

## Data Flow

1. `TaskAgent` or a caller creates a validated `TaskSpec`.
2. Action Engine emits `SceneRequirements` for the external Scene Engine.
3. After scene generation, Task Engine's scene adapter preserves geometry, physics,
   articulation, affordance evidence, and provenance in a versioned
   `StaticSceneManifest` while the existing redacted manifest remains compatible.
4. `FeasibilityBroker` intersects the selected task, role bindings, static scene,
   robot profile, and executable capability catalog without repairing unknowns.
5. Offline recipes and the online planner independently create complete
   `SeedGraph` candidates whose nodes are `AtomicAction` calls.
6. Runtime preflight checks the capability catalog and rejects planning-only
   actions before simulator motion starts.
7. `ActionGrounder` reads live robot, object, articulation, and camera state and
   materializes the typed goal and immutable action options just in time.
8. `ProgramExecutor` schedules the DAG, executes vectorized action masks, and
   verifies semantic postconditions from live state.

There is no persisted semantic task graph between `TaskSpec` and `SeedGraph`.
The standalone TaskAgent v1 planner and compiler remain available to their
existing callers, but the generation pipeline neither accepts nor publishes
TaskAgent v1 JSON.

## Protocols

### TaskSpec

`TaskSpec` owns the level, public instruction, E1-E9 task instances,
dependencies, path-independent success conditions, and a private oracle. The
online planner receives `public_task_spec(...)`, which removes the oracle and,
for L4, the hidden reference task instances. Online L4 TaskGroups are inferred
from the instruction and observations rather than matched to an oracle path.

Levels classify how the task is specified, not action count:

- L1: one E instance.
- L2: two or more instances of the same E type.
- L3: two or more different E types explicitly composed.
- L4: an abstract instruction that requires memory, visual semantics, pattern,
  logic, common-sense, or constraint reasoning.

Free-language L1-L3 generation uses two structured model calls. The first sees
only the instruction and E1-E9 catalog and emits typed steps whose scene
selectors are open natural-language references. The second sees those
references plus a coordinate-free semantic inventory and may return only
existing scene UIDs, status, and confidence. Local validation enforces complete
request coverage, candidate roles, cardinality, confidence, and non-self
targets; unresolved or ambiguous references fail instead of being guessed.
Each structured model stage gets at most one bounded repair attempt after an
invalid response. If repair still fails, or the model call itself fails,
generation stops before recipe expansion and artifact publication. It never
switches to keyword or rule-based instruction parsing.

The older public `planning.plan_task` adapter still produces a standalone
`TaskAgent` from structured LLM output, but it does not reinterpret the
instruction after that output exists. Axis, orientation, and arm-allocation
fields come only from the validated model result. It has no keyword fallback.

Generator callers without a configured LLM must provide a validated `TaskSpec`
with explicit role bindings (or a matching `SceneRequirements` sidecar). This
path is fully offline and never imports or calls an LLM client.

### SceneRequirements

`SceneRequirements` is the JSON hand-off to the external Scene Engine. It
declares object roles, counts, categories, affordances, initial states, spatial
constraints, camera requirements, and distractors. Scene results are never
silently repaired. Structural contradictions and explicit affordance
contradictions invalidate the task instance; an absent affordance declaration
remains unknown and is deferred to runtime physical validation.

The current tabletop importer has one deliberately narrow structural contract:
exactly one `background` object is the support surface and receives runtime UID
`table`; every movable object is assumed to begin on that surface. Zero or
multiple backgrounds are rejected rather than resolved from position, UID, or
description text. Semantic `category`, `color`, and `attributes` come only from
their explicit scene fields. Physics `attrs` are not semantic metadata.

For task-first inputs, explicit role bindings are authoritative unless they
contradict metadata that the scene actually declares. Automatic role binding
requires a unique match with complete structured category, attribute, state,
and affordance evidence. Object names and descriptions remain available to the
LLM grounding call, but deterministic validation never searches them for
semantic substrings.

### StaticSceneManifest And FeasibilityReport

`StaticSceneManifest` is an additive Scene Bridge artifact. It keeps the legacy
scene manifest stable while recording initial poses, geometry hashes, physics,
articulation payloads, structured affordance evidence, and provenance. Legacy
affordance strings become `declared` evidence; only structural facts derived by
the adapter are marked `verified`.

`FeasibilityReport` classifies each check as `proven`, `runtime_probe`, `unknown`,
or `contradicted`. Missing evidence remains unknown. Declared geometric
affordances require a runtime probe, and unavailable AtomicActions are explicit
contradictions. A contradicted report publishes an `infeasible` audit result and
does not invoke graph or bundle generation. Executable preflight remains a
second authoritative gate before bundle generation.

Scene Bridge reports arm-layout and whole-task pickup, handover,
target-interaction, and safety-clearance phases as `runtime_probe` evidence.
Arm-side compatibility is not claimed without live arm-base poses and workspace
geometry.

### SeedGraph

Every node directly names an `atomic_action`, scene `object_uid`, symbolic
`target_binding`, actor, control, dependencies, resources, pre/postconditions,
motion policy, E type, `task_instance_id`, and an Action Contract v2
`failure_policy`. `task_required` and `safety_required` failures invalidate the
candidate; `best_effort` failures remain observable but do not erase an already
verified task and safety result. `TaskGroup` groups all nodes of one E instance
with `role=primary|recovery`; it is metadata over the same DAG, not a second
graph.

Validation guarantees:

- node and TaskGroup dependencies are DAGs;
- every node belongs to exactly one TaskGroup;
- E groups contain their required core actions;
- concurrent nodes do not claim the same exclusive arm/object resource;
- object references resolve to scene UIDs;
- world poses, qpos, trajectories, grasp poses, and waypoints are rejected
  recursively;
- hashes use canonical strict JSON and are stable across processes.

The production loader accepts v3 graphs only. An older graph, whether supplied as
JSON or an in-memory mapping, receives an explicit regeneration error rather
than an implicit migration.

## Capability Boundary

`AtomicCapabilityRegistry` is the single runtime catalog. A descriptor declares
the action/option types, accepted symbolic bindings and controls, resource mode,
held-object state effect, target and config materializers, verifier, failure
classifier, retry mode, and runtime availability.

The executable catalog currently contains:

- `PickUp`, `MoveHeldObject`, `MoveEndEffector`, `MoveJoints`, and `Place`
- `Pour` and `Press`
- `CoordinatedPickment` and `CoordinatedPlacement`
- `HandOver`
- `Slide`, for opening or closing a live prismatic joint
- `OpenDoor`, for opening a revolute door hinge by its handle
- `Twist`, with explicit ordinal joint settings

Articulation grounding reads live joint types, link geometry, limits, and
positions. Zero-nearest joint endpoints represent closed or inactive states
for generated binary mechanisms. The Scene Engine source adapter records each
unambiguous `Slide`, `OpenDoor`, `Press`, or `Twist` joint together with its USD
`body1` link in `agent_config.articulation_interaction_links`. Contact-driven
`Slide` and `OpenDoor` targets additionally own one exact grasp mesh when USD
authoring provides an unambiguous handle. Runtime grounding uses that exact
joint, live link pose, and handle geometry, while old bundles retain the
name-based single-joint fallback. For generated USD drawers, GenSim samples the
exact handle and the complete scaled articulation in the moving-link frame and
passes both point clouds through `ObjectSemantics`, allowing
`SlideAffordance.resolve_from_object_geometry()` to own axis inference just as
the direct Atomic Action tutorial does. The adapter also declares a three-position ordinal
calibration only for unambiguous generated `knob`/`dial`/`rotary` joints, using
the USD-authored revolute limits; authored `joint_settings` always win, and
multiple calibrated joints remain an explicit ambiguity. GenSim converts
authored prismatic limits and USD link geometry to runtime body scale, reads USD
revolute limits in radians, and wraps equivalent live angles before invoking
`Slide`, `OpenDoor`, `Twist`, or `Press`. Idempotent articulation targets may be
accepted at entry. GenSim retains the tutorial `Slide` sampling and
hand-interpolation budgets and inserts an explicit close-pose settle segment so
simulator drives can converge before the drawer interaction starts.

New E6/E7 recipes declare `external_cleanup` on the interaction binding.
GenSim executes the planned prefix before the internal hand-open/release
segment. Cleanup runs detach, disengage, retreat, full-open, then required-home.
Detach stops further opening after three fresh control-step observations show
no target or obstacle contact and actual opening progress. It preserves the
partial opening during withdrawal instead of requiring full opening at the
original grasp pose. Raw hand observations remain separate from legal
master/mimic hold commands. A clear-at-entry hand is recorded separately
from observed contact release. Unknown or saturated contact buffers cannot prove
release. Full opening remains a separate verified action after withdrawal.
Hand-only detach and full-open keep the selected arm's existing controller
position target. They must not re-anchor that target to measured joint positions
and accumulate servo tracking error at each phase transition. PlanningContext
continues to contain the actual observation. Release support checks combine the
hand sweep with sampled observed-to-commanded arm FK poses before those arm-hold
commands are emitted. Ordinary arm motions and abort-time measured holds retain
their original behavior.
When another environment continues, a completed detach row freezes its existing
arm command and the first observed master's legal hand command rather than
re-anchoring them on every tick. A later hazard replaces that row's hold with its
first measured abort state; later clear observations cannot resume its normal
hold. The default stop behavior outside staged detachment is unchanged.
Internal release, retract, and push-return segments are recorded
as deferred, not executed. The original planner success mask still covers the
complete core plan; truncation never turns a failed plan into a successful one.
Bundles without this declaration retain their existing execution behavior.

The selected grasp request supplies the approach vector, saved in the owning
link frame independently of gripper roll. After that interaction executes,
detach checks inverse approach, tool-back, same-arm baseward, and world-up
corridors without shortening the configured distance. Each
candidate uses current-to-commanded hand geometry, dense target-link and table
clearance samples, and an RNG-isolated same-arm endpoint IK probe. Distance and
IK eligibility remain separate evidence; positive clearance alone cannot permit
withdrawal. Each environment selects the feasible candidate with the largest
minimum path clearance. Within a 1 micrometre numerical tie, larger terminal
clearance wins; an exact tie retains the listed candidate order. Terminal margin
never bypasses the path-clearance or endpoint-IK gates.
Disengagement entry fixes the selected world-space endpoint and wrist orientation
per task step, arm, and environment row. Retries replan to that same endpoint.
Reset clears this provenance, choices, hand holds, and endpoints. Legacy full-open
cleanup retains its inverse-approach route. Retreat keeps
the verified detached posture; full-open must pass before home. Core-terminal
joint observations are frozen before any cleanup. A physical core failure is
not blindly retried or erased by later cleanup, but actual execution provenance
still permits safety-required cleanup. Final predicates must also remain true.
The OpenDoor progress gate treats the first open-segment sample as the unchanged
starting configuration supplied by MotionGenerator, and the last as the full
requested hinge target. It must not wait for nonzero hinge progress while
repeatedly issuing the unchanged starting command. This preserves the original
progress tolerance and maximum repeat budget.
E6/E7 cores with external cleanup also observe robot/world contacts after every
control command, including commands repeated by the progress gate. Selected-hand
contact with the named moving link is allowed; other world contact or unknown
contact data latches a row-local failure. The first contact evidence is retained
even if later observations clear. Core stops flush one final robot command with
measured holds on aborted rows, including when peers finish normally, so the last
unsafe drive target is not left active. Invalid measured
holds are rejected instead of dispatched. Non-gated cleanup keeps its existing
stop semantics, and stopped batch rows stay held while peers continue.
Core records retain both joint-motion success and safety-aborted state; meeting
the joint target cannot turn a collision abort into success. Failure reports
separate contact-aborted rows from other joint-postcondition failures. These are
reactive last-substep observations, not continuous or predictive collision
certification, and a safely stopped task is still incomplete.

Before E6/E7 execution, GenSim checks each USD target's live per-link gravity
flags against its explicit `cfg.enable_gravity`. Physical-attribute construction
can overwrite that flag with a backend default. Only mismatching environment
rows are restored and read back; a failed readback stops execution. This applies
the existing generated configuration, not a new zero-gravity policy, joint latch,
qpos injection, or friction change. Per-action traces retain this run-entry
before/after snapshot; they are not repeated live gravity measurements.
The environment also restores configured gravity for selected reset rows before
its existing generated-USD reset pass. All relevant articulations are corrected
before any native reset, since a native reset advances the whole physics world.
Restoring gravity only at executor entry leaves velocity imparted by that reset
step. This ordering change adds no qpos, velocity, force, or target injection;
the existing initial-state reset still owns clearing dynamics. Runtime evidence
separately retains the reset-time restoration and the executor-entry readback.

Cleanup geometry does not alter core grasp selection. Each release samples the
observed hand configuration toward the requested opening at the hand action's
sampling resolution. Detach may use a collision-free prefix; full-open may not
substitute a partial opening. Withdrawal audits the observed-to-commanded hand
tracking envelope along actual planned FK poses, never a fictitious full closure.
Geometry is environment-local and is not cached across changing hand states.
Fallback and reachability-search candidates cannot bypass the support audit.
The audit only removes successful rows and records its scope and clearance.
Live contact, hand tracking, and articulation target-retention guards stop unsafe
cleanup rows without converting their failure into a task success.

Generated USD OpenDoor has separate core-contact checks, independent of the
external-cleanup flag. Approach/reach checks exclude the interaction joint's
moving subtree; the closing sweep excludes only the exact handle submesh and
checks the remaining door, other articulation links, and explicit support plane.
Both audits retain the raw planner mask and can only reject successful rows.
Each attempted OpenDoor candidate records its pre-audit trajectory and segments,
including a null trajectory when the planner supplies none. Failed raw rows
must not be interpreted as valid requested motion merely because a hold-filled
trajectory exists. Within the configured candidate budget, OpenDoor prioritizes
unmodified sampler roll when available and tries paired existing depth variants;
Slide retains its original ordering. These core-grasp choices require independent
physical qualification and are not evidence that cleanup succeeds.
For the existing horizontal-hinge wrist-roll adaptation, TCP depth offsets must
not move the sampled handle contact off the rigid door arc. The invocation-local
grasp selector supplies a detached TCP-frame contact offset independently of its
diagnostic trace, and clears it when the selection context exits. After reducing
wrist rotation, GenSim adjusts translation by `(R_rigid - R_relaxed) * offset`.
The door-link poses, target angle, roll fraction, and vertical-hinge behavior are
unchanged. This preserves the selected anchor geometrically, not a guaranteed
physical grasp; it still requires live contact and joint-motion validation.
When a validated seed graph actually binds OpenDoor to a generated USD object
with Robotiq, generation raises the robot's minimum numerical solver budget to
32 position / 8 velocity iterations, preserving higher template values. The
resolved RobotCfg stores this policy explicitly. It changes neither drive gains
nor contact/verifier tolerances, and is independent of external cleanup. Other
grippers, URDF doors, and graphs without that OpenDoor binding retain their
existing budget. Because solver iterations are articulation-wide, all robot
motions in a mixed episode containing this binding use the resolved budget;
they are not a per-joint or per-waypoint setting. This numerical qualification
does not prove grasping or complete E7 success.

A failed CLI run with zero commands and explicitly failed environment rows in
every report preserves the original failure without attempting to archive an
unproduced video. Any success, mixed or unknown state, or nonzero command count
still requires the fresh-video gate. Missing media never permits copying a stale
recording or turning a failed execution into success.

These checks do not certify collision-free withdrawal: the discrete hand samples
and trajectory waypoints are not continuous collision detection, visual meshes
are not necessarily backend collision hulls, and the support check does not
cover the full arm. The separate prospective corridor check covers all target
articulation links and the table, but endpoint IK does not certify an entire
arm trajectory. Contact guards observe the last physics
substep of each control step, not continuous collision-free motion. The current `ik_interp` path has no complete
world-collision validation, and the clearance verifier measures TCP endpoint
error and root distance. In particular,
the inverse-approach candidate can point toward a tabletop for a drop-down door.
Alternative directions are explicitly traced, geometrically checked candidates,
not a universally safe reversal of the interaction path.

Adding an executable skill consists of registering its descriptor and reusable
materializer/verifier hooks plus focused tests. Planner and executor dispatch
do not maintain a parallel action-class table.

## Offline And Online Planning

Offline recipes deterministically instantiate E1-E9 task instances. Current
task mappings are:

- `place_relative -> E1`
- `orient_object -> E2`
- `coordinated_transport -> E5`
- every member of `build_stack` and `arrange_line` -> one E1 instance

E5 uses `coordinated_transport` only as the semantic task-group operator. Its
motion graph contains one `CoordinatedPickment`; a `place` terminal behavior
adds synchronized left/right `MoveJoints(gripper_open)` nodes. The executor
clears coordinated hold state only after both grippers are observed open.

The online path first extracts auditable visual facts from multi-view RGB and,
when available, depth and camera calibration. Facts contain only known UIDs,
normalized bboxes/keypoints, canonical spatial relations, task predicates, and
confidence. Spatial relations use a shared ontology and fixed participant
order. Task-level judgments such as visual or pattern completion are accepted
only when the current `TaskSpec.success` explicitly requests them. A second
structured call produces a complete direct `AtomicAction` graph. Prompts
request facts and graph JSON only; hidden chain-of-thought is neither requested
nor stored.

Image-space constraints may use normalized keypoints, masks, bboxes, and
relative relations. The Grounder uses live depth and camera calibration to
convert them to world targets. The SeedGraph never stores that result.

## JIT Grounding

Each action is grounded again immediately before planning/execution. Grounding
therefore observes object displacement, current qpos, current held-object
ownership, articulation state, and fresh camera measurements. Coordinated and
handover actions are grounded as synchronized execution units. Automatic arm
selection, collision checks, live arrangement slots, and current predicate
semantics remain deterministic runtime responsibilities.

Arm allocation uses the live right-to-left arm-base axis and the live table
center. The preference therefore follows translated and rotated robot
workspaces; live motion planning remains authoritative for reachability.

Placement support is a relation, not an entity category. Static adaptation
accepts rigid `physical_entity` targets without requiring a `support_surface`
affordance. Articulations require a link-level runtime target interface and are
rejected until that interface is available. Runtime evaluates
`object_supported_by(payload, support, pose)` from live geometry and center of
mass, applies the requested `orientation_goal`, and requires low motion across a
bounded stability window. Successful relations form a per-environment support
graph that is checked for cycles and revalidated at task completion.

Orientation is compiled into hard `align_axis` or `match_rotation` terms plus a
separate minimum-rotation planning preference. An omitted orientation request
adds no hard acceptance term; `preserve` remains an explicit full-rotation
contract for persisted bundles, while `upright` constrains only the requested
local axis and declares whether that axis is directed. Grounding and runtime
verification consume the same compiled contract so reachability search cannot
silently relax a required terminal orientation. With no hard term, a live
upright state may still select upright-preserving yaw candidates as a planning
preference; this follows current state and automatically stops after that state
is invalidated rather than becoming a sticky success requirement.

Grasp generation keeps support-plane collision filtering as its strict first
pass. If diagnostics show that this heuristic alone exhausted otherwise
object-collision-free candidates, Action Engine retries without the heuristic;
the relaxed candidates still pass through the live robot and scene collision
planner before execution. This avoids treating a local support-plane proxy as
a proof of scene-level infeasibility, including for objects already held above
the support surface.

Grounding samples bounded support-relative placement poses. Planning failures
try the next pose before release; instability after release requires a fresh
grasp and an unused pose. The recovery keeps the original actor contract, and
its edges and failure provenance are recorded separately from the primary
attempt.

## Mainline Planning Contract

The runtime keeps only an Action Engine-local `ExecutionState` for full-robot
qpos and held-object relations. Each plan converts that state to the mainline
`PlanningContext` (`RobotObservation`, `TaskState`, and `SceneSnapshot`) and
submits an `ActionInvocation` to `AtomicActionEngine`. The returned
`StateDelta` remains speculative until physical and semantic verification; only
verified vectorized rows are committed.

`AtomicActionAdapter` accepts a shared `SceneProvider` and otherwise builds a
`RigidObjectSceneProvider` from live simulation entities. Planning snapshots now
carry monotonic timestamps and material-change scene/collision revisions. The
adapter also exposes `start_session(...)` for callers adopting
`ExecutionSession`; the existing compound and per-arm merged trajectory
scheduler remains as the compatibility execution path.

Single-arm arm motion uses cuRobo `motion_gen` by default. Hand-only and
coordinated dual-arm actions use `ik_interp`, because mainline coordinated
primitives do not support cuRobo motion generation. A failed single-arm cuRobo
row may fall back to `ik_interp` without replacing successful rows. Generated
background objects form the static cuRobo collision world; dynamic obstacles
are an explicit runtime-policy opt-in.

Generated mesh objects carry V-HACD settings in both the current shape-level
schema and legacy top-level fields. Before antipodal grasp construction, the
runtime prepares a checksummed V-HACD payload at the shared collision-checker
cache path so the unchanged mainline checker does not silently recompute CoACD.
Grasp generation samples multiple deviated approach directions and filters them
through the existing gripper collision model. Safety retreat planning searches
a bounded set of live height and baseward targets instead of treating one exact
height as a geometric reachability certificate.

## A/B Evaluation

Test mode retains both candidates. `run_strict_ab` creates distinct offline and
online environments with the same task, scene configuration, seed, Grounder,
verifiers, and retry policy. Both environments reset before execution and a
digest over robot qpos and object state must match exactly; a mismatch aborts
before either branch executes.

Artifacts are written under `offline/` and `online/`, with a shared
`comparison.json`. The comparison records graph hashes/differences, action and
path lengths, success, retries, recoveries, revisions, latency, record paths,
and planner/VLM metadata supplied by each candidate.

L4 A/B runs must supply a private-oracle evaluator. The built-in evaluator
checks memory reconstruction, visual completion, pattern completion, numeric
selection, functional placement, and stable/unobstructed goals from the final
state only. The comparison labels whether success came from runtime step
postconditions or the private oracle.

## Dynamic Recovery

The persisted `SeedGraph` is immutable. `RuntimeGraph` keeps a detached working
copy and an ordered revision log. One failed `AtomicAction` can be freshly
grounded and retried twice, for three total attempts, and only while its live
precondition remains true.

Failures use the bounded taxonomy `search_exhausted`, `plan_failed`,
`grasp_missed`, `object_fallen`, `object_dropped`, and
`postcondition_failed`. `search_exhausted` records the blocking edge, planning
stage, strategy, finite budget, and observed evidence; it does not claim that a
target is geometrically unreachable. Known recoverable states can insert a
complete `role=recovery` TaskGroup, such as an E2 upright group. Recovery keeps
the failed TaskGroup's actor contract, and primary, recovery, and replay events
are recorded separately. After recovery, the selected route replans only the
unfinished suffix. Offline and online dynamic replanners are explicit,
separate modes. Revision, recovery-action, transition, and retry budgets bound
every loop.

## Selection And Fusion

Product mode statically scores offline and online candidates using schema
validity, capability availability, UID validity, task coverage, visual
confidence, and estimated action cost. Exact mature-template matches favor the
offline route; L4 visual tasks favor sufficiently confident online results.

Fusion is conservative. It may choose only complete `TaskGroup` units, rewires
dependencies at group boundaries, and rejects unordered state changes to the
same object. It never splits one E instance across candidates.

## Artifacts

A normal generated bundle contains:

- `task_spec.json`
- `scene_requirements.json`
- `seed_task_graph.json`
- `seed_task_graph.png`
- `agent_config.json`
- `fast_gym_config.json`

Strict A/B adds branch-local graph/result artifacts and `comparison.json`.
Review graphs, runtime records, and videos never become execution inputs.

`prepare` lowers and preflights resolved semantic candidates in selection order.
A candidate-local lowering, symbolic planning, or preflight error rejects only
that candidate. If no resolved candidate is executable, the transaction
publishes `preparation_failure.json` with each attempted draft, verified
bindings, available grounded plan, failure stage, and exception instead of
leaving an older successful bundle in place.

## Invariants

- SeedGraph nodes are direct AtomicActions, not E-level operators.
- Lowering uses original instruction-step order as the stable tie-break among
  dependency-ready steps. Independent steps remain independent; the contract
  linker serializes only actual resource conflicts.
- E labels are subgraph grouping semantics only.
- Planning artifacts contain no grounded motion coordinates.
- Online planning never receives the private oracle.
- Runtime uses one capability registry for preflight, Grounding, config
  construction, execution, verification policy, and recovery policy.
- Required arms are never silently replaced.
- Failed or inactive vectorized rows preserve their last valid state.
- Current five task families preserve their v1 AtomicAction topology and live
  Grounding behavior after regeneration.
