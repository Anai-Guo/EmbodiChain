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

"""Candidate-level GenSim checks of commanded OpenDoor pre-contact geometry."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import actions
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.models import GroundedAction
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    ObjectSemantics,
    OpenDoorAffordance,
    OpenDoorGoal,
    PlannerDiagnostics,
    PlanningFailure,
    RecoveryPolicy,
    SlideAffordance,
    SlideGoal,
    TimedCommandSequence,
    TimedTrajectory,
    TrackingPolicy,
    TrajectorySegment,
)

from .test_actions import _FakeEngine, _commands_for
from .test_staged_articulation_release import _HAND_IDS, _env, _hand


def _plan(env, *, success=(True, True), x_offset=0.0):
    positions = env.robot.qpos[:, None].repeat(1, 8, 1)
    positions[:, :, 0] = torch.linspace(0.0, 0.35, 8) + x_offset
    positions[:, 4, [0, 2]] = positions[:, 3, [0, 2]]
    positions[:, :4, _HAND_IDS] = _hand(0.4)
    positions[:, 4:, _HAND_IDS] = _hand(0.7)
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=torch.arange(env.num_envs), step_dt=0.01
    )
    return ActionPlan(
        skill_id="open_door",
        plan_success=torch.tensor(success),
        commands=_commands_for(trajectory),
        joint_trajectory=trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=0,
        planned_collision_world_revision=(0,) * env.num_envs,
        diagnostics=PlannerDiagnostics(
            backend="fake",
            failure=None if all(success) else PlanningFailure("fixture_ik_failure"),
            metadata={
                "existing_evidence": "kept",
                "gensim_open_door_segment_planning": [
                    {"phase": "open", "success": torch.tensor(success)}
                ],
            },
        ),
        segments=(
            TrajectorySegment("approach", 0, 2),
            TrajectorySegment("reach", 2, 4),
            TrajectorySegment("close", 4, 5),
            TrajectorySegment("open", 5, 6),
            TrajectorySegment("release", 6, 7),
            TrajectorySegment("retract", 7, 8),
        ),
    )


def _grounded():
    affordance = OpenDoorAffordance(
        mesh_vertices=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]]),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=(0.0, 0.0, 0.0),
        joint_name="door_hinge",
        joint_limits=(0.0, 1.5),
    )
    return GroundedAction(
        action_class="OpenDoor",
        arm="left_arm",
        control="arm",
        target=OpenDoorGoal(
            ObjectSemantics(affordance=affordance, geometry={}, entity_id="door:door"),
            torch.eye(4).repeat(2, 1, 1),
            open_fraction=1.0,
        ),
        cfg={},
        motion_policy={
            "articulation_target_link_name": "door",
            "articulation_target_mesh_name": "handle",
            "articulation_external_cleanup": True,
            "interaction_grasp_candidate_count": 2,
        },
        object_uid="door",
    )


def _adapter(monkeypatch, *, asset="door.usdc"):
    env = _env()
    entity = SimpleNamespace(uid="door", cfg=SimpleNamespace(fpath=asset))
    env.sim.get_articulation = lambda uid: entity if uid == "door" else None
    adapter = AtomicActionAdapter(env)
    points = torch.tensor([[[[0.0, 0.0, 0.0], [0.03, 0.0, -0.06]]]]).repeat(2, 1, 1, 1)
    geometry = Mock(return_value=points)
    monkeypatch.setattr(actions, "_gripper_points", geometry)

    def fk(positions, control_part):
        assert control_part == "left_arm"
        poses = torch.eye(4).repeat(positions.shape[0], positions.shape[1], 1, 1)
        poses[:, :, 0, 3] = positions[:, :, 0]
        return poses, torch.empty(0)

    fk_spy = Mock(side_effect=fk)
    monkeypatch.setattr(adapter, "_arm_trajectory_fk", fk_spy)

    def clearance(grounded, poses, hand):
        assert grounded.object_uid == "door"
        assert poses.shape == (2, 4, 4, 4)
        torch.testing.assert_close(hand, points.flatten(1, 2))
        return (torch.abs(poses[:, :, 0, 3] - 0.1) - 0.005).amin(1)

    checker = Mock(side_effect=clearance)
    monkeypatch.setattr(
        adapter, "_open_door_precontact_clearance", checker, raising=False
    )
    monkeypatch.setattr(
        adapter,
        "_open_door_closure_clearance",
        Mock(return_value=torch.full((2,), 0.02)),
        raising=False,
    )
    return env, adapter, geometry, fk_spy, checker


def test_closure_audit_rejects_door_frame_contact_without_promoting_raw_failure(
    monkeypatch,
):
    env, adapter, _, _, _ = _adapter(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_open_door_closure_clearance",
        Mock(return_value=torch.tensor([-0.001, 0.01])),
    )
    plan = _plan(env)
    result = adapter._audit_open_door_closure(
        _grounded(), adapter.capabilities.get("OpenDoor"), plan
    )
    assert result.plan_success.tolist() == [False, True]
    assert result.diagnostics.failure.code == "open_door_grasp_closure_collision"
    failed = _plan(env, success=(False, False))
    assert (
        adapter._audit_open_door_closure(
            _grounded(), adapter.capabilities.get("OpenDoor"), failed
        )
        is failed
    )
    assert plan.plan_success.tolist() == [True, True]


def test_middle_fk_collision_rejects_candidate_with_safe_prefix_endpoints(monkeypatch):
    env, adapter, geometry, fk, checker = _adapter(monkeypatch)
    plan = _plan(env)
    before = env.robot.qpos.clone()
    grounded = _grounded()

    result = adapter._audit_open_door_precontact(
        grounded, adapter.capabilities.get("OpenDoor"), plan
    )

    assert result.plan_success.tolist() == [False, False]
    assert plan.plan_success.tolist() == [True, True]
    assert result.diagnostics.failure.code == "open_door_precontact_collision"
    trace = result.diagnostics.metadata["open_door_precontact"]
    assert trace["scope"] == "commanded_gripper_vs_non_target_articulation_links"
    assert trace["stage"] == "approach_reach"
    assert trace["world_collision_checked"] is False
    assert trace["minimum_clearance"] == pytest.approx(0.003)
    torch.testing.assert_close(
        trace["clearance"], torch.full((2,), -0.005), atol=1e-7, rtol=0
    )
    assert result.diagnostics.metadata["existing_evidence"] == "kept"
    geometry.assert_called_once()
    args, kwargs = geometry.call_args
    assert args == (env, "left_arm")
    assert set(kwargs) == {"qpos"}
    torch.testing.assert_close(kwargs["qpos"], plan.joint_trajectory.positions[:, 0])
    torch.testing.assert_close(trace["hand_qpos"], _hand([0.4, 0.4]))
    torch.testing.assert_close(env.robot.qpos, before)
    torch.testing.assert_close(
        fk.call_args.args[0], plan.joint_trajectory.positions[:, :4]
    )
    assert checker.call_count == 1


def test_safe_clearance_does_not_promote_a_preexisting_planner_failure(monkeypatch):
    env, adapter, _, _, checker = _adapter(monkeypatch)
    checker.side_effect = None
    checker.return_value = torch.tensor([0.02, 0.02])
    plan = _plan(env, success=(True, False))

    result = adapter._audit_open_door_precontact(
        _grounded(), adapter.capabilities.get("OpenDoor"), plan
    )

    assert result.plan_success.tolist() == [True, False]
    trace = result.diagnostics.metadata["open_door_precontact"]
    assert torch.as_tensor(trace["raw_plan_success"]).tolist() == [True, False]
    assert torch.as_tensor(trace["success"]).tolist() == [True, False]
    assert result.diagnostics.failure.code == "fixture_ik_failure"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.001])
def test_nonfinite_or_colliding_clearance_rejects_only_affected_environment(
    monkeypatch, invalid
):
    env, adapter, _, _, checker = _adapter(monkeypatch)
    checker.side_effect = None
    checker.return_value = torch.tensor([invalid, 0.02])

    result = adapter._audit_open_door_precontact(
        _grounded(), adapter.capabilities.get("OpenDoor"), _plan(env)
    )

    assert result.plan_success.tolist() == [False, True]


def test_varying_preshape_is_rejected_before_using_one_cached_geometry(monkeypatch):
    env, adapter, geometry, fk, checker = _adapter(monkeypatch)
    plan = _plan(env)
    plan.joint_trajectory.positions[:, 2, _HAND_IDS] = _hand(0.5)

    with pytest.raises(ValueError, match="constant.*pre-shape"):
        adapter._audit_open_door_precontact(
            _grounded(), adapter.capabilities.get("OpenDoor"), plan
        )
    geometry.assert_not_called()
    fk.assert_not_called()
    checker.assert_not_called()


@pytest.mark.parametrize("unaffected", ["slide", "urdf", "failed_plan"])
def test_e6_urdf_and_failed_plans_remain_unchanged(monkeypatch, unaffected):
    env, adapter, geometry, fk, checker = _adapter(
        monkeypatch, asset="door.urdf" if unaffected == "urdf" else "door.usdc"
    )
    plan = _plan(
        env, success=(False, False) if unaffected == "failed_plan" else (True, True)
    )
    if unaffected == "failed_plan":
        plan = replace(
            plan,
            joint_trajectory=TimedTrajectory.empty(
                batch_size=2,
                robot_dof=env.robot.dof,
                device="cpu",
                env_ids=torch.arange(2),
            ),
            commands=TimedCommandSequence(frames=(), env_ids=torch.arange(2)),
            segments=(),
        )
    capability = adapter.capabilities.get(
        "Slide" if unaffected == "slide" else "OpenDoor"
    )

    result = adapter._audit_open_door_precontact(_grounded(), capability, plan)

    assert result is plan
    geometry.assert_not_called()
    fk.assert_not_called()
    checker.assert_not_called()


def test_adapter_plan_rejects_colliding_rank_and_selects_next_safe_candidate(
    monkeypatch,
):
    env, adapter, geometry, _, checker = _adapter(monkeypatch)
    calls = []

    def engine_plan(invocation, context):
        calls.append(invocation)
        return _plan(env, x_offset=float(len(calls) - 1))

    engine = _FakeEngine(engine_plan)
    engine.grasp_pose_generators = {}
    monkeypatch.setattr(adapter, "_engine_for", lambda *_args: engine)
    outcome = adapter.plan(
        _grounded(), ExecutionState(last_qpos=env.robot.qpos.clone())
    )

    assert len(calls) == 2
    assert checker.call_count == 2
    assert geometry.call_count == 3
    assert geometry.call_args.kwargs["sample_count"] >= 5
    assert outcome.success.tolist() == [True, True]
    assert outcome.grounded.motion_policy["interaction_grasp_candidate_rank"] == 1
    attempts = outcome.planner_trace["interaction_grasp_search"]
    assert attempts[0]["plan_success"] == [False, False]
    assert attempts[1]["plan_success"] == [True, True]
    trace = outcome.planner_trace["primary_action_diagnostics"]["open_door_precontact"]
    assert torch.as_tensor(trace["success"]).tolist() == [True, True]
    torch.testing.assert_close(outcome.trajectory[:, 0, 0], torch.ones(2))


def test_adapter_plan_rejects_closure_collision_and_selects_next_candidate(
    monkeypatch,
):
    env, adapter, _, _, precontact = _adapter(monkeypatch)
    precontact.side_effect = None
    precontact.return_value = torch.full((2,), 0.02)
    closure = Mock(side_effect=[torch.full((2,), -0.001), torch.full((2,), 0.01)])
    monkeypatch.setattr(adapter, "_open_door_closure_clearance", closure)
    calls = []

    def engine_plan(invocation, context):
        calls.append(invocation)
        return _plan(env, x_offset=float(len(calls)))

    engine = _FakeEngine(engine_plan)
    engine.grasp_pose_generators = {}
    monkeypatch.setattr(adapter, "_engine_for", lambda *_args: engine)
    outcome = adapter.plan(
        _grounded(), ExecutionState(last_qpos=env.robot.qpos.clone())
    )

    assert len(calls) == 2
    assert outcome.success.tolist() == [True, True]
    attempts = outcome.planner_trace["interaction_grasp_search"]
    assert torch.as_tensor(attempts[0]["grasp_closure"]["success"]).tolist() == [
        False,
        False,
    ]
    assert torch.as_tensor(attempts[1]["grasp_closure"]["success"]).tolist() == [
        True,
        True,
    ]


@pytest.mark.parametrize("missing_failed_trajectory", [False, True])
def test_every_open_door_attempt_retains_detached_pre_audit_planner_evidence(
    monkeypatch, missing_failed_trajectory
):
    env, adapter, _, _, _ = _adapter(monkeypatch)
    plans = []
    expected_positions = []
    expected_segments = []

    def engine_plan(invocation, context):
        index = len(plans)
        plan = _plan(
            env,
            success=(False, False) if index == 1 else (True, True),
            x_offset=float(index),
        )
        if index == 1 and missing_failed_trajectory:
            plan = replace(
                plan,
                joint_trajectory=None,
                commands=TimedCommandSequence(frames=(), env_ids=torch.arange(2)),
                segments=(),
            )
        plans.append(plan)
        expected_positions.append(
            None
            if plan.joint_trajectory is None
            else plan.joint_trajectory.positions.clone()
        )
        expected_segments.append(
            {
                segment.name: {"start": segment.start, "stop": segment.stop}
                for segment in plan.segments
            }
        )
        return plan

    engine = _FakeEngine(engine_plan)
    engine.grasp_pose_generators = {}
    monkeypatch.setattr(adapter, "_engine_for", lambda *_args: engine)
    grounded = _grounded()
    grounded.motion_policy["interaction_grasp_candidate_count"] = 3

    outcome = adapter.plan(grounded, ExecutionState(last_qpos=env.robot.qpos.clone()))

    attempts = outcome.planner_trace["interaction_grasp_search"]
    assert len(attempts) == 3
    assert outcome.grounded.motion_policy["interaction_grasp_candidate_rank"] == 2
    assert [attempt["plan_success"] for attempt in attempts] == [
        [False, False],
        [False, False],
        [True, True],
    ]
    for index, attempt in enumerate(attempts):
        expected_success = [False, False] if index == 1 else [True, True]
        assert torch.as_tensor(attempt["raw_plan_success"]).tolist() == expected_success
        assert attempt["segment_planning"][0]["success"].tolist() == expected_success
        assert attempt["proposed_segments"] == expected_segments[index]
        if expected_positions[index] is None:
            assert attempt["proposed_trajectory"] is None
        else:
            assert isinstance(attempt["proposed_trajectory"], torch.Tensor)
            torch.testing.assert_close(
                attempt["proposed_trajectory"], expected_positions[index]
            )
            assert attempt["proposed_trajectory"].shape[1] == 8
    assert outcome.trajectory.shape[1] == 6
    for plan in plans:
        plan.plan_success.zero_()
        plan.diagnostics.metadata["gensim_open_door_segment_planning"][0][
            "success"
        ].zero_()
        if plan.joint_trajectory is not None:
            plan.joint_trajectory.positions.fill_(-99.0)
    for index, attempt in enumerate(attempts):
        assert torch.as_tensor(attempt["raw_plan_success"]).tolist() == (
            [False, False] if index == 1 else [True, True]
        )
        assert attempt["segment_planning"][0]["success"].tolist() == (
            [False, False] if index == 1 else [True, True]
        )
        if expected_positions[index] is not None:
            torch.testing.assert_close(
                attempt["proposed_trajectory"], expected_positions[index]
            )


def test_slide_grasp_search_does_not_add_open_door_raw_trajectory_payload(monkeypatch):
    env, adapter, geometry, _, checker = _adapter(monkeypatch)
    plan = _plan(env)
    renamed = {"open": "pull", "release": "open", "retract": "return"}
    plan = replace(
        plan,
        skill_id="slide",
        segments=tuple(
            TrajectorySegment(
                renamed.get(segment.name, segment.name), segment.start, segment.stop
            )
            for segment in plan.segments
        ),
    )
    engine = _FakeEngine(lambda *_args: plan)
    engine.grasp_pose_generators = {}
    monkeypatch.setattr(adapter, "_engine_for", lambda *_args: engine)
    door = _grounded()
    mesh = door.target.semantics.affordance
    grounded = replace(
        door,
        action_class="Slide",
        target=SlideGoal(
            ObjectSemantics(
                affordance=SlideAffordance(
                    mesh_vertices=mesh.mesh_vertices, mesh_triangles=mesh.mesh_triangles
                ),
                geometry={},
                entity_id="door:door",
            ),
            torch.eye(4).repeat(2, 1, 1),
        ),
    )

    outcome = adapter.plan(grounded, ExecutionState(last_qpos=env.robot.qpos.clone()))

    assert outcome.success.tolist() == [True, True]
    attempts = outcome.planner_trace["interaction_grasp_search"]
    assert len(attempts) == 1
    assert not {
        "proposed_trajectory",
        "proposed_segments",
        "raw_plan_success",
    }.intersection(attempts[0])
    geometry.assert_not_called()
    checker.assert_not_called()
