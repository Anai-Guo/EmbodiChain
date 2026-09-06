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

"""Support-plane audits of real gripper linkage sweeps and planned TCP poses."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import actions as actions_module
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.models import GroundedAction
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    EndEffectorPoseGoal,
    PlannerDiagnostics,
    PlanningFailure,
    RecoveryPolicy,
    RuntimeCommandFrame,
    TimedCommandSequence,
    TimedTrajectory,
    TrackingPolicy,
    TrajectorySegment,
)


def _plan(
    success: tuple[bool, ...] = (True,),
    *,
    release_name: str = "release",
) -> ActionPlan:
    batch_size = len(success)
    positions = torch.arange(batch_size * 10, dtype=torch.float32).reshape(
        batch_size, 5, 2
    )
    trajectory = TimedTrajectory.from_uniform_step(
        positions,
        env_ids=torch.arange(batch_size),
        step_dt=0.01,
    )
    frames = tuple(
        RuntimeCommandFrame(
            commands=(),
            active_mask=torch.ones(batch_size, dtype=torch.bool),
            env_ids=trajectory.env_ids,
            hold_duration=trajectory.dt[:, index],
        )
        for index in range(trajectory.waypoint_count)
    )
    return ActionPlan(
        skill_id="support_audit_fixture",
        plan_success=torch.tensor(success),
        commands=TimedCommandSequence(frames=frames, env_ids=trajectory.env_ids),
        joint_trajectory=trajectory,
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=3,
        planned_collision_world_revision=(3,) * batch_size,
        diagnostics=PlannerDiagnostics(
            backend="ik_interp",
            failure=None if all(success) else PlanningFailure("fixture_planner_failed"),
            metadata={"existing_evidence": "preserved"},
        ),
        segments=(
            TrajectorySegment("approach", 0, 1),
            TrajectorySegment("interaction", 1, 3),
            TrajectorySegment(release_name, 3, 4),
            TrajectorySegment("retract", 4, 5),
        ),
    )


def _grounded(policy: dict[str, Any], *, action: str = "OpenDoor") -> GroundedAction:
    target = torch.eye(4).unsqueeze(0)
    target[:, 2, 3] = 2.0
    return GroundedAction(
        action_class=action,
        arm="left_arm",
        control="arm",
        target=EndEffectorPoseGoal(xpos=target),
        cfg={},
        motion_policy=policy,
        object_uid="articulation_fixture",
    )


def _adapter(fk_poses: torch.Tensor, points: torch.Tensor) -> AtomicActionAdapter:
    adapter = object.__new__(AtomicActionAdapter)
    adapter.device = torch.device("cpu")
    adapter.num_envs = fk_poses.shape[0]
    adapter.env = SimpleNamespace(device="cpu", num_envs=fk_poses.shape[0])
    adapter._parts = Mock(return_value=("physical_left_arm", "physical_hand", 1))

    def sample_fk(
        positions: torch.Tensor, control_part: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert control_part == "physical_left_arm"
        indices = (positions[:, :, 0].remainder(10) / 2).to(torch.int64)
        rows = torch.arange(positions.shape[0])[:, None]
        return fk_poses[rows, indices], torch.empty(0)

    adapter._arm_trajectory_fk = Mock(side_effect=sample_fk)
    assert points.ndim == 3 and points.shape[0] == fk_poses.shape[0]
    adapter._departure_hand_geometry = Mock(return_value=points)
    return adapter


def _poses(heights: list[list[float]]) -> torch.Tensor:
    poses = torch.eye(4).repeat(len(heights), len(heights[0]), 1, 1)
    poses[:, :, 2, 3] = torch.tensor(heights)
    return poses


def _trace(result: ActionPlan) -> dict[str, Any]:
    return result.diagnostics.metadata["articulation_support_audit"]


@pytest.mark.parametrize(
    ("action", "materializer", "release_name"),
    [("Slide", "slide", "open"), ("OpenDoor", "open_door", "release")],
)
def test_core_external_cleanup_does_not_audit_hypothetical_release_geometry(
    monkeypatch, action: str, materializer: str, release_name: str
) -> None:
    plan = _plan(release_name=release_name)
    adapter = _adapter(
        _poses([[1.2, 1.2, 1.08, 1.2, 1.2]]),
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -0.10]]]),
    )
    grounded = _grounded(
        {
            "articulation_external_cleanup": True,
            "interaction_support_surface_z": 1.0,
        },
        action=action,
    )

    gripper_points = Mock(
        side_effect=AssertionError("Core action must not sample a future release.")
    )
    monkeypatch.setattr(
        actions_module, "_gripper_points", gripper_points, raising=False
    )
    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer=materializer), plan
    )

    assert result is plan
    assert plan.plan_success.tolist() == [True]
    assert result.diagnostics.metadata["existing_evidence"] == "preserved"
    adapter._departure_hand_geometry.assert_not_called()
    adapter._arm_trajectory_fk.assert_not_called()
    gripper_points.assert_not_called()


def test_departure_uses_actual_open_hand_not_virtual_closed_sweep() -> None:
    plan = _plan()
    adapter = _adapter(
        _poses([[1.05] * 5]),
        torch.tensor([[[0.0, 0.0, -0.03]]]),
    )
    virtual_closed_points = torch.tensor([[[0.0, 0.0, -0.10]]])
    adapter._interaction_release_geometry = Mock(return_value=virtual_closed_points)
    grounded = _grounded(
        {
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
        },
        action="MoveEndEffector",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer="eef_pose"), plan
    )

    assert result.plan_success.tolist() == [True]
    assert torch.as_tensor(
        _trace(result)["observed_clearance"]
    ).tolist() == pytest.approx([0.02], abs=1.0e-6)
    adapter._departure_hand_geometry.assert_called_once_with(grounded)
    adapter._interaction_release_geometry.assert_not_called()


def test_departure_rotates_actual_linkage_points_before_clearance_query() -> None:
    poses = _poses([[1.2, 1.2, 1.05, 1.2, 1.2]])
    poses[0, 2, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    adapter = _adapter(poses, torch.tensor([[[0.10, 0.0, 0.0]]]))
    grounded = _grounded(
        {
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
        },
        action="MoveEndEffector",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer="eef_pose"), _plan()
    )

    assert result.plan_success.tolist() == [False]
    assert torch.as_tensor(
        _trace(result)["observed_clearance"]
    ).tolist() == pytest.approx([-0.05], abs=1.0e-6)


def test_departure_audits_interior_fk_waypoints_not_only_safe_endpoints() -> None:
    plan = _plan()
    adapter = _adapter(
        _poses([[1.2, 1.2, 1.05, 1.2, 1.2]]),
        torch.tensor([[[0.0, 0.0, -0.10]]]),
    )
    grounded = _grounded(
        {
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
        },
        action="MoveEndEffector",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer="eef_pose"), plan
    )

    assert result.plan_success.tolist() == [False]
    assert _trace(result)["scope"] == "departure_trajectory"
    assert torch.as_tensor(
        _trace(result)["observed_clearance"]
    ).tolist() == pytest.approx([-0.05], abs=1.0e-6)
    fk_args, fk_kwargs = adapter._arm_trajectory_fk.call_args
    assert not fk_kwargs
    torch.testing.assert_close(fk_args[0], plan.joint_trajectory.positions)
    assert fk_args[1] == "physical_left_arm"


@pytest.mark.parametrize(
    ("policy_minimum", "height", "expected_minimum"),
    [(-0.01, 1.002, 0.003), (0.02, 1.01, 0.02)],
)
def test_support_margin_has_safety_floor_and_honors_stricter_policy(
    policy_minimum: float, height: float, expected_minimum: float
) -> None:
    adapter = _adapter(
        _poses([[height] * 5]),
        torch.tensor([[[0.0, 0.0, 0.0]]]),
    )
    grounded = _grounded(
        {
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
            "interaction_grasp_minimum_support_clearance": policy_minimum,
        },
        action="MoveEndEffector",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer="eef_pose"), _plan()
    )

    assert result.plan_success.tolist() == [False]
    assert _trace(result)["minimum_clearance"] == pytest.approx(expected_minimum)


@pytest.mark.parametrize("raw_success", [(True, True), (True, False)])
def test_support_failure_masks_are_row_local_and_never_promote_planner_failure(
    raw_success: tuple[bool, bool],
) -> None:
    plan = _plan(raw_success)
    adapter = _adapter(
        _poses([[1.2] * 5, [1.2] * 5]),
        torch.tensor([[[0.0, 0.0, -0.3]], [[0.0, 0.0, 0.0]]]),
    )
    grounded = _grounded(
        {
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
        },
        action="MoveEndEffector",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer="eef_pose"), plan
    )

    expected = [False, raw_success[1]]
    assert result.plan_success.tolist() == expected
    assert plan.plan_success.tolist() == list(raw_success)
    assert torch.as_tensor(_trace(result)["raw_plan_success"]).tolist() == list(
        raw_success
    )
    assert torch.as_tensor(_trace(result)["success"]).tolist() == expected
    assert torch.as_tensor(
        _trace(result)["observed_clearance"]
    ).tolist() == pytest.approx([-0.1, 0.2], abs=1.0e-6)


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"interaction_support_surface_z": 1.0},
        {"articulation_external_cleanup": True},
        {"articulation_support_audit": True},
    ],
)
def test_unrequested_or_unconfigured_support_audit_preserves_original_plan(
    policy: dict[str, Any],
) -> None:
    plan = _plan()
    adapter = _adapter(
        _poses([[0.0] * 5]),
        torch.tensor([[[0.0, 0.0, -1.0]]]),
    )

    result = adapter._audit_interaction_support(
        _grounded(policy, action="MoveEndEffector"),
        SimpleNamespace(target_materializer="eef_pose"),
        plan,
    )

    assert result is plan
    adapter._departure_hand_geometry.assert_not_called()
    adapter._arm_trajectory_fk.assert_not_called()


@pytest.mark.parametrize("materializer", ["slide", "open_door", "eef_pose"])
def test_failed_empty_plan_preserves_planner_failure_without_geometry_audit(
    materializer: str,
) -> None:
    plan = _plan((False, False))
    plan = replace(
        plan,
        joint_trajectory=TimedTrajectory.empty(
            batch_size=2,
            robot_dof=2,
            device="cpu",
            env_ids=torch.arange(2),
        ),
        commands=TimedCommandSequence(frames=(), env_ids=torch.arange(2)),
        segments=(),
    )
    adapter = _adapter(
        _poses([[0.0] * 5, [0.0] * 5]),
        torch.tensor([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]]),
    )
    grounded = _grounded(
        {
            "articulation_external_cleanup": True,
            "articulation_support_audit": True,
            "interaction_support_surface_z": 1.0,
        },
        action="MoveEndEffector" if materializer == "eef_pose" else "OpenDoor",
    )

    result = adapter._audit_interaction_support(
        grounded, SimpleNamespace(target_materializer=materializer), plan
    )

    assert result is plan
    assert result.diagnostics.failure.code == "fixture_planner_failed"
    adapter._departure_hand_geometry.assert_not_called()
    adapter._arm_trajectory_fk.assert_not_called()
