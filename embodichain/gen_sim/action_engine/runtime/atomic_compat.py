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

"""Action Engine-specific adapters for mainline atomic actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    JointPositionGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    OpenDoor,
    OpenDoorGoal,
    OpenDoorOptions,
    PlanningContext,
    ResolvedActionRequest,
    StateDelta,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix

__all__ = [
    "ActionEngineMoveJoints",
    "ActionEngineMoveJointsOptions",
    "ExactTargetMoveHeldObject",
    "ExactTargetMoveHeldObjectOptions",
]


_HORIZONTAL_HINGE_MAX_VERTICAL_ALIGNMENT = 0.25
_HORIZONTAL_HINGE_ROLL_FRACTION = 0.25


def _relax_horizontal_hinge_grasp_roll(
    link_pose: torch.Tensor,
    grasp_xpos: torch.Tensor,
    rotation_axis: torch.Tensor,
    hinge_rotation: torch.Tensor,
    opened_eef_poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the door arc while using axial freedom of a drop-down handle."""
    axis = torch.as_tensor(
        rotation_axis,
        dtype=link_pose.dtype,
        device=link_pose.device,
    )
    axis = axis / torch.linalg.vector_norm(axis)
    world_axis = torch.matmul(link_pose[:, :3, :3], axis)
    relaxed = torch.abs(world_axis[:, 2]) <= _HORIZONTAL_HINGE_MAX_VERTICAL_ALIGNMENT
    result = opened_eef_poses.clone()
    waypoint_count = opened_eef_poses.shape[1]
    fractions = torch.linspace(
        1.0 / waypoint_count,
        1.0,
        waypoint_count,
        dtype=link_pose.dtype,
        device=link_pose.device,
    )
    angles = hinge_rotation.to(link_pose)[:, None] * fractions[None]
    partial = axis_angle_to_rotation_matrix(
        angles[:, :, None] * axis * _HORIZONTAL_HINGE_ROLL_FRACTION
    )
    link_to_grasp = torch.matmul(
        link_pose[:, :3, :3].transpose(1, 2),
        grasp_xpos[:, :3, :3],
    )
    rotations = torch.matmul(
        torch.matmul(link_pose[:, None, :3, :3], partial),
        link_to_grasp[:, None],
    )
    result[relaxed, :, :3, :3] = rotations[relaxed]
    return result, relaxed


class _ActionEngineOpenDoor(OpenDoor):
    """Adapt drop-down handle roll without changing the core door arc."""

    binding_contract = OpenDoor.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[OpenDoorGoal, OpenDoorOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        self._gen_sim_relaxed_horizontal_hinge_roll = torch.zeros(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        )
        plan = super()._plan(request, context)
        return replace(
            plan,
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **dict(plan.diagnostics.metadata),
                    "gensim_open_door_horizontal_hinge_roll_fraction": (
                        _HORIZONTAL_HINGE_ROLL_FRACTION
                    ),
                    "gensim_open_door_relaxed_horizontal_hinge_roll": (
                        self._gen_sim_relaxed_horizontal_hinge_roll.detach()
                        .cpu()
                        .tolist()
                    ),
                },
            ),
        )

    def _opened_link_and_eef_poses(
        self,
        link_pose: torch.Tensor,
        grasp_xpos: torch.Tensor,
        rotation_axis: torch.Tensor,
        axis_origin: tuple[float, float, float],
        hinge_rotation: torch.Tensor,
        waypoint_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opened_link_poses, opened_eef_poses = super()._opened_link_and_eef_poses(
            link_pose,
            grasp_xpos,
            rotation_axis,
            axis_origin,
            hinge_rotation,
            waypoint_count,
        )
        opened_eef_poses, relaxed = _relax_horizontal_hinge_grasp_roll(
            link_pose,
            grasp_xpos,
            rotation_axis,
            hinge_rotation,
            opened_eef_poses,
        )
        self._gen_sim_relaxed_horizontal_hinge_roll = relaxed
        return opened_link_poses, opened_eef_poses


@dataclass(frozen=True, slots=True, eq=False)
class ExactTargetMoveHeldObjectOptions(MoveHeldObjectOptions):
    """Action Engine transport options for a grounded object target."""


class ExactTargetMoveHeldObject(MoveHeldObject):
    """Action Engine marker for mainline exact-target transport."""

    OptionsType = ExactTargetMoveHeldObjectOptions
    binding_contract = MoveHeldObject.binding_contract


@dataclass(frozen=True, slots=True, eq=False)
class ActionEngineMoveJointsOptions(MoveJointsOptions):
    """Joint motion with an explicit optional single-arm release effect."""

    single_release: bool = False
    """Whether a successful gripper-open command releases the held object."""

    def __post_init__(self) -> None:
        if type(self.single_release) is not bool:
            raise TypeError("single_release must be a boolean.")


class ActionEngineMoveJoints(MoveJoints):
    """Preserve ordinary joint motion and commit explicit release nodes."""

    OptionsType = ActionEngineMoveJointsOptions
    binding_contract = MoveJoints.binding_contract

    def _plan(
        self,
        request: ResolvedActionRequest[
            JointPositionGoal,
            ActionEngineMoveJointsOptions,
        ],
        context: PlanningContext,
    ) -> ActionPlan:
        endpoint = request.binding.endpoint("primary", "motion")
        task_state_key = endpoint.task_state_key
        if request.skill_options.single_release:
            if not isinstance(task_state_key, str) or not task_state_key:
                raise ValueError(
                    "Single-arm release requires a non-empty task-state key."
                )
            if context.task.get_held_object(task_state_key) is None:
                return self.failed_plan(
                    request,
                    context,
                    message=(
                        "Single-arm release requires an object held by task-state "
                        f"resource {task_state_key!r}."
                    ),
                )

        plan = super()._plan(request, context)
        if not request.skill_options.single_release:
            return plan
        return replace(
            plan,
            expected_effects=StateDelta(
                held_object_updates={task_state_key: None},
            ),
        )
