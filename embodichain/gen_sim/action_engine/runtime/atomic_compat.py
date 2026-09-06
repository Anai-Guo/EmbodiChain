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
from copy import deepcopy

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    JointPositionGoal,
    JointPositionTarget,
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
    *,
    grasp_contact_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the sampled handle point on its door arc while relaxing wrist roll."""
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
    if grasp_contact_offset is not None:
        if (
            not isinstance(grasp_contact_offset, torch.Tensor)
            or not grasp_contact_offset.is_floating_point()
            or grasp_contact_offset.shape != (link_pose.shape[0], 3)
            or not torch.isfinite(grasp_contact_offset).all()
        ):
            raise ValueError("Grasp contact offset must be finite floating (B, 3).")
        offset = grasp_contact_offset.to(result)[:, None, :, None]
        # Rotate about the sampler contact, not the backed-off TCP origin.
        correction = ((opened_eef_poses[:, :, :3, :3] - rotations) @ offset).squeeze(-1)
        result[relaxed, :, :3, 3] += correction[relaxed]
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
        self._gen_sim_grasp_contact_offset = None
        self._gen_sim_segment_plans = []
        grasp_target = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        self._gen_sim_grasp_target_id = grasp_target.target_id
        try:
            plan = super()._plan(request, context)
        finally:
            self._gen_sim_grasp_target_id = None
        return replace(
            plan,
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **dict(plan.diagnostics.metadata),
                    "gensim_open_door_segment_planning": deepcopy(
                        self._gen_sim_segment_plans
                    ),
                    "gensim_open_door_horizontal_hinge_roll_fraction": (
                        _HORIZONTAL_HINGE_ROLL_FRACTION
                    ),
                    "gensim_open_door_relaxed_horizontal_hinge_roll": (
                        self._gen_sim_relaxed_horizontal_hinge_roll.detach()
                        .cpu()
                        .tolist()
                    ),
                    "gensim_open_door_grasp_contact_offset": (
                        None
                        if self._gen_sim_grasp_contact_offset is None
                        else self._gen_sim_grasp_contact_offset.detach().cpu().tolist()
                    ),
                },
            ),
        )

    def _plan_pose_segment(
        self,
        target_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        request: ResolvedActionRequest[OpenDoorGoal, OpenDoorOptions],
        sample_count: int,
        *,
        interpolation_dt: float,
        cartesian_linear: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain segment evidence without altering the core planner result."""
        phases = ("approach", "reach", "open", "retract")
        index = len(self._gen_sim_segment_plans)
        record = {
            "call_index": index,
            "phase": phases[index] if index < len(phases) else "unmapped",
            "target_pose": target_pose.detach().clone(),
            "start_qpos": start_qpos.detach().clone(),
            "control_part": control_part,
            "requested_strategy": request.motion_policy.strategy,
            "effective_strategy": (
                "ik_interp" if cartesian_linear else request.motion_policy.strategy
            ),
            "configured_backend": self.planning_services.planner_name,
            "preserve_cartesian_samples": cartesian_linear,
            "sample_count": sample_count,
            "interpolation_dt": interpolation_dt,
        }
        success, positions = super()._plan_pose_segment(
            target_pose,
            start_qpos,
            control_part,
            request,
            sample_count,
            interpolation_dt=interpolation_dt,
            cartesian_linear=cartesian_linear,
        )
        record["success"] = success.detach().clone()
        record["planned_positions"] = positions.detach().clone()
        self._gen_sim_segment_plans.append(record)
        return success, positions

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
        target_id = getattr(self, "_gen_sim_grasp_target_id", None)
        generator = (
            None
            if target_id is None
            else self.planning_services.grasp_pose_generator(target_id)
        )
        contact_offset = getattr(generator, "interaction_contact_offset", None)
        opened_eef_poses, relaxed = _relax_horizontal_hinge_grasp_roll(
            link_pose,
            grasp_xpos,
            rotation_axis,
            hinge_rotation,
            opened_eef_poses,
            grasp_contact_offset=contact_offset,
        )
        self._gen_sim_grasp_contact_offset = (
            None if contact_offset is None else contact_offset.detach().clone()
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
