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

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import pytest

from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.atomic_compat import (
    _ActionEngineOpenDoor,
    ActionEngineMoveJoints,
    ActionEngineMoveJointsOptions,
    ExactTargetMoveHeldObject,
    ExactTargetMoveHeldObjectOptions,
    _relax_horizontal_hinge_grasp_roll,
)
from embodichain.lab.sim.atomic_actions import (
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    OpenDoor,
    PlannerDiagnostics,
    StateDelta,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix


def test_open_door_adapter_relaxes_only_horizontal_hinge_wrist_roll() -> None:
    link_pose = torch.eye(4).repeat(2, 1, 1)
    grasp_pose = link_pose.clone()
    rotation_axis = torch.tensor([1.0, 0.0, 0.0])
    hinge_rotation = torch.tensor([torch.pi / 2.0, torch.pi / 2.0])
    opened = torch.eye(4).repeat(2, 2, 1, 1)
    opened[:, -1, :3, :3] = axis_angle_to_rotation_matrix(
        rotation_axis.repeat(2, 1) * (torch.pi / 2.0)
    )
    link_pose[1, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, torch.pi / 2.0, 0.0]])
    )[0]

    result, relaxed = _relax_horizontal_hinge_grasp_roll(
        link_pose,
        grasp_pose,
        rotation_axis,
        hinge_rotation,
        opened,
    )

    expected = axis_angle_to_rotation_matrix(
        torch.tensor([[torch.pi / 8.0, 0.0, 0.0]])
    )[0]
    assert relaxed.tolist() == [True, False]
    torch.testing.assert_close(result[0, -1, :3, :3], expected)
    torch.testing.assert_close(result[1], opened[1])
    assert _ActionEngineOpenDoor.binding_contract is OpenDoor.binding_contract


@pytest.mark.parametrize("hinge_angle", [1.38, -1.38])
def test_relaxed_roll_preserves_offset_handle_contact_arc(hinge_angle):
    link = torch.eye(4).repeat(2, 1, 1)
    link[1, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([[0.0, torch.pi / 2, 0.0]])
    )[0]
    grasp = link.clone()
    grasp[:, :3, 3] = torch.tensor([0.02, -0.1, 0.12])
    axis = torch.tensor([1.0, 0.0, 0.0])
    angles = torch.full((2,), hinge_angle)
    # The selected TCP is backed away from the sampler contact point by 25 mm.
    offset = torch.tensor([[0.0, 0.0, 0.025], [0.0, 0.0, 0.025]])
    _, rigid = OpenDoor._opened_link_and_eef_poses(
        SimpleNamespace(device=torch.device("cpu")),
        link,
        grasp,
        axis,
        (0.0, -0.1, -0.06),
        angles,
        50,
    )
    before = rigid.clone()
    result, relaxed = _relax_horizontal_hinge_grasp_roll(
        link,
        grasp,
        axis,
        angles,
        rigid,
        grasp_contact_offset=offset,
    )
    rigid_contact = rigid[:, :, :3, 3] + (
        rigid[:, :, :3, :3] @ offset[:, None, :, None]
    ).squeeze(-1)
    relaxed_contact = result[:, :, :3, 3] + (
        result[:, :, :3, :3] @ offset[:, None, :, None]
    ).squeeze(-1)
    torch.testing.assert_close(relaxed_contact, rigid_contact, atol=1e-7, rtol=0)
    assert relaxed.tolist() == [True, False]
    torch.testing.assert_close(result[1], rigid[1])
    torch.testing.assert_close(rigid, before)


def test_grounded_target_transport_uses_mainline_exact_target_contract() -> None:
    action = ExactTargetMoveHeldObject()
    assert type(action).__dict__["binding_contract"] is MoveHeldObject.binding_contract
    assert action._plan.__func__ is MoveHeldObject._plan
    assert "_apply_automatic_transport_rotation" not in type(action).__dict__


@pytest.mark.parametrize(
    "offset",
    [
        torch.zeros(3),
        torch.zeros(2, 3),
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[0.0, 0.0, float("nan")]]),
    ],
)
def test_relaxed_roll_rejects_invalid_contact_offsets(offset):
    pose = torch.eye(4)[None]
    with pytest.raises(ValueError, match="Grasp contact offset"):
        _relax_horizontal_hinge_grasp_roll(
            pose,
            pose,
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0]),
            pose[:, None],
            grasp_contact_offset=offset,
        )


def test_open_door_plan_reads_current_contact_without_diagnostic_trace(monkeypatch):
    @dataclass
    class Plan:
        diagnostics: PlannerDiagnostics

    generator = SimpleNamespace(interaction_contact_offset=None)
    action = _ActionEngineOpenDoor()
    resolved = []

    def get_generator(target_id):
        assert generator.interaction_contact_offset is not None
        resolved.append(target_id)
        return generator

    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"), grasp_pose_generator=get_generator
    )
    endpoint = SimpleNamespace(
        require_target=lambda kind: SimpleNamespace(target_id="selected_hand")
    )
    request = SimpleNamespace(binding=SimpleNamespace(endpoint=lambda *args: endpoint))
    context = SimpleNamespace(
        batch_size=1, robot=SimpleNamespace(qpos=torch.zeros(1, 1))
    )
    contacts = []

    def planned_core(self, request, context):
        generator.interaction_contact_offset = torch.tensor([[0.0, 0.0, 0.025]])
        link = torch.eye(4)[None]
        grasp = link.clone()
        grasp[:, 2, 3] = 0.15
        opened_link, eef = self._opened_link_and_eef_poses(
            link,
            grasp,
            torch.tensor([1.0, 0.0, 0.0]),
            (0.0, 0.0, 0.0),
            torch.tensor([1.38]),
            10,
        )
        anchor = torch.tensor([0.0, 0.0, 0.175, 1.0])
        contacts.append((opened_link @ anchor)[..., :3])
        contacts.append(
            eef[..., :3, 3]
            + (
                eef[..., :3, :3]
                @ generator.interaction_contact_offset[:, None, :, None]
            ).squeeze(-1)
        )
        return Plan(PlannerDiagnostics(backend="test"))

    monkeypatch.setattr(OpenDoor, "_plan", planned_core)
    plan = action._plan(request, context)
    torch.testing.assert_close(*contacts)
    assert resolved == ["selected_hand"]
    assert action._gen_sim_grasp_target_id is None
    assert plan.diagnostics.metadata["gensim_open_door_grasp_contact_offset"] == [
        [0.0, 0.0, pytest.approx(0.025)]
    ]
    generator.interaction_contact_offset.fill_(1.0)
    monkeypatch.setattr(
        OpenDoor, "_plan", lambda *args: Plan(PlannerDiagnostics(backend="hold"))
    )
    held = action._plan(request, context)
    assert held.diagnostics.metadata["gensim_open_door_grasp_contact_offset"] is None
    assert resolved == ["selected_hand"]


def test_open_door_records_segment_masks_and_actual_routes_without_changing_plan(
    monkeypatch,
):
    @dataclass
    class Plan:
        plan_success: torch.Tensor
        diagnostics: PlannerDiagnostics

    action = _ActionEngineOpenDoor()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"), planner_name="curobo"
    )
    endpoint = SimpleNamespace(
        require_target=lambda kind: SimpleNamespace(target_id="hand")
    )
    request = SimpleNamespace(
        binding=SimpleNamespace(endpoint=lambda *args: endpoint),
        motion_policy=SimpleNamespace(strategy="motion_gen"),
    )
    context = SimpleNamespace(
        batch_size=2, robot=SimpleNamespace(qpos=torch.zeros(2, 7))
    )
    targets = torch.eye(4).repeat(2, 1, 1)
    outputs = []

    def segment(
        self, target_pose, start_qpos, control_part, request, sample_count, **kwargs
    ):
        index = len(outputs)
        success = torch.tensor([index != 1, index != 2])
        positions = torch.full((2, sample_count, 7), float(index))
        outputs.append((success, positions))
        return success, positions

    def plan_core(self, request, context):
        success = torch.ones(2, dtype=torch.bool)
        for linear in (False, True, False, True):
            mask, positions = self._plan_pose_segment(
                targets,
                context.robot.qpos,
                "left_arm",
                request,
                4,
                interpolation_dt=0.04,
                cartesian_linear=linear,
            )
            assert positions is outputs[-1][1]
            assert mask is outputs[-1][0]
            success &= mask
        return Plan(success, PlannerDiagnostics(backend="curobo"))

    monkeypatch.setattr(OpenDoor, "_plan_pose_segment", segment)
    monkeypatch.setattr(OpenDoor, "_plan", plan_core)
    plan = action._plan(request, context)
    assert plan.plan_success.tolist() == [False, False]
    stages = plan.diagnostics.metadata["gensim_open_door_segment_planning"]
    assert [s["phase"] for s in stages] == ["approach", "reach", "open", "retract"]
    assert [s["effective_strategy"] for s in stages] == [
        "motion_gen",
        "ik_interp",
        "motion_gen",
        "ik_interp",
    ]
    assert [s["success"].tolist() for s in stages] == [
        [True, True],
        [False, True],
        [True, False],
        [True, True],
    ]
    outputs[1][0].fill_(True)
    targets.fill_(0)
    assert stages[1]["success"].tolist() == [False, True]
    torch.testing.assert_close(stages[1]["target_pose"], torch.eye(4).repeat(2, 1, 1))
    monkeypatch.setattr(
        OpenDoor,
        "_plan",
        lambda *args: Plan(
            torch.ones(2, dtype=torch.bool), PlannerDiagnostics(backend="hold")
        ),
    )
    assert (
        action._plan(request, context).diagnostics.metadata[
            "gensim_open_door_segment_planning"
        ]
        == []
    )


def test_semantic_transport_config_has_no_task_facing_rotation_switch() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    action = SimpleNamespace(cfg={})
    capability = SimpleNamespace(
        config_type=MoveHeldObjectOptions,
        target_materializer="semantic_held_object",
    )

    options = adapter._build_single_arm_config(action, capability)

    assert isinstance(options, ExactTargetMoveHeldObjectOptions)
    assert not hasattr(options, "allow_automatic_transport_rotation")


def test_joint_config_materializes_single_release_only_when_requested() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    capability = SimpleNamespace(
        config_type=MoveJointsOptions,
        target_materializer="joint_state",
    )

    release = adapter._build_single_arm_config(
        SimpleNamespace(cfg={"single_release": True}),
        capability,
    )
    ordinary = adapter._build_single_arm_config(
        SimpleNamespace(cfg={}),
        capability,
    )

    assert isinstance(release, ActionEngineMoveJointsOptions)
    assert release.single_release
    assert isinstance(ordinary, ActionEngineMoveJointsOptions)
    assert not ordinary.single_release


def test_single_release_binds_hand_motion_to_the_arm_held_state_key() -> None:
    adapter = AtomicActionAdapter.__new__(AtomicActionAdapter)
    adapter._parts = lambda _arm: ("physical_left_arm", "physical_left_hand", 2)
    captured = {}

    class Engine:
        def bind_control_parts(self, skill_id, endpoints, *, task_state_keys=None):
            captured.update(
                skill_id=skill_id,
                endpoints=endpoints,
                task_state_keys=task_state_keys,
            )
            return object()

    adapter._binding(
        SimpleNamespace(
            arm="left_arm",
            control="hand",
            cfg={"single_release": True},
        ),
        SimpleNamespace(
            action_type=MoveJoints,
            config_materializer="single_arm",
        ),
        engine=Engine(),
    )

    assert captured == {
        "skill_id": "move_joints",
        "endpoints": {"primary": {"motion": "physical_left_hand"}},
        "task_state_keys": {"primary": "physical_left_arm"},
    }


def test_single_release_plan_removes_only_the_bound_arm_attachment(monkeypatch) -> None:
    @dataclass(frozen=True)
    class Plan:
        expected_effects: object

    monkeypatch.setattr(
        MoveJoints,
        "_plan",
        lambda _self, _request, _context: Plan(expected_effects=object()),
    )
    action = ActionEngineMoveJoints()
    request = SimpleNamespace(
        binding=SimpleNamespace(
            endpoint=lambda _slot, _endpoint: SimpleNamespace(task_state_key="left_arm")
        ),
        skill_options=ActionEngineMoveJointsOptions(single_release=True),
    )
    context = SimpleNamespace(
        task=SimpleNamespace(
            get_held_object=lambda key: object() if key == "left_arm" else None
        )
    )

    plan = action._plan(request, context)

    assert isinstance(plan.expected_effects, StateDelta)
    assert dict(plan.expected_effects.held_object_updates) == {"left_arm": None}
