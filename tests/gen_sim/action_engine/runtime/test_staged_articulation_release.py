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

"""Observed partial release and fresh-contact detachment contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import actions
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.executor import ProgramExecutor
from embodichain.gen_sim.action_engine.runtime.models import (
    ActionOutcome,
    GroundedAction,
)
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.gen_sim.action_engine.gripper_profiles import get_gripper_profile
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    JointPositionGoal,
    PlannerDiagnostics,
    RecoveryPolicy,
    TimedTrajectory,
    TrackingPolicy,
)

from .test_actions import _FakeEngine, _commands_for

_HAND_IDS = [1, 3, 4, 5, 6, 7]


def _hand(master) -> torch.Tensor:
    return torch.as_tensor(master, dtype=torch.float32)[..., None] * torch.tensor(
        [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
    )


class _Robot:
    uid = "robot"
    dof = 8
    joint_names = [
        "arm_joint_0",
        "left_finger_joint",
        "arm_joint_1",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    ]

    def __init__(self, batch_size: int = 2) -> None:
        self.qpos = torch.zeros(batch_size, self.dof)
        self.qpos[:, [0, 2]] = torch.tensor([0.2, -0.3])
        self.qpos[:, _HAND_IDS] = _hand(0.7)

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        if target:
            return getattr(self, "target_qpos", self.qpos.clone())
        return self.qpos

    def get_joint_ids(self, *, name: str) -> list[int]:
        return {"left_eef": _HAND_IDS, "left_arm": [0, 2]}[name]

    def set_qpos(self, *_args, **_kwargs) -> None:
        pytest.fail("Release observation and preparation must not write robot state.")


def _env(batch_size: int = 2):
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    pose[:, 2, 3] = 1.0
    return SimpleNamespace(
        device="cpu",
        num_envs=batch_size,
        robot=_Robot(batch_size),
        sim=SimpleNamespace(get_rigid_object=lambda _uid: None),
        agent_gripper_model="robotiq",
        physics_dt=0.01,
        open_state=_hand(0.0),
        close_state=_hand(0.7),
        get_current_xpos_agent=lambda: (pose, pose),
        get_agent_eef_control_part=lambda _left: "left_eef",
        step=lambda *_args: pytest.fail("Preparation must not step simulation."),
    )


def _grounded(mode: str | None, batch_size: int = 2) -> GroundedAction:
    return GroundedAction(
        action_class="MoveJoints",
        arm="left_arm",
        control="hand",
        target=JointPositionGoal(target=torch.zeros(batch_size, len(_HAND_IDS))),
        cfg={},
        motion_policy={
            "sample_interval": 5,
            "articulation_release_mode": mode,
            "verify_articulation_detach": mode == "detach",
            "interaction_support_surface_z": torch.ones(batch_size),
        },
        object_uid="drawer",
    )


def _sampled_geometry(env, clearances):
    reference = torch.tensor(clearances, dtype=torch.float32)

    def sample(_env, _arm, *, qpos, target_qpos, sample_count=5):
        assert _env is env
        master = qpos[:, 1]
        fraction = (master - target_qpos[:, 1]) / master.clamp_min(1.0e-6)
        path = fraction[:, None] * torch.linspace(0.0, 1.0, sample_count)[None]
        coordinate = path.clamp(0.0, 1.0) * (reference.shape[1] - 1)
        lower = coordinate.floor().to(torch.int64)
        upper = (lower + 1).clamp_max(reference.shape[1] - 1)
        weight = coordinate - lower
        clearance = torch.lerp(
            reference.gather(1, lower), reference.gather(1, upper), weight
        )
        points = qpos.new_zeros(env.num_envs, sample_count, 1, 3)
        points[:, :, 0, 2] = clearance.to(points)
        return points

    return Mock(side_effect=sample)


def _prepare(env, grounded, clearances, monkeypatch):
    adapter = object.__new__(AtomicActionAdapter)
    adapter.env = env
    adapter.device = torch.device("cpu")
    adapter.num_envs = env.num_envs
    adapter.gripper_profile = get_gripper_profile("robotiq")
    geometry = _sampled_geometry(env, clearances)
    monkeypatch.setattr(actions, "_gripper_points", geometry)
    result = adapter._prepare_articulation_release(grounded)
    return result, geometry


def test_detach_bounds_actual_opening_to_maximum_contiguous_safe_prefix(monkeypatch):
    env = _env()
    before = env.robot.qpos.clone()
    grounded = _grounded("detach")
    result, geometry = _prepare(
        env,
        grounded,
        [[0.020, 0.010, 0.006, -0.002, 0.020], [0.020, 0.010, 0.008, 0.006, 0.004]],
        monkeypatch,
    )

    assert result.motion_policy["articulation_release_path_valid"].tolist() == [
        True,
        True,
    ]
    torch.testing.assert_close(
        result.motion_policy["articulation_release_fraction"], torch.tensor([0.5, 1.0])
    )
    torch.testing.assert_close(result.target.target, _hand([0.35, 0.0]))
    args, kwargs = geometry.call_args
    assert args == (env, "left_arm")
    torch.testing.assert_close(kwargs["qpos"], before)
    torch.testing.assert_close(kwargs["target_qpos"][:, [0, 2]], before[:, [0, 2]])
    torch.testing.assert_close(
        kwargs["target_qpos"][:, _HAND_IDS], _hand(kwargs["target_qpos"][:, 1])
    )
    torch.testing.assert_close(env.robot.qpos, before)
    torch.testing.assert_close(grounded.target.target, torch.zeros(2, len(_HAND_IDS)))


def test_asymmetric_observation_is_preserved_while_every_release_endpoint_is_legal(
    monkeypatch,
):
    env = _env(1)
    env.robot.qpos[:, _HAND_IDS] = torch.tensor([[0.7, -0.46, 0.63, -0.4, -0.3, 0.67]])
    before = env.robot.qpos.clone()
    result, geometry = _prepare(
        env,
        _grounded("detach", 1),
        [[0.020, 0.010, 0.006, -0.002, 0.020]],
        monkeypatch,
    )

    torch.testing.assert_close(env.robot.qpos, before)
    torch.testing.assert_close(result.target.target, _hand([0.35]))
    raw_to_selected = False
    for call in geometry.call_args_list:
        start, target = call.kwargs["qpos"], call.kwargs["target_qpos"]
        torch.testing.assert_close(target[:, _HAND_IDS], _hand(target[:, 1]))
        torch.testing.assert_close(start[:, [0, 2]], before[:, [0, 2]])
        torch.testing.assert_close(target[:, [0, 2]], before[:, [0, 2]])
        raw_to_selected |= torch.equal(start, before) and torch.allclose(
            target[:, _HAND_IDS], result.target.target
        )
    assert raw_to_selected


def test_fully_open_cannot_pass_with_only_a_safe_partial_prefix(monkeypatch):
    env = _env()
    result, _ = _prepare(
        env,
        _grounded("fully_open"),
        [[0.020, 0.010, 0.006, -0.002, 0.020], [0.020, 0.010, 0.008, 0.006, 0.004]],
        monkeypatch,
    )

    assert result.motion_policy["articulation_release_path_valid"].tolist() == [
        False,
        True,
    ]
    torch.testing.assert_close(result.target.target[1], _hand(0.0))
    assert result.motion_policy["articulation_release_fraction"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("later_safe", [False, True])
def test_opening_trajectory_resolution_finds_narrow_prefix_without_skipping_unsafe_interval(
    monkeypatch, later_safe
):
    env = _env(1)
    clearances = [[0.0161932, 0.00185555, -0.0111511, -0.0224346, -0.0319372]]
    if later_safe:
        clearances[0][-2:] = [0.02, 0.03]
    coarse, _ = _prepare(env, _grounded("detach", 1), clearances, monkeypatch)
    assert coarse.motion_policy["articulation_release_path_valid"].tolist() == [False]
    grounded = _grounded("detach", 1)
    grounded.motion_policy["sample_interval"] = 15

    result, geometry = _prepare(env, grounded, clearances, monkeypatch)

    assert geometry.call_args.kwargs["sample_count"] == 15
    assert result.motion_policy["articulation_release_path_valid"].tolist() == [True]
    fraction = result.motion_policy["articulation_release_fraction"]
    torch.testing.assert_close(fraction, torch.tensor([3.0 / 14.0]))
    torch.testing.assert_close(result.target.target, _hand([0.55]))
    attempted_fractions = [
        float((0.7 - call.kwargs["target_qpos"][0, 1]) / 0.7)
        for call in geometry.call_args_list
    ]
    assert max(attempted_fractions) == pytest.approx(4.0 / 14.0)
    assert any(value == pytest.approx(3.0 / 14.0) for value in attempted_fractions)


@pytest.mark.parametrize(
    "clearance", [[0.002, 0.02, 0.02, 0.02, 0.02], [0.02, 0.002, 0.02, 0.02, 0.02]]
)
def test_detach_rejects_an_unsafe_start_or_a_prefix_with_no_opening_progress(
    monkeypatch, clearance
):
    env = _env(1)
    result, _ = _prepare(env, _grounded("detach", 1), [clearance], monkeypatch)

    assert result.motion_policy["articulation_release_path_valid"].tolist() == [False]
    torch.testing.assert_close(result.target.target, env.robot.qpos[:, _HAND_IDS])


@pytest.mark.parametrize("action_class", ["Slide", "OpenDoor"])
def test_core_interaction_is_never_reselected_by_release_preparation(
    monkeypatch, action_class
):
    env = _env()
    grounded = GroundedAction(
        action_class=action_class,
        arm="left_arm",
        control="arm",
        target=object(),
        cfg={},
        motion_policy={"articulation_external_cleanup": True},
    )
    result, geometry = _prepare(env, grounded, [[0.0] * 5] * 2, monkeypatch)

    assert result is grounded
    geometry.assert_not_called()


@pytest.mark.parametrize("mode", ["detach", "fully_open"])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_adapter_plan_wires_release_geometry_into_invocation_and_execution_mask(
    monkeypatch, mode, asymmetric
):
    env = _env()
    current = env.robot.qpos
    if asymmetric:
        current[:, _HAND_IDS] = torch.tensor([[0.7, -0.46, 0.63, -0.4, -0.3, 0.67]])
    before = current.clone()
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, 2, 3] = 1.0
    env.get_current_xpos_agent = lambda: (pose, pose)
    commands = []

    def step(command):
        commands.append(command.clone())
        current.copy_(command)

    env.step = step
    adapter = AtomicActionAdapter(env)
    clearance = torch.tensor(
        [[0.020, 0.010, 0.006, -0.002, 0.020], [0.020, 0.010, 0.008, 0.006, 0.004]]
    )
    geometry = _sampled_geometry(env, clearance.tolist())
    monkeypatch.setattr(actions, "_gripper_points", geometry)
    seen = []

    def plan(invocation, context):
        seen.append(invocation)
        positions = context.robot.qpos[:, None].repeat(1, 2, 1)
        positions[:, -1, _HAND_IDS] = invocation.goal.target
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=torch.arange(2), step_dt=0.01
        )
        return ActionPlan(
            skill_id="move_joints",
            plan_success=torch.ones(2, dtype=torch.bool),
            commands=_commands_for(trajectory),
            joint_trajectory=trajectory,
            recovery_policy=RecoveryPolicy(),
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=0,
            planned_collision_world_revision=(0, 0),
            diagnostics=PlannerDiagnostics(backend="fake"),
        )

    monkeypatch.setattr(adapter, "_engine", lambda: _FakeEngine(plan))
    outcome = adapter.plan(_grounded(mode), ExecutionState(last_qpos=before))

    assert geometry.call_count >= 1
    assert len(seen) == 1
    expected_goal = (
        _hand([0.35, 0.0]) if mode == "detach" else torch.zeros(2, len(_HAND_IDS))
    )
    expected_valid = [True, True] if mode == "detach" else [False, True]
    for row, successful in enumerate(expected_valid):
        if successful:
            torch.testing.assert_close(seen[0].goal.target[row], expected_goal[row])
    assert outcome.success.tolist() == expected_valid
    trace = outcome.planner_trace["primary_action_diagnostics"][
        "articulation_release_geometry"
    ]
    assert trace["mode"] == mode
    assert trace["success"].tolist() == expected_valid
    for row in range(2):
        if not expected_valid[row]:
            continue
        assert any(
            torch.equal(call.kwargs["qpos"][row], before[row])
            and torch.allclose(
                call.kwargs["target_qpos"][row, _HAND_IDS], expected_goal[row]
            )
            for call in geometry.call_args_list
        )
    torch.testing.assert_close(current, before)

    adapter.execute_trajectory(outcome.trajectory, active=outcome.success)

    assert commands
    if mode == "fully_open":
        for command in commands:
            torch.testing.assert_close(command[0], before[0])
        torch.testing.assert_close(outcome.next_state.last_qpos[0], before[0])
    for command in commands:
        for row, successful in enumerate(expected_valid):
            if successful:
                torch.testing.assert_close(
                    command[row, _HAND_IDS], _hand(command[row, 1])
                )
    torch.testing.assert_close(commands[-1][1, _HAND_IDS], torch.zeros(len(_HAND_IDS)))


class _Observer:
    def __init__(self, frames) -> None:
        self.frames = frames
        self.seen = {}

    def sample(self, arm, target_uid, *, tick, target_link=None):
        frame = self.frames[tick]
        result = {
            key: torch.tensor(frame.get(key, [False]))
            for key in ("target_contact", "obstacle_contact", "robot_world_contact")
        }
        result["known"] = torch.tensor(frame.get("known", [True]))
        result["non_target_robot_world_contact"] = (
            torch.tensor(frame["non_target_robot_world_contact"])
            if "non_target_robot_world_contact" in frame
            else result["robot_world_contact"] & ~result["target_contact"]
        )
        if tick <= self.seen.get((arm, target_uid), -1.0):
            result["known"].zero_()
        self.seen[(arm, target_uid)] = tick
        result["pairs"] = [[] for _ in result["known"]]
        return result


def _detachment(frames, *, mask=None, initial_hand=None):
    batch_size = len(next(iter(frames.values())).get("known", [True]))
    env = _env(batch_size)
    if initial_hand is not None:
        env.robot.qpos[:, _HAND_IDS] = _hand(initial_hand)
    executor = object.__new__(ProgramExecutor)
    executor.env = env
    executor.adapter = SimpleNamespace(
        _scene_time=0.0, gripper_profile=get_gripper_profile("robotiq")
    )
    executor.grounder = SimpleNamespace(_detached_hands={}, _departure_choices={})
    executor._interaction_contacts = _Observer(frames)
    executor._detachment_states = {}
    withdrawal = Mock(return_value=torch.full((batch_size,), 0.01))

    def choose_departure(step, arm, *, active=None):
        measured = withdrawal.return_value
        selected = (
            torch.ones(batch_size, dtype=torch.bool) if active is None else active
        )
        choice = executor.grounder._departure_choices.setdefault(
            (step.id, arm),
            {
                "valid": torch.zeros(batch_size, dtype=torch.bool),
                "direction": torch.tensor([[-1.0, 0.0, 0.0]]).repeat(batch_size, 1),
                "labels": ["inverse_approach"] * batch_size,
                "clearance": torch.zeros(batch_size),
                "attempts": [],
            },
        )
        choice["valid"] = torch.where(
            selected, torch.isfinite(measured) & (measured >= 0.003), choice["valid"]
        )
        choice["clearance"] = torch.where(selected, measured, choice["clearance"])
        return measured

    withdrawal.side_effect = choose_departure
    executor._withdrawal_clearance = withdrawal
    step = SimpleNamespace(id="task_01", object_uid="drawer")
    outcome = ActionOutcome(
        trajectory=env.robot.qpos[:, None].repeat(1, 8, 1),
        success=torch.ones(batch_size, dtype=torch.bool),
        next_state=ExecutionState(last_qpos=env.robot.qpos.clone()),
        grounded=_grounded("detach", batch_size),
    )
    mask = torch.ones(batch_size, dtype=torch.bool) if mask is None else mask
    stop = executor._detachment_stop(step, {"left_arm": outcome}, {"left_arm": mask})
    return executor, step, outcome, stop


def _tick(executor, stop, tick, *, opened=True):
    executor.adapter._scene_time = float(tick)
    if opened:
        executor.env.robot.qpos[:, _HAND_IDS] = _hand(0.58)
    return stop(tick - 1)


def test_detachment_requires_three_fresh_clear_ticks_and_retains_observed_partial_hand():
    frames = {
        0.0: {"target_contact": [True], "robot_world_contact": [True]},
        1.0: {},
        2.0: {},
        3.0: {},
    }
    executor, step, outcome, stop = _detachment(frames)

    assert not _tick(executor, stop, 1).any()
    assert not _tick(executor, stop, 2).any()
    assert _tick(executor, stop, 3).tolist() == [True]
    result = executor._verify_articulation_detachment(step, "left_arm", outcome)

    assert result.tolist() == [True]
    stored = executor.grounder._detached_hands[(step.id, "left_arm")]
    torch.testing.assert_close(stored, _hand([0.58]))
    assert stored.abs().max() > 0.5
    state = outcome.planner_trace["articulation_detachment"]
    assert state["seen_target_contact"].tolist() == [True]
    assert state["clear_at_entry_without_observed_contact"].tolist() == [False]
    executor.env.robot.qpos.zero_()
    torch.testing.assert_close(stored, _hand([0.58]))


def test_detachment_keeps_asymmetric_measurement_and_records_separate_legal_hold():
    frames = {0.0: {"target_contact": [True]}, 1.0: {}, 2.0: {}, 3.0: {}}
    executor, step, outcome, stop = _detachment(frames)
    measured = torch.tensor([[0.58, -0.53, 0.62, -0.47, -0.50, 0.63]])
    for tick in (1, 2, 3):
        executor.env.robot.qpos[:, _HAND_IDS] = measured
        _tick(executor, stop, tick, opened=False)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True]
    state = outcome.planner_trace["articulation_detachment"]
    torch.testing.assert_close(state["observed_hand_qpos"], measured)
    torch.testing.assert_close(state["commanded_hand_qpos"], _hand([0.58]))
    torch.testing.assert_close(
        executor.grounder._detached_hands[(step.id, "left_arm")], _hand([0.58])
    )
    torch.testing.assert_close(executor.env.robot.qpos[:, _HAND_IDS], measured)


@pytest.mark.parametrize(
    "blocking", ["target_contact", "obstacle_contact", "robot_world_contact", "unknown"]
)
def test_contact_or_unknown_observation_never_verifies_detachment(blocking):
    bad = {"known": [False]} if blocking == "unknown" else {blocking: [True]}
    frames = {
        0.0: {"target_contact": [True], "robot_world_contact": [True]},
        1.0: {},
        2.0: {},
        3.0: bad,
    }
    executor, step, outcome, stop = _detachment(frames)
    for tick in (1, 2, 3):
        _tick(executor, stop, tick)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [False]


def test_repeated_physics_tick_cannot_count_toward_detachment():
    frames = {0.0: {"target_contact": [True]}, 1.0: {}}
    executor, step, outcome, stop = _detachment(frames)
    for _ in range(3):
        _tick(executor, stop, 1)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [False]


def test_clear_at_entry_can_cleanup_after_real_opening_without_claiming_detachment():
    executor, step, outcome, stop = _detachment({0.0: {}, 1.0: {}, 2.0: {}, 3.0: {}})
    for tick in (1, 2, 3):
        _tick(executor, stop, tick)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True]
    state = outcome.planner_trace["articulation_detachment"]
    assert state["detached"].tolist() == [False]
    assert state["clear_at_entry_without_observed_contact"].tolist() == [True]


def test_clear_at_entry_without_any_actual_opening_is_not_cleanup_eligibility():
    executor, step, outcome, stop = _detachment({0.0: {}, 1.0: {}, 2.0: {}, 3.0: {}})
    for tick in (1, 2, 3):
        _tick(executor, stop, tick, opened=False)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [False]


def test_already_fully_open_hand_can_cleanup_without_new_opening_motion():
    executor, step, outcome, stop = _detachment(
        {0.0: {}, 1.0: {}, 2.0: {}, 3.0: {}}, initial_hand=0.0
    )
    for tick in (1, 2, 3):
        _tick(executor, stop, tick, opened=False)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True]
    state = outcome.planner_trace["articulation_detachment"]
    assert state["detached"].tolist() == [False]
    assert state["clear_at_entry_without_observed_contact"].tolist() == [True]


def test_reappearing_target_contact_resets_consecutive_clear_count():
    frames = {
        0.0: {"target_contact": [True]},
        1.0: {},
        2.0: {"target_contact": [True]},
        3.0: {},
        4.0: {},
        5.0: {},
    }
    executor, step, outcome, stop = _detachment(frames)
    for tick in (1, 2, 3, 4):
        assert not _tick(executor, stop, tick).any()
    assert _tick(executor, stop, 5).tolist() == [True]

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True]


def test_contact_free_ticks_cannot_stop_until_the_withdrawal_corridor_is_clear():
    frames = {
        0.0: {"target_contact": [True]},
        1.0: {},
        2.0: {},
        3.0: {},
        4.0: {},
    }
    executor, step, outcome, stop = _detachment(frames)
    executor._withdrawal_clearance.return_value = torch.tensor([0.002])
    for tick in (1, 2, 3):
        assert not _tick(executor, stop, tick).any()
    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [False]

    executor._withdrawal_clearance.return_value = torch.tensor([0.004])
    assert _tick(executor, stop, 4).tolist() == [True]
    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True]


def test_disappearing_contact_without_actual_hand_opening_is_not_detachment():
    frames = {0.0: {"target_contact": [True]}, 1.0: {}, 2.0: {}, 3.0: {}}
    executor, step, outcome, stop = _detachment(frames)
    for tick in (1, 2, 3):
        _tick(executor, stop, tick, opened=False)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [False]


def test_inactive_rows_do_not_gain_detachment_state():
    frames = {
        0.0: {"known": [True, True], "target_contact": [True, True]},
        **{
            float(tick): {
                "known": [True, True],
                "target_contact": [False, False],
                "obstacle_contact": [False, False],
                "robot_world_contact": [False, False],
            }
            for tick in (1, 2, 3)
        },
    }
    executor, step, outcome, stop = _detachment(
        frames, mask=torch.tensor([True, False])
    )
    for tick in (1, 2, 3):
        _tick(executor, stop, tick)

    assert executor._verify_articulation_detachment(
        step, "left_arm", outcome
    ).tolist() == [True, False]


@pytest.mark.parametrize("event", ["obstacle_contact", "unknown", "target_contact"])
def test_stopped_row_cannot_recover_from_contact_or_abort_while_another_row_continues(
    event,
):
    frames = {
        0.0: {
            "known": [True, True],
            "target_contact": [True, True],
            "robot_world_contact": [True, True],
        },
        **{
            float(tick): {
                "known": [True, True],
                "target_contact": [False, True],
                "robot_world_contact": [False, True],
                "obstacle_contact": [False, False],
            }
            for tick in (1, 2, 3, 4)
        },
        **{
            float(tick): {
                "known": [True, True],
                "target_contact": [False, False],
                "robot_world_contact": [False, False],
                "obstacle_contact": [False, False],
            }
            for tick in (5, 6, 7)
        },
    }
    if event == "unknown":
        frames[4.0]["known"][0] = False
    elif event == "obstacle_contact":
        frames[4.0]["obstacle_contact"][0] = True
        frames[4.0]["robot_world_contact"][0] = True
    else:
        frames[4.0]["target_contact"][0] = True
        frames[4.0]["robot_world_contact"][0] = True
    executor, step, outcome, stop = _detachment(frames)
    previous_hold = _hand([0.66, 0.66])
    executor.grounder._detached_hands[(step.id, "left_arm")] = previous_hold.clone()
    for tick in (1, 2):
        assert not _tick(executor, stop, tick).any()
    assert _tick(executor, stop, 3).tolist() == [True, False]
    for tick in (4, 5, 6):
        assert _tick(executor, stop, tick).tolist() == [True, False]
    assert _tick(executor, stop, 7).tolist() == [True, True]

    verified = executor._verify_articulation_detachment(step, "left_arm", outcome)

    assert verified.tolist() == [False, True]
    state = outcome.planner_trace["articulation_detachment"]
    assert state["aborted"].tolist() == [True, False]
    assert state["detached"].tolist() == [False, True]
    stored = executor.grounder._detached_hands[(step.id, "left_arm")]
    torch.testing.assert_close(stored[0], previous_hold[0])
    torch.testing.assert_close(stored[1], _hand(0.58))


def test_release_plan_holds_existing_arm_target_without_changing_observation_or_plan(
    monkeypatch,
):
    env = _env()
    observed = env.robot.qpos.clone()
    env.robot.target_qpos = observed.clone()
    arm_target = torch.tensor([[0.6, -0.45], [0.7, -0.55]])
    env.robot.target_qpos[:, [0, 2]] = arm_target
    adapter = AtomicActionAdapter(env)
    grounded = _grounded("detach")
    grounded.motion_policy["interaction_support_surface_z"] = None
    seen = []

    def plan(invocation, context):
        torch.testing.assert_close(context.robot.qpos, observed)
        positions = context.robot.qpos[:, None].repeat(1, 2, 1)
        positions[:, -1, _HAND_IDS] = invocation.goal.target
        trajectory = TimedTrajectory.from_uniform_step(
            positions, env_ids=torch.arange(2), step_dt=0.01
        )
        result = ActionPlan(
            skill_id="move_joints",
            plan_success=torch.ones(2, dtype=torch.bool),
            commands=_commands_for(trajectory),
            joint_trajectory=trajectory,
            recovery_policy=RecoveryPolicy(),
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=0,
            planned_collision_world_revision=(0, 0),
            diagnostics=PlannerDiagnostics(backend="fake"),
        )
        seen.append((result, positions.clone()))
        return result

    monkeypatch.setattr(adapter, "_engine", lambda: _FakeEngine(plan))

    outcome = adapter.plan(grounded, ExecutionState(last_qpos=observed.clone()))

    policy = outcome.grounded.motion_policy
    torch.testing.assert_close(policy["articulation_arm_hold_qpos"], arm_target)
    torch.testing.assert_close(
        policy["articulation_arm_hold_observed_qpos"], observed[:, [0, 2]]
    )
    torch.testing.assert_close(
        outcome.trajectory[:, :, [0, 2]], arm_target[:, None].repeat(1, 2, 1)
    )
    torch.testing.assert_close(
        outcome.trajectory[:, :, _HAND_IDS], seen[0][1][:, :, _HAND_IDS]
    )
    torch.testing.assert_close(seen[0][0].joint_trajectory.positions, seen[0][1])
    torch.testing.assert_close(env.robot.qpos, observed)


def test_release_support_checks_hand_shape_cartesian_product_with_arm_hold_path(
    monkeypatch,
):
    env = _env()
    env.robot.target_qpos = env.robot.qpos.clone()
    env.robot.target_qpos[:, [0, 2]] += 0.01
    calls = []

    def fk(_adapter, positions, control_part):
        assert control_part == "left_arm"
        assert positions.shape[1] == 5
        torch.testing.assert_close(positions[:, 0, [0, 2]], env.robot.qpos[:, [0, 2]])
        torch.testing.assert_close(
            positions[:, -1, [0, 2]], env.robot.target_qpos[:, [0, 2]]
        )
        calls.append(positions.clone())
        poses = torch.eye(4).repeat(2, 5, 1, 1)
        poses[:, :, 2, 3] = 1.0
        poses[:, 2, 2, 3] = 0.98
        return poses, torch.empty(0)

    monkeypatch.setattr(AtomicActionAdapter, "_arm_trajectory_fk", fk)
    result, _ = _prepare(
        env, _grounded("fully_open"), [[0.02, 0.1, 0.1, 0.1, 0.02]] * 2, monkeypatch
    )

    assert result.motion_policy["articulation_release_path_valid"].tolist() == [
        False,
        False,
    ]
    assert calls


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_release_rejects_invalid_existing_robot_targets(monkeypatch, invalid):
    env = _env()
    env.robot.target_qpos = env.robot.qpos.clone()
    if invalid == "shape":
        env.robot.target_qpos = env.robot.target_qpos[:, :-1]
    else:
        env.robot.target_qpos[0, 0] = float("nan")

    with pytest.raises(ValueError, match="target"):
        _prepare(env, _grounded("detach"), [[0.02] * 5] * 2, monkeypatch)


def _detachment_hold_fixture():
    executor, step, outcome, _ = _detachment(
        {0.0: {"known": [True, True], "target_contact": [True, True]}},
        mask=torch.tensor([True, False]),
    )
    robot = executor.env.robot
    robot.joint_names = [*robot.joint_names, "auxiliary_joint_0", "auxiliary_joint_1"]
    robot.qpos = torch.cat((robot.qpos, torch.full((2, 2), 0.3)), dim=1)
    robot.dof = robot.qpos.shape[1]
    command = robot.qpos.clone()
    command[:, [0, 2]] = torch.tensor([0.45, -0.1])
    command[:, -2:] = torch.tensor([3.0, 4.0])
    robot.qpos[:, _HAND_IDS] = torch.tensor([[0.61, -0.55, 0.65, -0.53, -0.57, 0.64]])
    state = executor._detachment_states[(step.id, "left_arm")]
    state["withdrawal_clearance_valid"] = torch.tensor([True, False])
    hold = executor._detachment_hold(
        step, {"left_arm": outcome}, {"left_arm": torch.tensor([True, False])}
    )
    return executor, state, command, hold


def test_normal_detachment_hold_preserves_commanded_arm_and_freezes_first_observed_master():
    executor, _, command, hold = _detachment_hold_fixture()
    before = command.clone()
    observed = executor.env.robot.qpos.clone()

    first = hold(command, torch.tensor([True, True]))

    torch.testing.assert_close(command, before)
    torch.testing.assert_close(first[0, [0, 2]], before[0, [0, 2]])
    torch.testing.assert_close(first[0, _HAND_IDS], _hand(0.61))
    torch.testing.assert_close(first[0, -2:], before[0, -2:])
    torch.testing.assert_close(first[1], before[1])
    torch.testing.assert_close(executor.env.robot.qpos, observed)
    executor.env.robot.qpos[0, [0, 2]] += 0.05
    executor.env.robot.qpos[0, _HAND_IDS] = _hand(0.42)
    later = command.clone()
    later[0, [0, 2]] += 0.2
    later[0, -2:] = torch.tensor([7.0, 8.0])

    repeated = hold(later, torch.tensor([True, False]))

    torch.testing.assert_close(repeated[0, [0, 2]], first[0, [0, 2]])
    torch.testing.assert_close(repeated[0, _HAND_IDS], first[0, _HAND_IDS])
    torch.testing.assert_close(repeated[0, -2:], later[0, -2:])
    torch.testing.assert_close(repeated[1], later[1])


def test_late_abort_switches_normal_hold_to_first_measured_full_state_and_never_recovers():
    executor, state, command, hold = _detachment_hold_fixture()
    first = hold(command, torch.tensor([True, False]))
    state["aborted"][0] = True
    executor.env.robot.qpos[0, [0, 2]] = torch.tensor([0.2, -0.35])
    executor.env.robot.qpos[0, _HAND_IDS] = torch.tensor(
        [0.54, -0.45, 0.57, -0.47, -0.50, 0.6]
    )
    executor.env.robot.qpos[0, -2:] = torch.tensor([9.0, 10.0])
    at_abort = executor.env.robot.qpos[0].clone()

    aborted = hold(first, torch.tensor([True, False]))

    torch.testing.assert_close(aborted[0], at_abort)
    torch.testing.assert_close(aborted[1], first[1])
    executor.env.robot.qpos[0] += 0.1
    state["aborted"][0] = False
    later = command + 0.5

    still_aborted = hold(later, torch.tensor([True, False]))

    torch.testing.assert_close(still_aborted[0], at_abort)
    torch.testing.assert_close(still_aborted[1], later[1])
