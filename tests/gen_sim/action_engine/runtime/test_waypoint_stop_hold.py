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

"""Per-row hold callbacks after normal or late-aborted waypoint stops."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.actions import _WaypointProgressGate


def _fixture():
    adapter = object.__new__(AtomicActionAdapter)
    adapter.num_envs = 2
    adapter._scene_time = 0.0
    observed = torch.zeros(2, 2)
    commands = []

    def step(command):
        commands.append(command.clone())
        observed.copy_(command + 0.05)

    adapter.env = SimpleNamespace(
        robot=SimpleNamespace(get_qpos=lambda: observed), step=step, physics_dt=1.0
    )
    trajectory = torch.arange(5.0)[None, :, None].repeat(2, 1, 2)
    return adapter, trajectory, commands


def test_already_stopped_eligible_rows_continue_to_reach_the_hold_callback():
    adapter, trajectory, commands = _fixture()
    tick = -1
    calls = []

    def stop(index):
        nonlocal tick
        tick = index
        return torch.tensor([index >= 1, index >= 3])

    def hold(command, stopped):
        calls.append((tick, command.clone(), stopped.clone()))
        result = command.clone()
        if stopped[0]:
            result[0] = 42.0 if tick >= 2 else 10.0
        return result

    active = torch.tensor([True, True])
    adapter.execute_trajectory(
        trajectory, active=active, waypoint_stop=stop, waypoint_stop_hold=hold
    )

    relevant = [(tick, mask.tolist()) for tick, _, mask in calls if bool(mask.any())]
    assert relevant == [(1, [True, False]), (2, [True, False]), (3, [True, True])]
    assert len(commands) == 4
    torch.testing.assert_close(commands[2][0], torch.full((2,), 10.0))
    torch.testing.assert_close(commands[3][0], torch.full((2,), 42.0))
    torch.testing.assert_close(active, torch.tensor([True, True]))


def test_default_waypoint_stop_preserves_legacy_first_measured_safety_hold():
    adapter, trajectory, commands = _fixture()

    adapter.execute_trajectory(
        trajectory,
        active=torch.tensor([True, True]),
        waypoint_stop=lambda index: torch.tensor([index >= 1, index >= 3]),
    )

    assert len(commands) == 4
    torch.testing.assert_close(commands[2][0], commands[1][0] + 0.05)
    torch.testing.assert_close(commands[3][0], commands[2][0])


@pytest.mark.parametrize("flush_hold", [False, True])
def test_safety_stop_interrupts_progress_repeats_before_another_arc_command(flush_hold):
    adapter, trajectory, commands = _fixture()
    calls = []
    active = torch.tensor([True, False])

    def stop(index):
        calls.append((index, len(commands)))
        return torch.tensor([len(commands) >= 2, False])

    gate = _WaypointProgressGate(
        start=0,
        stop=5,
        maximum_repeats=50,
        needs_repeat=lambda index: torch.tensor([True, False]),
        trace={},
    )
    kwargs = {"flush_stop_hold": True} if flush_hold else {}
    adapter.execute_trajectory(
        trajectory,
        active=active,
        waypoint_progress_gate=gate,
        waypoint_stop=stop,
        **kwargs,
    )
    assert calls == [(0, 1), (0, 2)]
    assert len(commands) == (3 if flush_hold else 2)
    torch.testing.assert_close(commands[1], commands[0])
    if flush_hold:
        torch.testing.assert_close(commands[-1][0], commands[1][0] + 0.05)
    torch.testing.assert_close(active, torch.tensor([True, False]))
    assert gate.trace["repeated_waypoints"][0]["timed_out_env_ids"] == []


def test_stopped_row_remains_held_while_peer_repeats_and_advances():
    adapter, trajectory, commands = _fixture()
    gate = _WaypointProgressGate(
        start=0,
        stop=5,
        maximum_repeats=5,
        needs_repeat=lambda index: torch.tensor([True, len(commands) < 4]),
        trace={},
    )
    adapter.execute_trajectory(
        trajectory,
        active=torch.tensor([True, True]),
        waypoint_progress_gate=gate,
        waypoint_stop=lambda index: torch.tensor([len(commands) >= 2, index >= 2]),
        flush_stop_hold=True,
    )
    assert len(commands) == 7
    for command in commands[2:]:
        torch.testing.assert_close(command[0], commands[1][0] + 0.05)
    torch.testing.assert_close(commands[-1][1], commands[-2][1] + 0.05)


def test_abort_hold_never_dispatches_nonfinite_observation():
    adapter, trajectory, commands = _fixture()
    read = adapter.env.robot.get_qpos

    def observed():
        value = read().clone()
        if len(commands) >= 1:
            value[0, 0] = float("nan")
        return value

    adapter.env.robot.get_qpos = observed
    with pytest.raises(ValueError, match="observed safety hold"):
        adapter.execute_trajectory(
            trajectory,
            active=torch.tensor([True, False]),
            waypoint_stop=lambda index: torch.tensor([True, False]),
            flush_stop_hold=True,
        )
    assert len(commands) == 1
    assert bool(torch.isfinite(commands[0]).all())


def test_final_waypoint_stop_flushes_hold_even_when_peer_completes_normally():
    adapter, trajectory, commands = _fixture()
    adapter.execute_trajectory(
        trajectory,
        active=torch.tensor([True, True]),
        waypoint_stop=lambda index: torch.tensor([index == 4, False]),
        flush_stop_hold=True,
    )
    assert len(commands) == 6
    torch.testing.assert_close(commands[-1][0], commands[-2][0] + 0.05)
    torch.testing.assert_close(commands[-1][1], commands[-2][1])


def test_stop_hold_mask_does_not_include_initially_inactive_rows():
    adapter, trajectory, _ = _fixture()
    masks = []

    def hold(command, stopped):
        masks.append(stopped.clone())
        return command.clone()

    adapter.execute_trajectory(
        trajectory,
        active=torch.tensor([True, False]),
        waypoint_stop=lambda _index: torch.tensor([True, True]),
        waypoint_stop_hold=hold,
    )

    assert masks
    assert all(mask.tolist() == [True, False] for mask in masks)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_stop_hold_rejects_invalid_full_robot_commands(invalid):
    adapter, trajectory, commands = _fixture()
    callback_calls = []

    def hold(command, _stopped):
        callback_calls.append(command.clone())
        if invalid == "shape":
            return command[:, :1]
        result = command.clone()
        result[0, 0] = float("nan")
        return result

    with pytest.raises((ValueError, TypeError)):
        adapter.execute_trajectory(
            trajectory,
            active=torch.tensor([True, True]),
            waypoint_stop=lambda _index: torch.tensor([True, False]),
            waypoint_stop_hold=hold,
        )
    assert callback_calls
    assert all(bool(torch.isfinite(command).all()) for command in commands)
