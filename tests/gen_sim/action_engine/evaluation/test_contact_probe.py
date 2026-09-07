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

import json
import sys
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_engine.evaluation import _contact_probe as probe
from embodichain.gen_sim.action_engine.evaluation._contact_probe import (
    _ContactWindow,
    _finger_table_metrics,
    main,
)


def _sample(**changes):
    return {
        "now": 10.0,
        "known": True,
        "unsafe": True,
        "allowed": True,
        "penetration": 0.0002,
        "impulse": 0.01,
        "tcp": (0.0, 0.0, 0.0),
        "tracking_error": 0.01,
        **changes,
    }


def test_probe_allows_only_three_additional_commands() -> None:
    window = _ContactWindow()
    for now in (10.0, 10.02, 10.04):
        assert window.decide(**_sample(now=now)) == "grace"
    assert window.decide(**_sample(now=10.06)) == "command_limit"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"known": False}, "unknown_contact"),
        ({"allowed": False}, "forbidden_contact"),
        ({"penetration": 0.0011}, "penetration_limit"),
        ({"tracking_error": 0.151}, "tracking_limit"),
        ({"impulse": 0.11}, "impulse_limit"),
        ({"now": float("nan")}, "invalid_observation"),
        ({"tcp": (float("nan"), 0, 0)}, "invalid_observation"),
    ],
)
def test_probe_limits_apply_before_continuing(changes, reason) -> None:
    assert _ContactWindow().decide(**_sample(**changes)) == reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"now": 10.13}, "time_limit"),
        ({"tcp": (0.004, 0, 0)}, "travel_limit"),
        ({"impulse": 0.095}, "impulse_limit"),
        ({"now": 9.0}, "invalid_observation"),
    ],
)
def test_probe_bounds_accumulate_from_first_contact(changes, reason) -> None:
    window = _ContactWindow()
    assert window.decide(**_sample()) == "grace"
    assert window.decide(**_sample(**{"now": 10.04, **changes})) == reason


def test_probe_requires_two_clear_samples_and_never_reopens_window() -> None:
    window = _ContactWindow()
    assert window.decide(**_sample(unsafe=False)) == "clear"
    assert window.decide(**_sample()) == "grace"
    assert window.decide(**_sample(now=10.04, unsafe=False)) == "grace"
    assert window.decide(**_sample(now=10.08, unsafe=False)) == "recovered"
    assert window.decide(**_sample(now=10.12)) == "window_used"


def _pair(link="left_inner_finger", obstacle="table"):
    return {
        "obstacle_contact": True,
        "non_target_robot_world_contact": True,
        "bodies": [
            {"entity_uid": "robot", "link_name": link},
            {"entity_uid": obstacle, "link_name": None},
        ],
        "distance": -0.0002,
        "impulse": 0.01,
    }


@pytest.mark.parametrize(
    "pair",
    [_pair("left_fr3_link7"), _pair("right_inner_finger"), _pair(obstacle="cup")],
)
def test_probe_does_not_relax_other_bodies(pair) -> None:
    assert not _finger_table_metrics([pair], "robot", "left_arm")[0]


def test_probe_keeps_worst_depth_and_summed_sample_impulse() -> None:
    allowed, depth, impulse = _finger_table_metrics(
        [_pair(), {**_pair(), "distance": -0.0005}], "robot", "left_arm"
    )
    assert allowed
    assert depth == pytest.approx(0.0005)
    assert impulse == pytest.approx(0.02)


@pytest.mark.parametrize("mode", ["bounded", "observe"])
def test_probe_rejects_changing_two_hypotheses(monkeypatch, tmp_path, mode) -> None:
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contact_probe",
            "--source-bundle",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--mode",
            mode,
            "--candidate-rank",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("unsafe", [False, True])
def test_observation_mode_has_no_contact_or_short_window_limits(unsafe) -> None:
    decision = probe._observe_contact(
        **_sample(
            now=1000.0,
            unsafe=unsafe,
            allowed=False,
            penetration=0.02,
            impulse=10.0,
            tcp=(1.0, 2.0, 3.0),
            tracking_error=0.5,
        )
    )
    assert decision == ("observed_contact" if unsafe else "clear")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"known": False}, "unknown_contact"),
        ({"penetration": float("nan")}, "invalid_observation"),
        ({"tcp": (float("inf"), 0, 0)}, "invalid_observation"),
    ],
)
def test_observation_mode_still_rejects_unusable_observations(changes, reason) -> None:
    assert probe._observe_contact(**_sample(**changes)) == reason


@pytest.mark.parametrize(
    ("worker_code", "status", "expected"),
    [
        (0, "succeeded", 0),
        (0, "failed", 2),
        (0, "aborted", 2),
        (0, None, 2),
        (1, "succeeded", 1),
        (124, "succeeded", 124),
    ],
)
def test_probe_exit_code_uses_canonical_execution_status(
    worker_code, status, expected
) -> None:
    assert probe._probe_exit_code(worker_code, status) == expected


def test_diagnostic_strict_guard_does_not_inherit_observe_bundle(monkeypatch, tmp_path):
    from embodichain.gen_sim.action_engine.runtime.executor import ProgramExecutor
    from embodichain.gen_sim.task_engine import _bundle_runner

    outcome = SimpleNamespace(
        grounded=SimpleNamespace(
            motion_policy={"articulation_core_contact_policy": "observe"}
        ),
        planner_trace={},
    )
    modes = []

    def original_guard(self, step, outcomes, masks):
        mode = outcome.grounded.motion_policy["articulation_core_contact_policy"]
        modes.append(mode)
        outcome.planner_trace["articulation_core_contact_guard"] = {"policy": mode}
        return lambda index: None, None

    def execute(*args):
        executor = object.__new__(ProgramExecutor)
        executor.env = SimpleNamespace(
            num_envs=1, robot=SimpleNamespace(get_joint_ids=lambda **kwargs: [0])
        )
        executor._core_contact_stop(None, {"left_arm": outcome}, {})
        return 0

    monkeypatch.setattr(ProgramExecutor, "_core_contact_stop", original_guard)
    monkeypatch.setattr(_bundle_runner, "execute_bundle", execute)
    assert (
        probe._worker(
            SimpleNamespace(output=tmp_path, mode="strict", candidate_rank=None)
        )
        == 0
    )
    assert modes == ["stop"]
    assert (
        outcome.grounded.motion_policy["articulation_core_contact_policy"] == "observe"
    )


def test_probe_records_replayable_module_command_and_preserves_source(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {"env": {"events": {"record_camera": {"params": {}}}}}
    original = json.dumps(config)
    (source / "fast_gym_config.json").write_text(original)
    output = tmp_path / "run"
    arguments = ["--source-bundle", str(source), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [probe.__file__, *arguments])
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **k: "test-head\n")
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2)
    )

    assert main() == 2

    manifest = json.loads((output / "probe_manifest.json").read_text())
    assert manifest["command"] == [sys.executable, "-m", probe.__name__, *arguments]
    assert (source / "fast_gym_config.json").read_text() == original
    copied = json.loads((output / "bundle/fast_gym_config.json").read_text())
    assert copied["env"]["events"]["record_camera"]["params"].pop("save_path") == str(
        output / "videos"
    )
    assert copied == config
    assert manifest["task_acceptance"] is False
