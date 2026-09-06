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
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.gen_sim.action_engine.cli import run_agent as run_agent_module
from embodichain.gen_sim.action_engine.cli.run_agent import (
    _ABWorkerConfig,
    _SerializedABBranch,
    _capture_ab_initial_frame,
    _prepare_ab_branches,
    _publish_task_engine_report,
    _task_engine_exit_code,
    _validate_run_contract,
)
from embodichain.gen_sim.action_engine.runtime import (
    ExecutionReport,
    build_execution_provenance,
)


class record_camera_data:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeEnv:
    def __init__(self, recorder=None) -> None:
        self.unwrapped = self
        self.event_manager = SimpleNamespace(
            _mode_functor_cfgs={
                "interval": (
                    [
                        SimpleNamespace(
                            func=recorder,
                            params={"name": "record_cam_audience_view"},
                        )
                    ]
                    if recorder is not None
                    else []
                )
            }
        )


def _solver_run_contract(ik_solver: str) -> tuple[dict, dict]:
    class_type = "URSolver" if ik_solver == "ur" else "PytorchSolver"
    gym_config = {
        "env": {
            "extensions": {
                "action_engine": {
                    "task_name": "solver_task",
                    "seed_task_graph_hash": "a" * 64,
                    "planning_mode": "offline",
                    "gripper_model": "pgi",
                    "ik_solver": ik_solver,
                },
                "agent_ik_solver": ik_solver,
            }
        },
        "robot": {
            "solver_cfg": {
                "left_arm": {"class_type": class_type},
                "right_arm": {"class_type": class_type},
            }
        },
    }
    agent_config = {
        "task_name": "solver_task",
        "seed_task_graph_hash": "a" * 64,
        "planning_mode": "offline",
        "gripper_model": "pgi",
        "ik_solver": ik_solver,
    }
    return gym_config, agent_config


@pytest.mark.parametrize("ik_solver", ["ur", "pytorch"])
def test_run_contract_accepts_matching_concrete_ik_solver(ik_solver: str) -> None:
    gym_config, agent_config = _solver_run_contract(ik_solver)

    _validate_run_contract(gym_config, agent_config, "solver_task")


def test_run_contract_rejects_ik_solver_artifact_drift() -> None:
    gym_config, agent_config = _solver_run_contract("pytorch")
    agent_config["ik_solver"] = "ur"

    with pytest.raises(ValueError, match="different IK solvers"):
        _validate_run_contract(gym_config, agent_config, "solver_task")

    agent_config["ik_solver"] = "pytorch"
    gym_config["robot"]["solver_cfg"]["right_arm"]["class_type"] = "URSolver"
    with pytest.raises(ValueError, match="right_arm.*PytorchSolver"):
        _validate_run_contract(gym_config, agent_config, "solver_task")


def test_capture_ab_initial_frame_invokes_only_audience_recorder() -> None:
    recorder = record_camera_data()
    env = _FakeEnv(recorder)

    _capture_ab_initial_frame(env)

    assert len(recorder.calls) == 1
    args, kwargs = recorder.calls[0]
    assert args == (env, None)
    assert kwargs == {"name": "record_cam_audience_view"}


def test_capture_ab_initial_frame_requires_audience_recorder() -> None:
    with pytest.raises(RuntimeError, match="audience recorder"):
        _capture_ab_initial_frame(_FakeEnv())


def test_cli_archives_task_video_after_final_reset_and_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Env:
        def __init__(self) -> None:
            self.unwrapped = self
            self.final_reset = False

        def reset(self, *, seed=None, options=None) -> None:
            if options == {"final": True}:
                self.final_reset = True
                events.append("final_reset")

        def get_wrapper_attr(self, _name):
            return lambda **_kwargs: SimpleNamespace(
                already_executed=True,
                runtime_success=[True],
                runtime_graph_output_dir=None,
            )

        def close(self) -> None:
            events.append("close")

    env = Env()
    monkeypatch.setattr(
        run_agent_module,
        "build_env_cfg_from_args",
        lambda _args: (
            SimpleNamespace(seed=None),
            {"id": "ActionEngine-v1", "max_episodes": 1},
            None,
        ),
    )
    monkeypatch.setattr(
        run_agent_module,
        "load_config",
        lambda _path: {"planning_mode": "offline"},
    )
    monkeypatch.setattr(run_agent_module, "_validate_gym_id", lambda _cfg: None)
    monkeypatch.setattr(
        run_agent_module,
        "_validate_run_contract",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        run_agent_module,
        "load_agent_execution_program",
        lambda *_args, **_kwargs: SimpleNamespace(seed_graph=None),
    )
    monkeypatch.setattr(
        run_agent_module,
        "_load_grounded_task_plan",
        lambda _path: None,
    )
    monkeypatch.setattr(run_agent_module.gymnasium, "make", lambda **_kwargs: env)

    def archive(completed_env, task_id, *, previous_sources=None):
        assert completed_env is env
        assert env.final_reset is True
        assert previous_sources is None
        events.append(f"archive:{task_id}")

    monkeypatch.setattr(run_agent_module, "_archive_task_recording", archive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent",
            "--task_name",
            "task2_1",
            "--gym_config",
            "gym.json",
            "--agent_config",
            "agent.json",
        ],
    )

    assert run_agent_module.cli() is None
    assert events == ["final_reset", "archive:task2_1", "close"]


@pytest.mark.parametrize(
    ("episodes", "skip_archive"),
    [
        ([([False], 0)], True),
        ([([False], 0), ([False], 0)], True),
        ([([True], 0)], False),
        ([([False], 3)], False),
        ([([True], 3), ([False], 0)], False),
        ([([False], 0), ([True], 3)], False),
        ([([False], 0), ([True], 0)], False),
        ([([True], 0), ([False], 0)], False),
        ([([False, True], 0)], False),
    ],
    ids=[
        "zero-command-failure",
        "all-episodes-zero-command-failures",
        "zero-command-success-still-needs-fresh-video",
        "executed-failure-still-needs-fresh-video",
        "earlier-executed-episode-prevents-skip",
        "later-executed-episode-prevents-skip",
        "later-zero-command-success-prevents-skip",
        "earlier-zero-command-success-prevents-skip",
        "failed-report-with-successful-environment-prevents-skip",
    ],
)
def test_cli_preserves_zero_command_failure_without_reusing_stale_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    episodes: list[tuple[list[bool], int]],
    skip_archive: bool,
) -> None:
    from embodichain.gen_sim.action_engine.agent import ActionAgent

    events = []
    reports = [
        ExecutionReport(
            task_id="task",
            plan_hash="0" * 64,
            action_graph_hash="1" * 64,
            status="succeeded" if all(successes) else "failed",
            run_id="run",
            episode_id=str(index),
            provenance=build_execution_provenance(),
            action_count=count,
            environments=tuple(
                {
                    "env_id": str(env_id),
                    "success": success,
                    "semantic_success": {"open_door": success},
                    "action_count": count,
                    "retry_count": 0,
                    "recovery_count": 0,
                    "revision_count": 0,
                    "failures": [] if success else [{"type": "planning_failed"}],
                }
                for env_id, success in enumerate(successes)
            ),
            failure_events=(
                ()
                if all(successes)
                else (
                    {
                        "type": "planning_failed",
                        "edge_id": "open_door",
                        "error": "OpenDoor precontact rejected before execution",
                    },
                )
            ),
        )
        for index, (successes, count) in enumerate(episodes)
    ]
    results = iter(
        SimpleNamespace(
            already_executed=True,
            runtime_success=successes,
            runtime_graph_output_dir=None,
            report=report,
        )
        for (successes, _), report in zip(episodes, reports)
    )

    class Env:
        num_envs = len(episodes[0][0])

        def reset(self, *, seed=None, options=None) -> None:
            if options == {"final": True}:
                events.append("final_reset")

        def get_wrapper_attr(self, name):
            assert name == "create_demo_action_list"
            return lambda **_kwargs: next(results)

        def close(self) -> None:
            events.append("close")

    env = Env()
    monkeypatch.setattr(
        run_agent_module,
        "build_env_cfg_from_args",
        lambda _args: (
            SimpleNamespace(seed=None),
            {"id": "ActionEngine-v1", "max_episodes": len(episodes)},
            None,
        ),
    )
    monkeypatch.setattr(run_agent_module, "load_config", lambda _path: {})
    monkeypatch.setattr(run_agent_module, "_validate_gym_id", lambda _cfg: None)
    monkeypatch.setattr(run_agent_module, "_validate_run_contract", lambda *_args: None)
    monkeypatch.setattr(
        run_agent_module,
        "load_agent_execution_program",
        lambda *_args, **_kwargs: SimpleNamespace(seed_graph={}),
    )
    monkeypatch.setattr(run_agent_module, "_load_grounded_task_plan", lambda _path: {})
    monkeypatch.setattr(run_agent_module.gymnasium, "make", lambda **_kwargs: env)
    monkeypatch.setattr(
        ActionAgent,
        "report_execution_result",
        lambda _self, result, **_kwargs: result.report,
    )
    abortion_report = Mock(
        return_value=ExecutionReport(
            task_id="task",
            plan_hash="0" * 64,
            action_graph_hash="1" * 64,
            status="aborted",
            run_id="run",
            episode_id="0",
            provenance=build_execution_provenance(),
            error="No fresh task recording",
        )
    )
    monkeypatch.setattr(ActionAgent, "abortion_report", abortion_report)
    sources = {"old_video": "unchanged"}
    monkeypatch.setattr(
        run_agent_module, "_snapshot_task_recording", lambda _env: sources
    )

    def archive(completed_env, task_id, *, previous_sources=None):
        assert completed_env is env
        assert task_id == "task"
        assert previous_sources is sources
        assert events == ["final_reset"]
        events.append("archive")
        raise RuntimeError("No fresh task recording")

    archive_mock = Mock(side_effect=archive)
    monkeypatch.setattr(run_agent_module, "_archive_task_recording", archive_mock)
    warnings = []
    monkeypatch.setattr(run_agent_module, "log_warning", warnings.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent",
            "--task_name",
            "task",
            "--gym_config",
            str(tmp_path / "gym.json"),
            "--agent_config",
            str(tmp_path / "agent.json"),
            "--task-engine-report",
        ],
    )

    exit_code = run_agent_module.cli()
    payload = json.loads((tmp_path / "execution_report.json").read_text())

    if skip_archive:
        assert exit_code == 1
        archive_mock.assert_not_called()
        abortion_report.assert_not_called()
        assert events == ["final_reset", "close"]
        assert payload == reports[-1].as_mapping()
        assert any("no commands" in warning.lower() for warning in warnings)
        assert any("no new video" in warning.lower() for warning in warnings)
    else:
        assert exit_code == 3
        archive_mock.assert_called_once()
        abortion_report.assert_called_once()
        assert events == ["final_reset", "archive", "close"]
        assert payload["status"] == "aborted"
        assert payload["error"] == "No fresh task recording"


def _worker_config(route: str) -> _ABWorkerConfig:
    return _ABWorkerConfig(
        route=route,
        gym_config={},
        env_options={},
        gym_id="ActionEngine-v1",
        agent_config={},
        agent_config_path="agent_config.json",
        task_name="smoke",
        runtime_backend="independent",
        seed=7,
        camera_uids=("vlm_front",),
        staging_dir=f"/tmp/ab/{route}/video",
    )


class _MemoryAwareFakeWorker:
    instances = []

    def __init__(self, config: _ABWorkerConfig) -> None:
        self.config = config
        self.closed = False
        self.startup_snapshot = {
            "robot_qpos": [0.0, 1.0],
            "object_poses": {"object": [0.0, 0.0, 0.0]},
        }
        self.startup_observation = {"route": config.route}
        self.events = []
        self.instances.append(self)
        if (
            config.route == "online"
            and Path(config.staging_dir).parent.name == config.route
        ):
            raise RuntimeError("CUDA out of memory")

    def preflight(self, graph):
        self.events.append(("preflight", graph))
        return True

    def run(self, graph, **kwargs):
        self.events.append(("run", graph, kwargs))
        return SimpleNamespace(success=True)

    def finalize(self, branch_dir: Path, *, episode_index: int):
        self.events.append(("finalize", branch_dir, episode_index))
        return [(branch_dir / "video.mp4").as_posix()]

    def close(self):
        self.closed = True


def test_ab_serializes_workers_after_startup_oom() -> None:
    _MemoryAwareFakeWorker.instances = []
    branches, snapshots = _prepare_ab_branches(
        {"offline": _worker_config("offline"), "online": _worker_config("online")},
        worker_factory=_MemoryAwareFakeWorker,
        prefer_serial=False,
    )

    assert set(branches) == {"offline", "online"}
    assert all(isinstance(branch, _SerializedABBranch) for branch in branches.values())
    assert snapshots["offline"] == snapshots["online"]
    for route, branch in branches.items():
        assert branch.preflight({"route": route}) is True
        branch.run(
            {"route": route},
            run_id=f"run-{route}",
            episode_index=0,
            record_root=Path("/tmp/ab/runtime"),
        )
        assert branch.finalize(Path(f"/tmp/ab/{route}"), episode_index=0) == [
            f"/tmp/ab/{route}/video.mp4"
        ]
        branch.close()

    phases = [
        Path(worker.config.staging_dir).parent.name
        for worker in _MemoryAwareFakeWorker.instances
        if worker.config.route == "offline"
    ]
    assert phases == ["offline", "probe", "preflight", "execute"]


@pytest.mark.parametrize(
    ("status", "success"),
    [("succeeded", True), ("failed", False)],
)
def test_task_engine_report_is_mirrored_into_bundle_only_when_enabled(
    tmp_path: Path,
    status: str,
    success: bool,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    agent_config = bundle / "agent_config.json"
    report = ExecutionReport(
        task_id="task",
        plan_hash="0" * 64,
        action_graph_hash="1" * 64,
        status=status,
        run_id="run",
        episode_id="0",
        provenance=build_execution_provenance(episode_seed=7),
        environments=(
            {
                "env_id": "0",
                "success": success,
                "semantic_success": {"task_01": success},
                "action_count": 3,
                "retry_count": 0,
                "recovery_count": 0,
                "revision_count": 0,
                "failures": [],
            },
        ),
        action_count=3,
        record_dir=(tmp_path / "runtime-records").as_posix(),
    )

    assert _publish_task_engine_report(agent_config, report, enabled=False) is None
    assert not (bundle / "execution_report.json").exists()

    path = _publish_task_engine_report(agent_config, report, enabled=True)

    assert path == bundle / "execution_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == status
    assert payload["record_dir"] == report.record_dir


def test_task_engine_exit_code_uses_report_status() -> None:
    success = SimpleNamespace(status="succeeded")
    failure = SimpleNamespace(status="failed")

    assert _task_engine_exit_code(False, [success]) == 0
    assert _task_engine_exit_code(False, [success, failure]) == 1
    assert _task_engine_exit_code(True, []) == 1
