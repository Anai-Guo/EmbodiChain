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

"""Run isolated, explicitly non-acceptance articulation contact experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch

__all__: list[str] = []

# Conservative experimental bounds, not calibrated production safety limits.
_LIMITS = {
    "additional_commands": 3,
    "seconds": 0.12,
    "penetration_m": 0.001,
    "tcp_travel_m": 0.003,
    "arm_tracking_error_rad": 0.15,
    "sampled_impulse_sum": 0.10,
}
_ALLOW = {"clear", "grace", "recovered", "observed_contact"}


def _observe_contact(
    *, now, known, unsafe, allowed, penetration, impulse, tcp, tracking_error
) -> str:
    """Observe finite contacts without imposing the bounded probe's limits."""
    if not all(
        math.isfinite(x) for x in (now, penetration, impulse, tracking_error, *tcp)
    ):
        return "invalid_observation"
    if not known:
        return "unknown_contact"
    return "observed_contact" if unsafe else "clear"


class _ContactWindow:
    def __init__(self) -> None:
        self.start: float | None = None
        self.origin: tuple[float, ...] = ()
        self.commands = 0
        self.impulse = 0.0
        self.clear_samples = 0
        self.used = False

    def decide(
        self, *, now, known, unsafe, allowed, penetration, impulse, tcp, tracking_error
    ) -> str:
        if not all(
            math.isfinite(x) for x in (now, penetration, impulse, tracking_error, *tcp)
        ):
            return "invalid_observation"
        if not known:
            return "unknown_contact"
        if self.start is None and not unsafe:
            return "clear"
        if self.used:
            return "window_used" if unsafe else "clear"
        if unsafe and not allowed:
            return "forbidden_contact"
        if penetration > _LIMITS["penetration_m"]:
            return "penetration_limit"
        if tracking_error > _LIMITS["arm_tracking_error_rad"]:
            return "tracking_limit"
        if self.start is None:
            self.start, self.origin = now, tuple(tcp)
        if now < self.start:
            return "invalid_observation"
        if now - self.start >= _LIMITS["seconds"] - 1e-9:
            return "time_limit"
        if math.dist(tcp, self.origin) > _LIMITS["tcp_travel_m"]:
            return "travel_limit"
        self.impulse += impulse
        if self.impulse > _LIMITS["sampled_impulse_sum"]:
            return "impulse_limit"
        self.clear_samples = 0 if unsafe else self.clear_samples + 1
        if self.clear_samples >= 2:
            self.used = True
            return "recovered"
        if self.commands >= _LIMITS["additional_commands"]:
            return "command_limit"
        self.commands += 1
        return "grace"


def _finger_table_metrics(pairs, robot_uid, arm) -> tuple[bool, float, float]:
    relevant = [
        p for p in pairs if p["obstacle_contact"] or p["non_target_robot_world_contact"]
    ]
    allowed = bool(relevant)
    prefix = "left_" if arm == "left_arm" else "right_"
    for pair in relevant:
        robot = [b for b in pair["bodies"] if b["entity_uid"] == robot_uid]
        other = [b for b in pair["bodies"] if b["entity_uid"] != robot_uid]
        allowed &= (
            len(robot) == len(other) == 1
            and str(robot[0]["link_name"]).startswith(prefix)
            and "finger" in str(robot[0]["link_name"])
            and other[0]["entity_uid"] == "table"
        )
    return (
        allowed,
        max([0.0, *[-float(p["distance"]) for p in relevant]]),
        sum(abs(float(p["impulse"])) for p in relevant),
    )


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _probe_exit_code(worker_code: int, status: str | None) -> int:
    """Preserve process failures and use the runner's canonical success status."""
    if worker_code:
        return worker_code
    return 0 if status == "succeeded" else 2


def _worker(args) -> int:
    import torch

    from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
    from embodichain.gen_sim.action_engine.runtime.executor import ProgramExecutor
    from embodichain.gen_sim.action_engine.runtime.recording import _jsonable
    from embodichain.gen_sim.action_engine.runtime.robot_parts import arm_control_part
    from embodichain.gen_sim.task_engine._bundle_runner import execute_bundle

    original_init = ProgramExecutor.__init__
    original_guard = ProgramExecutor._core_contact_stop
    original_candidates = AtomicActionAdapter._adapt_interaction_grasp_candidates
    sink = (args.output / "contacts.jsonl").open("x")

    def emit(data):
        sink.write(
            json.dumps(
                _jsonable({"diagnostic_only": True, "task_acceptance": False, **data}),
                allow_nan=False,
            )
            + "\n"
        )
        sink.flush()

    def initialize(self, *positional, **kwargs):
        kwargs["record_root"] = str(args.output / "runtime")
        original_init(self, *positional, **kwargs)

    def candidates(grounded, capability):
        choices = original_candidates(grounded, capability)
        if args.candidate_rank is None or capability.target_materializer not in {
            "slide",
            "open_door",
        }:
            return choices
        return tuple(
            c
            for c in choices
            if c.motion_policy["interaction_grasp_candidate_rank"]
            == args.candidate_rank
        )

    def factory(self, step, outcomes, masks):
        live = [(arm, o) for arm, o in outcomes.items() if o is not None]
        if self.env.num_envs != 1 or len(live) != 1:
            raise ValueError(
                "The diagnostic supports one environment and one active arm."
            )
        arm, outcome = live[0]
        # An explicit diagnostic baseline must not inherit newer bundle defaults.
        with patch.dict(
            outcome.grounded.motion_policy, {"articulation_core_contact_policy": "stop"}
        ):
            strict_stop, aborted = original_guard(self, step, outcomes, masks)
        trace = outcome.planner_trace["articulation_core_contact_guard"]
        if args.mode != "strict":
            for key in (
                "contact_observed",
                "contact_sample_count",
                "first_contact",
                "last_contact",
            ):
                trace.pop(key, None)
            trace["policy"] = args.mode
        trace["diagnostic_policy"] = args.mode
        trace["task_acceptance"] = False
        window = _ContactWindow()
        sampler = self._sample_interaction_contacts
        arm_ids = self.env.robot.get_joint_ids(name=arm_control_part(self.env, arm))
        captured = {}

        def capture(*a, **kw):
            sample = sampler(*a, **kw)
            captured["sample"] = sample
            return sample

        def stop(index):
            if args.mode == "strict":
                with patch.object(self, "_sample_interaction_contacts", capture):
                    stopped = strict_stop(index)
                sample = captured["sample"]
            else:
                sample = sampler(step, arm)
            q = self.env.robot.get_qpos().detach().clone()
            command = self.env.robot.get_qpos(target=True).detach().clone()
            tcp_pose = self.env.get_current_xpos_agent()[0 if arm == "left_arm" else 1]
            tcp = torch.as_tensor(tcp_pose).reshape(-1, 4, 4)[0, :3, 3].tolist()
            allowed, penetration, impulse = _finger_table_metrics(
                sample["pairs"][0], self.env.robot.uid, arm
            )
            unsafe = bool(
                (
                    ~sample["known"]
                    | sample["obstacle_contact"]
                    | sample["non_target_robot_world_contact"]
                )[0]
            )
            metrics = {
                "now": float(self.env.sim._visualization_sim_time),
                "known": bool(sample["known"][0]),
                "unsafe": unsafe,
                "allowed": allowed,
                "penetration": penetration,
                "impulse": impulse,
                "tcp": tcp,
                "tracking_error": float(
                    (q[:, arm_ids] - command[:, arm_ids]).abs().max()
                ),
            }
            if args.mode != "strict":
                decision = (
                    window.decide(**metrics)
                    if args.mode == "bounded"
                    else _observe_contact(**metrics)
                )
                if decision not in _ALLOW and bool(masks[arm][0] & outcome.success[0]):
                    aborted[0] = True
                    trace["aborted"][0] = True
                    trace["failure_reason"][0] = decision
                    trace["first_failure"][0] = {
                        "waypoint_index": index,
                        "pairs": sample["pairs"][0],
                        **metrics,
                    }
                trace["sample_count"] += 1
                stopped = aborted
            else:
                decision = "strict_stop" if bool(stopped[0]) else "clear"
            observation = self._action_execution_observation(
                step.object_uid, grounded=outcome.grounded
            )
            emit(
                {
                    "event": "core_control_sample",
                    "mode": args.mode,
                    "step": step.id,
                    "arm": arm,
                    "waypoint_index": index,
                    "decision": decision,
                    "strict_would_stop": unsafe,
                    "scope": "last_physics_substep_per_control_command_not_continuous",
                    "physics_dt": self.env.physics_dt,
                    "control_dt": self.env.step_dt,
                    "metrics": metrics,
                    "contacts": sample,
                    "robot_qpos": q,
                    "commanded_qpos": command,
                    "observation": observation,
                }
            )
            return stopped

        return stop, aborted

    try:
        with (
            patch.object(ProgramExecutor, "__init__", initialize),
            patch.object(ProgramExecutor, "_core_contact_stop", factory),
            patch.object(
                AtomicActionAdapter,
                "_adapt_interaction_grasp_candidates",
                staticmethod(candidates),
            ),
        ):
            return execute_bundle(
                args.output / "bundle",
                [
                    "--headless",
                    "--filter_dataset_saving",
                    "--max_episodes",
                    "1",
                    "--num_envs",
                    "1",
                    "--seed",
                    "0",
                    "--failure-policy",
                    "stop",
                ],
            )
    finally:
        sink.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("strict", "bounded", "observe"), default="strict"
    )
    parser.add_argument("--candidate-rank", type=int, choices=range(8))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode != "strict" and args.candidate_rank is not None:
        parser.error(
            "Change only one hypothesis: contact policy cannot change grasp rank."
        )
    args.output = args.output.resolve()
    args.source_bundle = args.source_bundle.resolve()
    if args.worker:
        return _worker(args)
    if not args.source_bundle.is_dir() or args.source_bundle in args.output.parents:
        parser.error(
            "Use an existing source bundle and a separate new output directory."
        )
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    shutil.copytree(
        args.source_bundle,
        bundle,
        ignore=shutil.ignore_patterns(
            "execution_report.json", ".task-engine-runtime-*"
        ),
    )
    config_path = bundle / "fast_gym_config.json"
    config = json.loads(config_path.read_text())
    config["env"]["events"]["record_camera"]["params"]["save_path"] = str(
        args.output / "videos"
    )
    _write(config_path, config)
    command = [
        sys.executable,
        "-u",
        "-m",
        __spec__.name,
        "--worker",
        "--source-bundle",
        str(args.source_bundle),
        "--output",
        str(args.output),
        "--mode",
        args.mode,
    ]
    if args.candidate_rank is not None:
        command.extend(["--candidate-rank", str(args.candidate_rank)])
    dexsim = Path(importlib.util.find_spec("dexsim").origin).parent
    manifest = {
        "diagnostic_only": True,
        "task_acceptance": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", __spec__.name, *sys.argv[1:]],
        "worker_command": command,
        "python": sys.version,
        "dexsim": (dexsim / "version.txt").read_text(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "probe_sha256": _sha(Path(__file__)),
        "mode": args.mode,
        "candidate_rank": args.candidate_rank,
        "limits": _LIMITS if args.mode == "bounded" else None,
        "observe_contacts_without_stopping": args.mode == "observe",
        "source_bundle": str(args.source_bundle),
        "output": str(args.output),
        "source_hashes": {p.name: _sha(p) for p in args.source_bundle.glob("*.json")},
        "copied_config_change": "record_camera.params.save_path only",
    }
    _write(args.output / "probe_manifest.json", manifest)
    started = time.monotonic()
    with (args.output / "process.log").open("x") as log:
        try:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=300
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    report_path = bundle / "execution_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    videos = [
        {
            "path": str(p),
            "sha256": _sha(p),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in sorted((args.output / "videos").glob("*.mp4"))
    ]
    summary = {
        "diagnostic_only": True,
        "task_acceptance": False,
        "worker_exit_code": code,
        "driver_exit_code": _probe_exit_code(code, report.get("status")),
        "runtime_status": report.get("status"),
        "action_count": report.get("action_count"),
        "record_dir": report.get("record_dir"),
        "elapsed_seconds": time.monotonic() - started,
        "videos": videos,
    }
    _write(args.output / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary["driver_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
