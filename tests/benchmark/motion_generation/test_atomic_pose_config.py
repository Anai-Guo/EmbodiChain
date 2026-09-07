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

"""Configuration tests for batched random-pose Atomic Task benchmarks."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from scripts.benchmark.motion_generation.config import (
    TrackCfg,
    load_suite,
    resolve_atomic_batch_sizes,
    resolve_atomic_pose_randomization,
)
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    _randomization_parameters,
    _randomized_vector_batch,
    create_atomic_skill_provider,
)


def test_pose_batch_suite_declares_translation_only_batched_trials() -> None:
    """The shipped pose-batch suite maps one random pose to one env row."""
    suite = load_suite("atomic_franka_pgi_curobo_pose_batch")
    track = suite.enabled_tracks()[0]

    settings = resolve_atomic_pose_randomization(track)
    assert settings.enabled is True
    assert settings.mode == "translation"
    assert settings.pose_batch_size == 8
    assert settings.object_translation_jitter_m == (0.03, 0.03, 0.0)
    assert settings.target_translation_jitter_m == (0.03, 0.03, 0.02)
    assert resolve_atomic_batch_sizes(track) == [8]

    skills = {entry["id"]: entry for entry in track.config["skills"]}
    assert skills["pick_up"]["grasp_source"] == "fixed"
    assert skills["pick_up"]["object_position_jitter_m"] == [0.03, 0.03, 0.0]
    assert skills["move_end_effector"]["target_offset_jitter_m"] == [
        0.03,
        0.03,
        0.02,
    ]


def test_pose_batch_aliases_are_normalized() -> None:
    """Readable aliases work without changing the canonical manifest."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "random_pose_count": 4,
            "randomization": {
                "random_pose_count": 4,
                "object_position_jitter_m": [0.01, 0.02, 0.0],
                "target_offset_jitter_m": [0.03, 0.0, 0.01],
            },
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    assert settings.enabled is True
    assert settings.pose_batch_size == 4
    assert settings.object_translation_jitter_m == (0.01, 0.02, 0.0)
    assert settings.target_translation_jitter_m == (0.03, 0.0, 0.01)
    assert resolve_atomic_batch_sizes(track) == [4]


def test_atomic_pose_config_rejects_null_track_mapping() -> None:
    """Report malformed YAML track entries with a useful type error."""
    track = TrackCfg(id="atomic-task", scenario="atomic_task", config=None)
    with pytest.raises(TypeError, match="track config must be a mapping"):
        resolve_atomic_pose_randomization(track)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pose_batch_size", 0, "pose_batch_size"),
        ("pose_batch_size", -2, "pose_batch_size"),
        ("object_translation_jitter_m", [0.01, -0.01, 0.0], "object_translation"),
        ("target_translation_jitter_m", [0.01, 0.0], "target_translation"),
    ],
)
def test_invalid_pose_randomization_is_rejected(
    field: str, value: object, message: str
) -> None:
    """Reject malformed batch and translation perturbation controls early."""
    randomization = {field: value}
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={"randomization": randomization},
    )

    with pytest.raises((TypeError, ValueError), match=message):
        resolve_atomic_pose_randomization(track)


def test_explicit_batch_sizes_must_match_random_pose_batch() -> None:
    """Avoid silently running a different batch size than the pose matrix."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "batch_sizes": [1],
            "randomization": {"enabled": True, "pose_batch_size": 8},
        },
    )

    with pytest.raises(ValueError, match="exactly"):
        resolve_atomic_batch_sizes(track)


def test_none_mode_can_batch_identical_poses_without_jitter() -> None:
    """``none`` disables perturbations but may still exercise a batched planner."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "mode": "none",
                "pose_batch_size": 3,
            }
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    assert settings.enabled is True
    assert settings.mode == "none"
    assert settings.pose_batch_size == 3
    assert resolve_atomic_batch_sizes(track) == [3]


def test_none_mode_rejects_nonzero_jitter_when_enabled() -> None:
    """A disabled perturbation mode must not silently ignore configured jitter."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "enabled": True,
                "mode": "none",
                "pose_batch_size": 3,
                "target_translation_jitter_m": [0.01, 0.0, 0.0],
            }
        },
    )

    with pytest.raises(ValueError, match="mode='none'"):
        resolve_atomic_pose_randomization(track)


def test_b1_offset_manifest_keeps_legacy_waypoint_shape() -> None:
    """B=1 metadata remains ``[waypoint][xyz]`` after batched generation."""
    robot = Mock(device=torch.device("cpu"))
    robot.get_qpos.return_value = torch.zeros(1, 7)
    robot.compute_fk.return_value = torch.eye(4).unsqueeze(0)
    scenario = Mock(robot=robot, control_part="arm")
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], 7
    )

    case = create_atomic_skill_provider("move_end_effector").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        {
            "name": "legacy_shape",
            "target_offsets_m": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        },
        seed=11,
        batch_size=1,
    )

    offsets = case.case_parameters["target_offsets_m"]
    assert len(offsets) == 2
    assert len(offsets[0]) == 3
    assert not isinstance(offsets[0][0], list)


def test_pose_batch_moveeef_shares_translation_across_waypoints() -> None:
    """One pose sample translates a whole MoveEEF waypoint path together."""
    batch_size = 4
    robot = Mock(device=torch.device("cpu"))
    robot.get_qpos.side_effect = lambda name=None: torch.zeros(
        batch_size, 7 if name == "arm" else 9
    )
    robot.compute_fk.return_value = torch.eye(4).repeat(batch_size, 1, 1)
    scenario = Mock(robot=robot, control_part="arm")
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], start.shape[-1]
    )
    config = {
        "name": "shared_target",
        "target_offsets_m": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        "target_offset_jitter_m": [0.02, 0.02, 0.02],
        "_pose_batch_enabled": True,
        "_pose_batch_seed_stride": 1,
    }

    case = create_atomic_skill_provider("move_end_effector").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        config,
        seed=11,
        batch_size=batch_size,
    )

    base = torch.tensor(config["target_offsets_m"], dtype=torch.float32)
    actual = case.target_waypoints[:, :, :3, 3]
    jitter = actual - base.unsqueeze(0)
    torch.testing.assert_close(jitter[:, 0], jitter[:, 1])
    assert not torch.allclose(jitter[0], jitter[1])
    assert len(case.case_parameters["target_offsets_m"]) == batch_size


def test_randomize_object_pose_returns_live_pose_after_optional_settle() -> None:
    """PickUp can freeze the post-gravity pose used by its grasp target."""
    pose = torch.eye(4).unsqueeze(0)
    settled = pose.clone()
    settled[:, 2, 3] = 0.025
    entity = Mock()
    entity.get_local_pose.return_value = settled
    handle = Mock(initial_pose=pose, entity=entity)
    simulation = Mock()
    scenario = AtomicTaskScenario()
    scenario.simulation = simulation

    result = scenario.randomize_object_pose(
        handle,
        {},
        seed=11,
        stream=201,
        settle_steps=50,
    )

    simulation.update.assert_called_once_with(step=50)
    entity.get_local_pose.assert_called_once_with(to_matrix=True)
    torch.testing.assert_close(result, settled)
    torch.testing.assert_close(entity.set_local_pose.call_args.args[0], pose)


def test_pickup_case_pins_settle_steps_in_pose_manifest() -> None:
    """PickUp forwards its settle window so reset/replay can reproduce it."""
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 2, 3] = 0.025
    handle = Mock(
        object_id="cube",
        entity=Mock(),
        config={"settle_steps": 10},
    )
    robot = Mock(device=torch.device("cpu"))
    robot.get_qpos.side_effect = lambda name=None: (
        torch.zeros(1, 7) if name == "arm" else torch.zeros(1, 9)
    )
    robot.compute_fk.return_value = pose
    robot.compute_ik.side_effect = lambda pose, joint_seed, name: (
        torch.ones(1, dtype=torch.bool),
        joint_seed,
    )
    scenario = Mock(robot=robot, control_part="arm")
    scenario.activate_object.return_value = handle
    scenario.randomize_object_pose.return_value = pose.clone()
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], 7
    )

    case = create_atomic_skill_provider("pick_up").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        {
            "name": "settled_pick",
            "object": "cube",
            "grasp_source": "fixed",
            "pre_action_settle_steps": 50,
        },
        seed=11,
        batch_size=1,
    )

    scenario.randomize_object_pose.assert_called_once_with(
        handle,
        {
            "name": "settled_pick",
            "object": "cube",
            "grasp_source": "fixed",
            "pre_action_settle_steps": 50,
        },
        seed=11,
        stream=201,
        settle_steps=50,
    )
    assert case.case_parameters["pre_action_settle_steps"] == 50
    torch.testing.assert_close(
        torch.tensor(case.case_parameters["object_initial_pose"]), pose
    )


def test_reset_case_reapplies_frozen_object_pose_after_settle() -> None:
    """Repeated planner attempts start from the manifest pose, not drifted state."""
    pose = torch.eye(4).unsqueeze(0)
    entity = Mock()
    handle = Mock(object_id="cube", entity=entity)
    simulation = Mock()
    robot = Mock(device=torch.device("cpu"))
    scenario = AtomicTaskScenario()
    scenario._objects = {"cube": handle}

    case = Mock(
        full_start_qpos=torch.zeros(1, 7),
        object_id="cube",
        case_parameters={
            "object_initial_pose": pose.tolist(),
            "pre_action_settle_steps": 50,
        },
    )

    scenario.reset_case(simulation, robot, case, "arm")

    simulation.update.assert_called_once_with(step=50)
    assert entity.set_local_pose.call_count == 2
    torch.testing.assert_close(entity.set_local_pose.call_args.args[0], pose)
    assert entity.clear_dynamics.call_count == 2


def test_pose_fallback_does_not_leak_into_local_grasp_offsets() -> None:
    """Track pose ranges apply only to target/object pose jitter keys."""
    config = {
        "_pose_batch_translation_jitter_m": [0.25, 0.25, 0.25],
        "_pose_batch_object_translation_jitter_m": [0.25, 0.25, 0.25],
        "_pose_batch_seed_stride": 1,
    }
    target = _randomized_vector_batch(
        [0.0, 0.0, 0.0],
        config,
        jitter_name="target_offset_jitter_m",
        seed=11,
        stream=101,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    held = _randomized_vector_batch(
        [0.0, 0.0, 0.0],
        config,
        jitter_name="held_object_offset_jitter_m",
        seed=11,
        stream=101,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.allclose(held, torch.zeros_like(held))
    assert not torch.allclose(target, torch.zeros_like(target))


def test_pose_batch_internal_ranges_are_not_exposed_as_legacy_jitter() -> None:
    """Manifest ranges retain user keys while dedicated pose metadata is explicit."""
    parameters = _randomization_parameters(
        {
            "_pose_batch_enabled": True,
            "_pose_batch_seed_stride": 1,
            "_pose_batch_translation_jitter_m": [0.1, 0.1, 0.0],
            "_pose_batch_object_translation_jitter_m": [0.2, 0.2, 0.0],
            "target_offset_jitter_m": [0.1, 0.1, 0.0],
        },
        seed=11,
    )
    assert all(not key.startswith("_pose_batch_") for key in parameters["ranges"])
    assert parameters["pose_batch"] is True
