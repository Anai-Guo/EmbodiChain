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

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.grasp_diagnostics import (
    _TracingAntipodalGraspPoseGenerator,
)
from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg


class _CollisionChecker:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return torch.tensor([False, True, False]), torch.tensor([0.01, -0.002, 0.0])
        return torch.tensor([True, False]), torch.tensor([-0.004, 0.003])


class _Backend:
    device = torch.device("cpu")
    _max_deviation_angle = torch.pi / 6
    _approach_direction_samples = 4

    def __init__(self) -> None:
        self.antipodal_pairs = torch.tensor(
            [
                [[-0.08, -0.08, 0.0], [0.08, -0.08, 0.0]],
                [[-0.08, -0.06, 0.0], [0.08, -0.06, 0.0]],
                [[-0.08, 0.06, 0.0], [0.08, 0.06, 0.0]],
                [[-0.08, 0.08, 0.0], [0.08, 0.08, 0.0]],
            ],
            dtype=torch.float32,
        )
        self._collision_checker = _CollisionChecker()

    def get_dual_arm_valid_grasp_poses(self, **_kwargs):
        pose = torch.eye(4).repeat(3, 1, 1)
        left_colliding, _ = self._collision_checker.query(None, pose, torch.ones(3))
        right_colliding, _ = self._collision_checker.query(
            None, pose[:2], torch.ones(2)
        )
        return {
            "left": {
                "is_success": True,
                "grasp_poses": pose[~left_colliding],
                "open_lengths": torch.ones(2),
                "total_cost": torch.tensor([0.1, 0.2]),
            },
            "right": {
                "is_success": True,
                "grasp_poses": pose[:2][~right_colliding],
                "open_lengths": torch.ones(1),
                "total_cost": torch.tensor([0.3]),
            },
        }


def test_dual_grasp_trace_separates_generation_angle_nms_and_collision(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="trace_test")
    )
    backend = _Backend()
    monkeypatch.setattr(generator, "_backend", lambda *_args: backend)
    vertices = torch.tensor(
        [
            [-0.1, -0.1, -0.02],
            [0.1, -0.1, -0.02],
            [0.1, 0.1, 0.02],
            [-0.1, 0.1, 0.02],
        ]
    )

    result = generator.get_dual_arm_valid_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        obj_poses=torch.eye(4).unsqueeze(0),
        left_to_right_arm_direction=torch.tensor([0.0, 1.0, 0.0]),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        middle_empty_ratio=0.4,
    )

    assert result[0] is not None
    trace = generator.last_dual_trace
    assert trace is not None
    assert trace["S1_grasp_pair_generation"]["antipodal_pair_count"] == 4
    assert trace["S2_approach_angle_filtering"] == {
        "left_partition_pair_count": 2,
        "right_partition_pair_count": 2,
        "left_angle_valid_pair_count": 2,
        "right_angle_valid_pair_count": 2,
    }
    assert trace["S3_nms"] == {
        "left_candidate_count": 3,
        "right_candidate_count": 2,
    }
    assert trace["S4_collision_filtering"]["left_candidate_count"] == 2
    assert trace["S4_collision_filtering"]["right_candidate_count"] == 1
    assert trace["S5_left_right_pairing"] == {
        "left_final_count": 2,
        "right_final_count": 1,
        "paired": True,
    }
    assert generator.last_dual_trace is not trace


def test_generator_context_selects_non_crossing_pair_and_canonical_half_turn() -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="pair_test")
    )
    left_poses = torch.stack((_pose_with_y(-0.2), _pose_with_y(0.2)))
    left_poses[0, 0, 0] = -1.0
    left_poses[0, 1, 1] = -1.0
    right_poses = torch.stack((_pose_with_y(0.2), _pose_with_y(-0.2)))
    result = {
        "left": {
            "is_success": True,
            "grasp_poses": left_poses,
            "open_lengths": torch.ones(2),
            "total_cost": torch.tensor([0.1, 0.0]),
        },
        "right": {
            "is_success": True,
            "grasp_poses": right_poses,
            "open_lengths": torch.ones(2),
            "total_cost": torch.tensor([0.1, 0.0]),
        },
    }

    with generator.dual_arm_selection_context(
        left_eef=torch.eye(4).unsqueeze(0),
        right_eef=torch.eye(4).unsqueeze(0),
        left_base=_pose_with_y(-0.3).unsqueeze(0),
        right_base=_pose_with_y(0.3).unsqueeze(0),
        left_to_right_direction=torch.tensor([0.0, 1.0, 0.0]),
        pair_rank=0,
        minimum_separation=0.08,
        minimum_lateral_gap=0.05,
    ):
        selected, trace = generator._select_pair(result, row_index=0)

    assert selected is not None
    assert trace is not None
    assert trace["selected"] is True
    assert trace["selected_left_index"] == 0
    assert trace["selected_right_index"] == 0
    assert trace["selected_left_half_turn"] is True
    torch.testing.assert_close(
        selected["left"]["grasp_poses"][0, :3, :3],
        torch.eye(3),
    )


def test_upright_context_rejects_end_clamps_and_ranks_mid_body_grasps(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="upright_test")
    )
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 1, 3] = torch.tensor([0.5, 0.05, 0.5, 0.7])
    poses[0, :3, 0] = torch.tensor([0.0, 1.0, 0.0])
    poses[0, :3, 1] = torch.tensor([-1.0, 0.0, 0.0])

    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(poses, torch.tensor([0.0, 0.01, 0.2, 0.3]))],
    )
    vertices = torch.tensor(
        [
            [-0.1, 0.0, -0.1],
            [0.1, 0.0, 0.1],
            [-0.1, 1.0, 0.1],
            [0.1, 1.0, -0.1],
        ]
    )

    with generator.upright_selection_context(local_axis=torch.tensor([0.0, 1.0, 0.0])):
        result = generator.get_valid_grasp_poses(
            mesh_vertices=vertices,
            mesh_triangles=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )

    ranked_poses, ranked_costs = result[0]
    assert ranked_poses[0, 1, 3] == 0.5
    assert torch.isfinite(ranked_costs[:3]).all()
    assert torch.isinf(ranked_costs[-1])
    trace = generator.last_upright_trace
    assert trace is not None
    assert trace["local_axis"] == [0.0, 1.0, 0.0]
    assert trace["candidate_count"] == 4
    assert trace["side_compatible_count"] == 3
    assert trace["central_band_count"] == 3
    assert trace["side_and_central_count"] == 2
    assert trace["retained_count"] == 3
    assert trace["best_candidate_axis_alignment"] == 0.0
    assert trace["best_candidate_axis_fraction"] == 0.5


def test_interaction_context_records_selected_candidate(monkeypatch) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_trace_test")
    )
    poses = torch.stack((_pose_with_y(0.01), _pose_with_y(0.04)))
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(poses, torch.tensor([0.2, 0.1]))],
    )

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0),
        candidate_rank=1,
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], poses[0])
    trace = generator.last_interaction_trace
    assert trace is not None
    row = trace["environment_rows"][0]
    assert row["requested_candidate_rank"] == 1
    assert row["selected_candidate_index"] == 0
    assert row["candidate_count"] == 2
    assert generator.last_interaction_trace is not trace


@pytest.mark.parametrize("paired", [False, True])
def test_interaction_can_try_existing_depth_variant_before_next_base(
    monkeypatch, paired
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="depth_pair_test"),
        interaction_depth_offset=0.025,
    )
    poses = torch.stack((_pose_with_y(0.01), _pose_with_y(0.05)))
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(poses.clone(), torch.tensor([0.0, 0.1]))],
    )
    options = {"pair_depth_variants": True} if paired else {}
    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4)[None], candidate_rank=1, **options
    ):
        assert generator.interaction_contact_offset is None
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=torch.eye(4)[None],
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )
        offset = generator.interaction_contact_offset
        torch.testing.assert_close(
            offset, torch.tensor([[0.0, 0.0, 0.025 if paired else 0.0]])
        )
        offset.fill_(1.0)
        assert generator.interaction_contact_offset[0, 0] == 0.0
    assert generator.interaction_contact_offset is None
    expected = poses[0 if paired else 1].clone()
    if paired:
        expected[:3, 3] -= expected[:3, 2] * 0.025
    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], expected)
    row = generator.last_interaction_trace["environment_rows"][0]
    assert row["selected_base_candidate_rank"] == (0 if paired else 1)
    assert row["selected_roll_degrees"] == 0.0
    assert row["candidate_count"] == 4


def test_interaction_context_skips_near_duplicate_ranks(monkeypatch) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_diversity_test")
    )
    poses = torch.stack((_pose_with_y(0.0), _pose_with_y(0.001), _pose_with_y(0.04)))
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(poses, torch.tensor([0.0, 0.01, 0.02]))],
    )

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0),
        candidate_rank=1,
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], poses[2])
    trace = generator.last_interaction_trace
    assert trace is not None
    row = trace["environment_rows"][0]
    assert row["raw_candidate_count"] == 3
    assert row["candidate_count"] == 2


def test_interaction_can_prefer_sampler_frame_over_closer_wrist_roll(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="native_roll_test")
    )
    sampled = torch.eye(4)[None]
    angle = torch.tensor(-torch.pi / 6)
    reference = sampled.clone()
    reference[0, 1, 1] = reference[0, 2, 2] = angle.cos()
    reference[0, 1, 2] = -angle.sin()
    reference[0, 2, 1] = angle.sin()
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(sampled.clone(), torch.zeros(1))],
    )
    with generator.interaction_selection_context(
        reference_xpos=reference,
        candidate_rank=0,
        roll_degrees=(0.0, -30.0),
        prefer_unmodified_roll=True,
    ):
        _, selected, _ = generator.get_best_grasp_poses(
            obj_poses=sampled, approach_direction=torch.tensor([0.0, 0.0, 1.0])
        )
    torch.testing.assert_close(selected, sampled)
    assert (
        generator.last_interaction_trace["environment_rows"][0]["selected_roll_degrees"]
        == 0.0
    )


def test_interaction_context_expands_axial_roll_and_depth_candidates(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_frame_test"),
        interaction_depth_offset=0.025,
    )
    sampled = _pose_with_y(0.04).unsqueeze(0)
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(sampled, torch.zeros(1))],
    )

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0),
        candidate_rank=3,
        roll_degrees=(0.0, -30.0),
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

    angle = torch.deg2rad(torch.tensor(-30.0))
    roll = torch.eye(4)
    roll[1, 1] = torch.cos(angle)
    roll[1, 2] = -torch.sin(angle)
    roll[2, 1] = torch.sin(angle)
    roll[2, 2] = torch.cos(angle)
    expected = sampled[0] @ roll
    expected[:3, 3] -= expected[:3, 2] * 0.025
    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], expected)
    trace = generator.last_interaction_trace
    assert trace is not None
    assert trace["environment_rows"][0]["selected_roll_degrees"] == -30.0
    assert trace["environment_rows"][0]["interaction_depth_offset"] == 0.025


def test_interaction_context_prefers_handle_cross_section_over_long_axis(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(
            model_id="interaction_aperture_test",
            max_opening_width=0.1,
        )
    )
    along_length = torch.eye(4)
    across_width = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    poses = torch.stack((along_length, across_width))
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(poses, torch.zeros(2))],
    )
    vertices = torch.tensor(
        [
            [x, y, z]
            for x in (-0.04, 0.04)
            for y in (-0.005, 0.005)
            for z in (-0.005, 0.005)
        ]
    )

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0),
        candidate_rank=0,
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            mesh_vertices=vertices,
            mesh_triangles=torch.tensor([[0, 1, 2]]),
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], across_width)
    trace = generator.last_interaction_trace
    assert trace is not None
    assert abs(trace["environment_rows"][0]["selected_opening_width"] - 0.01) < 1.0e-6


def test_interaction_context_rejects_gripper_geometry_below_support(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_support_test")
    )
    unsafe = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    safe = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [
            (torch.stack((unsafe, safe)), torch.tensor([0.0, 0.1]))
        ],
    )

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0),
        candidate_rank=0,
        support_surface_z=torch.tensor([0.0]),
        minimum_support_clearance=0.0,
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=torch.eye(4).unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

    assert success.tolist() == [True]
    torch.testing.assert_close(selected[0], safe)
    trace = generator.last_interaction_trace
    assert trace is not None
    row = trace["environment_rows"][0]
    assert row["support_rejection_count"] == 1
    assert row["support_surface_z"] == 0.0
    assert row["support_clearance"] > 0.0


@pytest.mark.parametrize("roll_degrees", [0.0, -30.0, 60.0])
def test_interaction_trace_preserves_object_local_approach_despite_grasp_roll(
    monkeypatch, roll_degrees: float
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_approach_test")
    )
    object_pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 1.0, 0.0, 0.2],
            [-1.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(object_pose.unsqueeze(0), torch.zeros(1))],
    )

    with generator.interaction_selection_context(
        reference_xpos=object_pose.unsqueeze(0),
        candidate_rank=0,
        roll_degrees=(roll_degrees,),
    ):
        success, selected, _ = generator.get_best_grasp_poses(
            obj_poses=object_pose.unsqueeze(0),
            approach_direction=torch.tensor([2.0, 0.0, 0.0]),
        )

    assert success.tolist() == [True]
    row = generator.last_interaction_trace["environment_rows"][0]
    assert row["approach_direction_world"] == [1.0, 0.0, 0.0]
    assert row["approach_direction_local"] == [0.0, 0.0, 1.0]
    if roll_degrees:
        assert not torch.allclose(selected[0, :3, 2], torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("shared_direction", [False, True])
@pytest.mark.parametrize("candidate_rank", [0, 1])
def test_interaction_trace_uses_matching_object_frame_for_each_environment(
    monkeypatch, shared_direction: bool, candidate_rank: int
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_batch_approach_test")
    )
    object_poses = torch.eye(4).repeat(2, 1, 1)
    object_poses[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [
            (pose.unsqueeze(0), torch.zeros(1)) for pose in object_poses
        ],
    )
    direction = (
        torch.tensor([0.0, 3.0, 0.0])
        if shared_direction
        else torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    )

    with generator.interaction_selection_context(
        reference_xpos=object_poses,
        candidate_rank=candidate_rank,
    ):
        generator.get_best_grasp_poses(
            obj_poses=object_poses, approach_direction=direction
        )

    rows = generator.last_interaction_trace["environment_rows"]
    assert [row["selected"] for row in rows] == [candidate_rank == 0] * 2
    assert rows[0]["approach_direction_local"] == (
        [0.0, 1.0, 0.0] if shared_direction else [0.0, 0.0, 1.0]
    )
    assert rows[1]["approach_direction_world"] == [0.0, 1.0, 0.0]
    assert rows[1]["approach_direction_local"] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "direction",
    [
        torch.zeros(3),
        torch.tensor([float("nan"), 0.0, 1.0]),
        torch.tensor([0.0, float("inf"), 1.0]),
        torch.ones(2),
        torch.ones(1, 3),
        torch.ones(2, 1, 3),
    ],
)
def test_interaction_trace_rejects_invalid_approach_before_grasp_generation(
    monkeypatch, direction: torch.Tensor
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_invalid_approach_test")
    )

    def unexpected_grasp_generation(*_args, **_kwargs):
        pytest.fail("Invalid approach must fail before grasp generation.")

    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        unexpected_grasp_generation,
    )
    object_poses = torch.eye(4).repeat(2, 1, 1)
    with generator.interaction_selection_context(
        reference_xpos=object_poses, candidate_rank=0
    ):
        with pytest.raises(ValueError, match="approach_direction"):
            generator.get_best_grasp_poses(
                obj_poses=object_poses, approach_direction=direction
            )
    assert generator.last_interaction_trace is None


def test_interaction_trace_rejects_mismatched_candidate_batch(monkeypatch) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_mismatched_batch_test")
    )
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        lambda _self, **_kwargs: [(torch.eye(4).unsqueeze(0), torch.zeros(1))],
    )
    object_poses = torch.eye(4).repeat(2, 1, 1)
    with generator.interaction_selection_context(
        reference_xpos=object_poses, candidate_rank=0
    ):
        with pytest.raises(ValueError, match="batch"):
            generator.get_best_grasp_poses(
                obj_poses=object_poses,
                approach_direction=torch.tensor([0.0, 0.0, 1.0]),
            )


@pytest.mark.parametrize("invalid_pose", ["unbatched", "nonfinite", "zero_rotation"])
def test_interaction_trace_rejects_invalid_object_frame_before_generation(
    monkeypatch, invalid_pose: str
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="interaction_invalid_frame_test")
    )

    def unexpected_grasp_generation(*_args, **_kwargs):
        pytest.fail("Invalid object frame must fail before grasp generation.")

    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_valid_grasp_poses",
        unexpected_grasp_generation,
    )
    object_poses = torch.eye(4).unsqueeze(0)
    if invalid_pose == "unbatched":
        object_poses = object_poses[0]
    elif invalid_pose == "nonfinite":
        object_poses[0, 0, 0] = float("nan")
    else:
        object_poses[0, :3, :3] = 0.0

    with generator.interaction_selection_context(
        reference_xpos=torch.eye(4).unsqueeze(0), candidate_rank=0
    ):
        with pytest.raises(ValueError, match="obj_poses|approach_direction"):
            generator.get_best_grasp_poses(
                obj_poses=object_poses,
                approach_direction=torch.tensor([0.0, 0.0, 1.0]),
            )


def test_ordinary_grasp_selection_does_not_require_interaction_direction(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="ordinary_grasp_test")
    )
    expected = (torch.ones(1, dtype=torch.bool), torch.eye(4), torch.zeros(1))
    monkeypatch.setattr(
        AntipodalGraspPoseGenerator,
        "get_best_grasp_poses",
        lambda _self, **_kwargs: expected,
    )
    assert generator.get_best_grasp_poses() is expected
    assert generator.last_interaction_trace is None


def _pose_with_y(y: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[1, 3] = y
    return pose
