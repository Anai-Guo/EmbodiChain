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

"""Finite, contact-constrained departure choices without simulation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.executor import ProgramExecutor
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.runtime.models import ActionOutcome
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState

from .test_staged_articulation_release import (
    _HAND_IDS,
    _Observer,
    _env,
    _grounded,
    _hand,
)

_DIRECTIONS = {
    "inverse_approach": torch.tensor([-1.0, 0.0, 0.0]),
    "tool_back": torch.tensor([0.0, 0.0, -1.0]),
    "baseward": torch.tensor([0.0, -1.0, 0.0]),
    "world_up": torch.tensor([0.0, 0.0, 1.0]),
}


def _label(direction):
    for label, expected in _DIRECTIONS.items():
        if torch.allclose(direction, expected, atol=1e-5, rtol=0):
            return label
    pytest.fail(f"Unexpected departure direction {direction.tolist()}.")


def _fixture(monkeypatch, clearances, *, ik_success=None, terminal_clearances=None):
    env = _env(2)
    env.robot.qpos[:, _HAND_IDS] = torch.tensor([[0.7, -0.45, 0.6, -0.55, -0.48, 0.64]])
    tcp = env.get_current_xpos_agent()[0]
    initial_qpos = env.robot.qpos.clone()
    surface = torch.tensor([0.80, 0.81])
    table_pose = torch.eye(4).repeat(2, 1, 1)
    table_pose[:, 2, 3] = surface
    table = SimpleNamespace(
        get_vertices=lambda **_kwargs: torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]
        ),
        get_local_pose=lambda **_kwargs: table_pose,
    )
    articulation = SimpleNamespace(
        get_link_pose=lambda link, **_kwargs: torch.eye(4).repeat(2, 1, 1)
    )
    env.sim.get_rigid_object = lambda uid: table if uid == "table" else None
    env.sim.get_articulation = lambda uid: articulation if uid == "drawer" else None

    def base_pose(*, name, to_matrix):
        assert to_matrix and name in {"left_arm", "right_arm"}
        result = tcp.clone()
        result[:, 1, 3] += -1.0 if name == "left_arm" else 1.0
        return result

    monkeypatch.setattr(
        env.robot, "get_control_part_base_pose", base_pose, raising=False
    )
    ik_calls = []

    def ik(*, pose, name, joint_seed):
        assert name == "left_arm"
        assert pose.shape == (2, 1, 4, 4)
        assert joint_seed.shape == (2, 1, 2)
        torch.testing.assert_close(joint_seed[:, 0], env.robot.qpos[:, [0, 2]])
        torch.testing.assert_close(pose[:, 0, :3, :3], tcp[:, :3, :3])
        delta = pose[:, 0, :3, 3] - tcp[:, :3, 3]
        torch.testing.assert_close(
            delta.norm(dim=1), torch.full((2,), 0.1), atol=1e-6, rtol=0
        )
        labels = [_label(value / value.norm()) for value in delta]
        ik_calls.append({"labels": labels, "seed": joint_seed.clone()})
        torch.rand(1)
        valid = torch.tensor(
            [
                (ik_success or {}).get(label, [True, True])[row]
                for row, label in enumerate(labels)
            ]
        )
        return valid[:, None], joint_seed.clone()

    monkeypatch.setattr(env.robot, "compute_batch_ik", ik, raising=False)
    geometry_calls = []

    def withdrawal(
        arm,
        target_uid,
        *,
        direction,
        distance,
        hand_target_qpos,
        support_surface_z,
        diagnostics,
    ):
        assert arm == "left_arm" and target_uid == "drawer"
        assert distance == pytest.approx(0.1)
        torch.testing.assert_close(
            direction.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(support_surface_z, surface)
        torch.testing.assert_close(
            hand_target_qpos[:, _HAND_IDS], _hand(env.robot.qpos[:, 1])
        )
        torch.testing.assert_close(
            hand_target_qpos[:, [0, 2]], env.robot.qpos[:, [0, 2]]
        )
        labels = [_label(row) for row in direction]
        geometry_calls.append(
            {"labels": labels, "direction": direction.clone(), "distance": distance}
        )
        values = torch.tensor(
            [
                clearances.get(label, [-0.01, -0.01])[row]
                for row, label in enumerate(labels)
            ]
        )
        diagnostics["terminal_clearance"] = torch.tensor(
            [
                (terminal_clearances or {}).get(
                    label, clearances.get(label, [-0.01, -0.01])
                )[row]
                for row, label in enumerate(labels)
            ]
        )
        return values

    runtime_policy = SimpleNamespace(
        grounding={"semantic_defaults": {"safe_retreat_distance": 0.1}}
    )
    grounder = object.__new__(ActionGrounder)
    grounder.env = env
    grounder.runtime_policy = runtime_policy
    grounder._executed_interactions = {
        ("task_01", "left_arm"): {
            "object_uid": "drawer",
            "link": "moving_link",
            "direction": torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
            "valid": torch.ones(2, dtype=torch.bool),
        }
    }
    grounder._departure_choices = {}
    grounder._departure_anchors = {}
    grounder._detached_hands = {}
    executor = object.__new__(ProgramExecutor)
    executor.env = env
    executor.adapter = AtomicActionAdapter(env)
    executor.grounder = grounder
    executor.runtime_policy = runtime_policy
    executor._interaction_clearance = SimpleNamespace(withdrawal=withdrawal)
    executor._detachment_states = {}
    step = SimpleNamespace(id="task_01", object_uid="drawer")
    return executor, step, geometry_calls, ik_calls, initial_qpos


def test_maximum_path_clearance_beats_the_original_direction_priority(
    monkeypatch,
):
    executor, step, calls, ik_calls, before = _fixture(
        monkeypatch, {"inverse_approach": [0.02, 0.02], "tool_back": [0.1, 0.1]}
    )

    distance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["valid"].tolist() == [True, True]
    assert choice["labels"] == ["tool_back", "tool_back"]
    torch.testing.assert_close(distance, torch.tensor([0.1, 0.1]))
    assert len(calls) == 4
    assert len(ik_calls) == 2
    torch.testing.assert_close(executor.env.robot.qpos, before)


def test_tool_back_is_selected_when_inverse_approach_is_geometrically_blocked(
    monkeypatch,
):
    executor, step, calls, _, _ = _fixture(monkeypatch, {"tool_back": [0.02, 0.02]})

    executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["tool_back", "tool_back"]
    torch.testing.assert_close(
        choice["direction"], _DIRECTIONS["tool_back"].repeat(2, 1)
    )
    assert [call["labels"][0] for call in calls] == list(_DIRECTIONS)


def test_table_blocked_candidates_never_reach_ik_and_world_up_can_escape(monkeypatch):
    executor, step, calls, ik_calls, _ = _fixture(
        monkeypatch,
        {
            "tool_back": [-0.002, -0.002],
            "baseward": [0.001, 0.001],
            "world_up": [0.02, 0.02],
        },
    )

    executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["world_up", "world_up"]
    assert [call["labels"][0] for call in calls] == list(_DIRECTIONS)
    assert [call["labels"][0] for call in ik_calls] == ["world_up"]
    assert all(call["distance"] == 0.1 for call in calls)


def test_clear_but_ik_failed_inverse_approach_tries_next_direction_with_rng_isolated(
    monkeypatch,
):
    executor, step, _, ik_calls, _ = _fixture(
        monkeypatch,
        {"inverse_approach": [0.02, 0.02], "tool_back": [0.03, 0.03]},
        ik_success={"inverse_approach": [False, False]},
    )
    before_rng = torch.random.get_rng_state().clone()

    clearance = executor._withdrawal_clearance(step, "left_arm")

    torch.testing.assert_close(torch.random.get_rng_state(), before_rng)
    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["tool_back", "tool_back"]
    torch.testing.assert_close(clearance, torch.tensor([0.03, 0.03]))
    assert [call["labels"][0] for call in ik_calls] == ["inverse_approach", "tool_back"]


def test_batched_environments_rank_path_clearances_independently(monkeypatch):
    executor, step, _, _, _ = _fixture(
        monkeypatch, {"inverse_approach": [0.05, -0.01], "world_up": [0.04, 0.03]}
    )

    clearance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["valid"].tolist() == [True, True]
    assert choice["labels"] == ["inverse_approach", "world_up"]
    torch.testing.assert_close(
        choice["direction"],
        torch.stack((_DIRECTIONS["inverse_approach"], _DIRECTIONS["world_up"])),
    )
    torch.testing.assert_close(clearance, torch.tensor([0.05, 0.03]))


def test_all_geometry_failed_directions_leave_no_valid_choice(monkeypatch):
    executor, step, calls, ik_calls, _ = _fixture(monkeypatch, {})

    clearance = executor._withdrawal_clearance(step, "left_arm")

    assert not executor.grounder._departure_choices[(step.id, "left_arm")][
        "valid"
    ].any()
    assert bool((clearance < 0.003).all())
    assert len(calls) == 4
    assert ik_calls == []


def test_positive_geometry_does_not_become_detachment_when_every_ik_candidate_fails(
    monkeypatch,
):
    executor, step, _, _, _ = _fixture(
        monkeypatch,
        {key: [0.02, 0.02] for key in _DIRECTIONS},
        ik_success={key: [False, False] for key in _DIRECTIONS},
    )
    clearance = executor._withdrawal_clearance(step, "left_arm")
    assert bool((clearance >= 0.003).all())
    assert not executor.grounder._departure_choices[(step.id, "left_arm")][
        "valid"
    ].any()
    executor._interaction_contacts = _Observer(
        {
            0.0: {
                "known": [True, True],
                "target_contact": [True, True],
                "robot_world_contact": [True, True],
            },
            **{float(tick): {"known": [True, True]} for tick in (1, 2, 3)},
        }
    )
    outcome = ActionOutcome(
        trajectory=executor.env.robot.qpos[:, None].repeat(1, 5, 1),
        success=torch.ones(2, dtype=torch.bool),
        next_state=ExecutionState(last_qpos=executor.env.robot.qpos.clone()),
        grounded=_grounded("detach", 2),
    )
    stop = executor._detachment_stop(
        step, {"left_arm": outcome}, {"left_arm": torch.ones(2, dtype=torch.bool)}
    )
    for tick in (1, 2, 3):
        executor.adapter._scene_time = float(tick)
        executor.env.robot.qpos[:, _HAND_IDS] = _hand(0.58)
        assert not stop(tick - 1).any()
    assert not executor._verify_articulation_detachment(step, "left_arm", outcome).any()


def test_grounder_uses_selected_direction_and_retry_keeps_the_original_endpoint(
    monkeypatch,
):
    executor, step, _, _, _ = _fixture(monkeypatch, {"world_up": [0.02, 0.02]})
    executor._withdrawal_clearance(step, "left_arm")
    reference = executor.env.get_current_xpos_agent()[0].clone()
    policy = {}

    first = executor.grounder._articulation_departure_target(
        step, "left_arm", reference, policy
    )

    target = reference.clone()
    target[:, 2, 3] += 0.1
    torch.testing.assert_close(first[:, -1], target)
    shifted = reference.clone()
    shifted[:, 0, 3] += 0.03
    shifted[:, 2, 3] += 0.02
    retry = executor.grounder._articulation_departure_target(
        step, "left_arm", shifted, {}
    )
    torch.testing.assert_close(retry[:, -1], target)
    assert policy["articulation_departure_valid"].tolist() == [True, True]


def test_search_updates_only_active_rows_and_preserves_an_already_selected_row(
    monkeypatch,
):
    clearances = {"inverse_approach": [0.02, -0.01], "tool_back": [0.04, 0.03]}
    executor, step, _, _, before = _fixture(monkeypatch, clearances)
    executor._withdrawal_clearance(step, "left_arm", active=torch.tensor([True, False]))
    initial = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert initial["valid"].tolist() == [True, False]
    direction = initial["direction"][0].clone()
    clearance = initial["clearance"][0].clone()
    label = initial["labels"][0]
    clearances["inverse_approach"] = [-0.02, -0.02]

    executor._withdrawal_clearance(step, "left_arm", active=torch.tensor([False, True]))

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["valid"].tolist() == [True, True]
    assert choice["labels"] == [label, "tool_back"]
    torch.testing.assert_close(choice["direction"][0], direction)
    torch.testing.assert_close(choice["clearance"][0], clearance)
    torch.testing.assert_close(executor.env.robot.qpos, before)


def test_reexecution_invalidates_only_affected_choice_and_anchor_rows(monkeypatch):
    executor, step, _, _, _ = _fixture(monkeypatch, {"tool_back": [0.02, 0.02]})
    executor._withdrawal_clearance(step, "left_arm")
    grounder = executor.grounder
    grounder._articulation_departure_target(
        step, "left_arm", executor.env.get_current_xpos_agent()[0], {}
    )
    key = (step.id, "left_arm")
    old_direction = grounder._departure_choices[key]["direction"][0].clone()
    new_interaction = replace(
        _grounded("detach", 2),
        motion_policy={
            "articulation_approach_direction_local": torch.tensor(
                [[0.0, 1.0, 0.0]]
            ).repeat(2, 1),
            "articulation_target_link_name": "moving_link",
        },
    )

    grounder._record_executed_interaction(
        step.id, new_interaction, torch.tensor([False, True])
    )

    assert grounder._departure_choices[key]["valid"].tolist() == [True, False]
    assert grounder._departure_anchors[key]["valid"].tolist() == [True, False]
    torch.testing.assert_close(
        grounder._departure_choices[key]["direction"][0], old_direction
    )


def test_reset_removes_choice_provenance_and_cached_departure_endpoints(monkeypatch):
    executor, step, _, _, _ = _fixture(monkeypatch, {"world_up": [0.02, 0.02]})
    executor._withdrawal_clearance(step, "left_arm")
    grounder = executor.grounder
    grounder._articulation_departure_target(
        step, "left_arm", executor.env.get_current_xpos_agent()[0], {}
    )
    grounder._detached_hands[(step.id, "left_arm")] = _hand([0.7, 0.7])

    grounder._clear_interaction_cleanup()

    assert grounder._departure_choices == {}
    assert grounder._departure_anchors == {}
    assert grounder._executed_interactions == {}
    assert grounder._detached_hands == {}


def test_staged_departure_without_a_verified_choice_fails_closed(monkeypatch):
    executor, step, _, _, _ = _fixture(monkeypatch, {})
    executor.grounder._detached_hands[(step.id, "left_arm")] = _hand([0.7, 0.7])

    with pytest.raises(ValueError, match="choice"):
        executor.grounder._articulation_departure_target(
            step, "left_arm", executor.env.get_current_xpos_agent()[0], {}
        )


def test_invalid_choice_row_holds_pose_without_falling_back_to_inverse_approach(
    monkeypatch,
):
    executor, step, _, _, _ = _fixture(monkeypatch, {"tool_back": [0.02, 0.02]})
    executor._withdrawal_clearance(step, "left_arm")
    key = (step.id, "left_arm")
    executor.grounder._departure_choices[key]["valid"][1] = False
    executor.grounder._detached_hands[key] = _hand([0.7, 0.7])
    reference = executor.env.get_current_xpos_agent()[0].clone()
    policy = {}

    trajectory = executor.grounder._articulation_departure_target(
        step, "left_arm", reference, policy
    )

    assert policy["articulation_departure_valid"].tolist() == [True, False]
    torch.testing.assert_close(trajectory[1, -1], reference[1])
    torch.testing.assert_close(
        trajectory[0, -1, :3, 3], reference[0, :3, 3] + _DIRECTIONS["tool_back"] * 0.1
    )


def test_legacy_cleanup_without_detached_hand_keeps_inverse_approach(monkeypatch):
    executor, step, _, _, _ = _fixture(monkeypatch, {})
    reference = executor.env.get_current_xpos_agent()[0].clone()
    policy = {}

    trajectory = executor.grounder._articulation_departure_target(
        step, "left_arm", reference, policy
    )

    torch.testing.assert_close(
        trajectory[:, -1, :3, 3],
        reference[:, :3, 3] + _DIRECTIONS["inverse_approach"] * 0.1,
    )
    assert policy["articulation_departure_valid"].tolist() == [True, True]


def test_tied_path_clearance_prefers_a_wider_terminal_clearance_per_environment(
    monkeypatch,
):
    executor, step, calls, _, _ = _fixture(
        monkeypatch,
        {"inverse_approach": [0.01, 0.01], "tool_back": [0.01, 0.01]},
        terminal_clearances={
            "inverse_approach": [0.015, 0.08],
            "tool_back": [0.10, 0.04],
        },
    )

    clearance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["tool_back", "inverse_approach"]
    assert choice["valid"].tolist() == [True, True]
    torch.testing.assert_close(clearance, torch.tensor([0.01, 0.01]))
    assert len(calls) == 4


def test_path_tie_tolerance_is_numeric_not_a_large_safety_relaxation(monkeypatch):
    executor, step, _, _, _ = _fixture(
        monkeypatch,
        {"inverse_approach": [0.0100005, 0.010002], "tool_back": [0.01, 0.01]},
        terminal_clearances={
            "inverse_approach": [0.02, 0.02],
            "tool_back": [0.10, 0.10],
        },
    )

    clearance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["tool_back", "inverse_approach"]
    torch.testing.assert_close(clearance, torch.tensor([0.01, 0.010002]))


def test_larger_terminal_clearance_cannot_hide_a_worse_path_minimum(monkeypatch):
    executor, step, _, _, _ = _fixture(
        monkeypatch,
        {"inverse_approach": [0.02, 0.02], "tool_back": [0.01, 0.01]},
        terminal_clearances={
            "inverse_approach": [0.021, 0.021],
            "tool_back": [0.5, 0.5],
        },
    )

    clearance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["inverse_approach", "inverse_approach"]
    torch.testing.assert_close(clearance, torch.tensor([0.02, 0.02]))


def test_exact_score_ties_keep_priority_only_after_all_four_candidates_are_checked(
    monkeypatch,
):
    executor, step, calls, ik_calls, _ = _fixture(
        monkeypatch,
        {label: [0.02, 0.02] for label in _DIRECTIONS},
        terminal_clearances={label: [0.04, 0.04] for label in _DIRECTIONS},
    )

    executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["inverse_approach", "inverse_approach"]
    assert [call["labels"][0] for call in calls] == list(_DIRECTIONS)
    assert [call["labels"][0] for call in ik_calls] == list(_DIRECTIONS)


def test_high_scores_cannot_admit_geometry_or_ik_failed_directions(monkeypatch):
    executor, step, _, ik_calls, _ = _fixture(
        monkeypatch,
        {
            "inverse_approach": [0.02, 0.02],
            "tool_back": [0.10, 0.10],
            "baseward": [0.0029, 0.0029],
            "world_up": [0.015, 0.015],
        },
        ik_success={"tool_back": [False, False]},
        terminal_clearances={
            "tool_back": [0.5, 0.5],
            "baseward": [1.0, 1.0],
            "world_up": [0.9, 0.9],
        },
    )

    clearance = executor._withdrawal_clearance(step, "left_arm")

    choice = executor.grounder._departure_choices[(step.id, "left_arm")]
    assert choice["labels"] == ["inverse_approach", "inverse_approach"]
    torch.testing.assert_close(clearance, torch.tensor([0.02, 0.02]))
    assert [call["labels"][0] for call in ik_calls] == [
        "inverse_approach",
        "tool_back",
        "world_up",
    ]
