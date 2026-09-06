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

import math
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import interaction_clearance as module


@pytest.fixture
def fixture(monkeypatch):
    batch_size = 2
    tcp = torch.eye(4).repeat(batch_size, 1, 1)
    poses = {name: torch.eye(4).repeat(batch_size, 1, 1) for name in ("base", "moving")}
    meshes = {
        name: (
            torch.tensor([[index, 0.0, 0.0], [index, 1.0, 0.0], [index, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2]]),
        )
        for index, name in enumerate(poses)
    }
    pose_calls = []

    def link_pose(name, to_matrix=False):
        assert to_matrix
        pose_calls.append(name)
        return poses[name]

    articulation = SimpleNamespace(link_names=list(poses), get_link_pose=link_pose)
    qpos = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    env = SimpleNamespace(
        robot=SimpleNamespace(get_qpos=lambda: qpos),
        sim=SimpleNamespace(
            get_articulation=lambda uid: articulation if uid == "target" else None
        ),
        get_current_xpos_agent=lambda: (tcp, tcp.clone()),
    )
    hand = torch.zeros(batch_size, 1, 1, 3)
    geometry_calls = []

    def gripper_points(actual_env, arm, *, qpos, target_qpos=None):
        assert actual_env is env
        geometry_calls.append(
            (arm, qpos.clone(), None if target_qpos is None else target_qpos.clone())
        )
        return data.hand

    monkeypatch.setattr(module, "_gripper_points", gripper_points)
    monkeypatch.setattr(
        module, "_scaled_link_geometry", lambda target, link: meshes[link]
    )
    checkers = []
    query_calls = []
    data = SimpleNamespace(
        env=env,
        poses=poses,
        tcp=tcp,
        hand=hand,
        meshes=meshes,
        articulation=articulation,
        checkers=checkers,
        query_calls=query_calls,
        pose_calls=pose_calls,
        qpos=qpos,
        geometry_calls=geometry_calls,
        distances=lambda points, mesh_index: points[..., 2] + mesh_index,
    )

    class Checker:
        def __init__(self, base_mesh_verts, base_mesh_faces):
            self.index = int(base_mesh_verts[0, 0])
            checkers.append(self)

        def query_batch_points(
            self, batch_points, collision_threshold=0.0, is_visual=False
        ):
            assert not is_visual
            query_calls.append((self.index, batch_points.clone(), collision_threshold))
            distances = data.distances(batch_points, self.index)
            return distances <= collision_threshold, distances

    monkeypatch.setattr(module, "ConvexCollisionChecker", Checker)
    return data


def test_closure_excludes_only_handle_not_the_rest_of_moving_link(fixture, monkeypatch):
    calls = []

    def named(target, link, *, mesh_names=(), excluded_mesh_names=()):
        calls.append((link, mesh_names, excluded_mesh_names))
        assert target is fixture.articulation and link == "moving"
        if mesh_names:
            assert mesh_names == ("handle",)
            return fixture.meshes["moving"]
        assert excluded_mesh_names == ("handle",)
        vertices, faces = fixture.meshes["moving"]
        return vertices + torch.tensor([1.0, 0.0, 0.0]), faces

    monkeypatch.setattr(module, "_named_link_geometry", named, raising=False)
    fixture.distances = lambda points, index: (
        torch.tensor([[-0.001], [0.01]]).expand(points.shape[:2])
        if index == 2
        else torch.full(points.shape[:2], 0.02)
    )
    checker = module._InteractionClearance(fixture.env)
    result = checker.grasp_closure(
        "target", "moving", "handle", fixture.tcp, fixture.hand.repeat(1, 3, 1, 1)
    )
    torch.testing.assert_close(result, torch.tensor([-0.001, 0.01]))
    assert calls == [("moving", ("handle",), ()), ("moving", (), ("handle",))]
    assert [x[0] for x in fixture.query_calls] == [0, 2]
    assert checker._link_checker("target", fixture.articulation, "moving").index == 1


def test_closure_preserves_per_row_support_sweep(fixture, monkeypatch):
    monkeypatch.setattr(
        module,
        "_named_link_geometry",
        lambda *args, **kwargs: fixture.meshes["moving"],
        raising=False,
    )
    fixture.distances = lambda points, index: torch.full(points.shape[:2], 0.02)
    fixture.hand = torch.zeros(2, 3, 1, 3)
    fixture.hand[:, :, 0, 2] = torch.tensor([[0.02, 0.001, 0.03], [0.02, 0.008, 0.03]])
    result = module._InteractionClearance(fixture.env).grasp_closure(
        "target",
        "moving",
        "handle",
        fixture.tcp,
        fixture.hand,
        support_surface_z=torch.zeros(2),
    )
    torch.testing.assert_close(result, torch.tensor([0.001, 0.008]))


def test_withdrawal_uses_actual_tcp_and_each_live_link_frame_without_mixing_rows(
    fixture,
) -> None:
    fixture.articulation.link_names = ["base"]
    fixture.hand[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    fixture.hand[1, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
    fixture.tcp[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    fixture.tcp[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    fixture.poses["base"][0, :3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    fixture.poses["base"][1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    fixture.poses["base"][:, :3, 3] = torch.tensor(
        [[4.0, 5.0, 6.0], [-4.0, -5.0, -6.0]]
    )
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]]),
        distance=0.4,
    )
    _, queried, threshold = fixture.query_calls[0]
    sample_count = max(5, math.ceil(0.4 / 0.003) + 1)
    expected = torch.tensor([[[3.0, 2.0, -3.0]], [[4.0, -3.0, 3.0]]]).repeat(
        1, sample_count, 1
    )
    expected[0, :, 0] -= torch.linspace(0.0, 0.4, sample_count)
    expected[1, :, 2] += torch.linspace(0.0, 0.4, sample_count)
    torch.testing.assert_close(queried, expected)
    torch.testing.assert_close(result, torch.tensor([-3.0, 3.0]))
    assert threshold == 0.003
    assert fixture.geometry_calls[0][0] == "left_arm"
    torch.testing.assert_close(fixture.geometry_calls[0][1], fixture.qpos)
    assert fixture.geometry_calls[0][2] is None


def test_withdrawal_forwards_hold_target_without_replacing_raw_observation(
    fixture,
) -> None:
    raw_before = fixture.qpos.clone()
    hold_target = torch.tensor([[0.61, -0.61], [0.51, -0.51]])
    target_before = hold_target.clone()
    module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=0.04,
        hand_target_qpos=hold_target,
    )
    _, observed, target = fixture.geometry_calls[0]
    torch.testing.assert_close(observed, raw_before)
    torch.testing.assert_close(target, target_before)
    assert not torch.equal(observed, target)
    torch.testing.assert_close(fixture.qpos, raw_before)
    torch.testing.assert_close(hold_target, target_before)


def test_withdrawal_includes_unsafe_command_tracking_sample_without_mixing_envs(
    fixture,
) -> None:
    fixture.hand = torch.zeros(2, 3, 2, 3)
    fixture.hand[..., 2] = 0.02
    # Current and target geometry are clear; the intermediate shape is not in env 0.
    fixture.hand[0, 1, 1, 2] = -0.005
    fixture.hand[1, 1, 1, 2] = 0.006
    raw_before = fixture.qpos.clone()
    target = raw_before.clone()
    target[:, 1] = torch.tensor([-0.2, -0.4])
    target_before = target.clone()
    result = module._InteractionClearance(fixture.env).withdrawal(
        "right_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=0.04,
        hand_target_qpos=target,
    )
    torch.testing.assert_close(result, torch.tensor([-0.005, 0.006]))
    assert (result >= 0.003).tolist() == [False, True]
    path_samples = max(5, math.ceil(0.04 / 0.003) + 1)
    assert all(
        points.shape == (2, path_samples * 3 * 2, 3)
        for _, points, _ in fixture.query_calls
    )
    torch.testing.assert_close(fixture.geometry_calls[0][1], raw_before)
    torch.testing.assert_close(fixture.geometry_calls[0][2], target_before)
    torch.testing.assert_close(fixture.qpos, raw_before)
    torch.testing.assert_close(target, target_before)


def test_withdrawal_includes_interior_path_sample_and_all_target_links(fixture) -> None:
    def distances(points, mesh_index):
        values = torch.full(points.shape[:2], 0.05)
        if mesh_index == 1:
            values[0, 2] = -0.01
            values[1, 2] = 0.002
        return values

    fixture.distances = distances
    result = module._InteractionClearance(fixture.env).withdrawal(
        "right_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=0.04,
    )
    torch.testing.assert_close(result, torch.tensor([-0.01, 0.002]))
    assert [call[0] for call in fixture.query_calls] == [0, 1]
    assert all(call[2] == 0.003 for call in fixture.query_calls)
    assert fixture.geometry_calls[0][0] == "right_arm"


def test_withdrawal_diagnostics_separate_initial_support_path_and_target(
    fixture,
) -> None:
    fixture.hand = torch.zeros(2, 3, 2, 3)
    fixture.hand[0, 1, 1, 2] = -0.02
    fixture.hand[1, 1, 1, 2] = -0.03
    fixture.tcp[:, 2, 3] = torch.tensor([0.1, 0.2])
    fixture.distances = lambda points, _index: torch.tensor([0.04, 0.002])[
        :, None
    ].expand(points.shape[:2])
    checker = module._InteractionClearance(fixture.env)
    kwargs = {
        "direction": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
        "distance": 0.04,
        "hand_target_qpos": fixture.qpos + 0.01,
        "support_surface_z": torch.tensor([0.05, 0.1]),
    }
    diagnostics = {"candidate": "kept"}

    result = checker.withdrawal(
        "right_arm", "target", diagnostics=diagnostics, **kwargs
    )
    without_diagnostics = checker.withdrawal("right_arm", "target", **kwargs)

    torch.testing.assert_close(result, without_diagnostics)
    torch.testing.assert_close(result, torch.tensor([-0.01, 0.002]))
    torch.testing.assert_close(
        diagnostics["target_clearance"], torch.tensor([0.04, 0.002])
    )
    torch.testing.assert_close(
        diagnostics["support_clearance"], torch.tensor([-0.01, 0.07])
    )
    torch.testing.assert_close(
        diagnostics["initial_support_clearance"], torch.tensor([0.03, 0.07])
    )
    assert diagnostics["candidate"] == "kept"
    for key in ("target_clearance", "support_clearance", "initial_support_clearance"):
        assert diagnostics[key].shape == (2,)
        assert not diagnostics[key].requires_grad
        diagnostics[key].fill_(1.0)
    torch.testing.assert_close(result, torch.tensor([-0.01, 0.002]))


def test_withdrawal_diagnostics_leave_unconfigured_support_unmeasured(fixture) -> None:
    diagnostics = {"support_clearance": "stale", "initial_support_clearance": "stale"}
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.ones(2, 3),
        distance=0.04,
        diagnostics=diagnostics,
    )
    torch.testing.assert_close(diagnostics["target_clearance"], result)
    assert diagnostics["support_clearance"] is None
    assert diagnostics["initial_support_clearance"] is None
    diagnostics["target_clearance"].fill_(1.0)
    torch.testing.assert_close(result, torch.zeros(2))


@pytest.mark.parametrize("with_support", [False, True])
def test_withdrawal_terminal_clearance_uses_last_path_pose_and_all_hand_shapes(
    fixture, with_support: bool
) -> None:
    fixture.hand = torch.zeros(2, 3, 2, 3)
    fixture.hand[0, 1, 1, 2] = -0.02
    fixture.hand[1, 1, 1, 2] = -0.03
    fixture.tcp[:, 2, 3] = torch.tensor([0.1, 0.2])
    fixture.distances = lambda points, index: points[..., 2] - index * 0.01
    diagnostics = {}
    kwargs = {
        "direction": torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        "distance": 0.04,
        "hand_target_qpos": fixture.qpos + 0.01,
        "support_surface_z": torch.tensor([0.0, 0.04]) if with_support else None,
    }
    checker = module._InteractionClearance(fixture.env)

    result = checker.withdrawal("left_arm", "target", diagnostics=diagnostics, **kwargs)
    queried = list(fixture.query_calls)
    without_diagnostics = checker.withdrawal("left_arm", "target", **kwargs)

    torch.testing.assert_close(result, without_diagnostics)
    torch.testing.assert_close(
        result, torch.tensor([0.07, 0.13 if with_support else 0.16])
    )
    torch.testing.assert_close(
        diagnostics["target_clearance"], torch.tensor([0.07, 0.16])
    )
    expected_terminal = torch.tensor([0.11, 0.17 if with_support else 0.20])
    torch.testing.assert_close(diagnostics["terminal_clearance"], expected_terminal)
    if with_support:
        torch.testing.assert_close(
            diagnostics["support_clearance"], torch.tensor([0.08, 0.13])
        )
        torch.testing.assert_close(
            diagnostics["initial_support_clearance"], torch.tensor([0.08, 0.13])
        )
    else:
        assert diagnostics["support_clearance"] is None
        assert diagnostics["initial_support_clearance"] is None
    assert [index for index, _points, _threshold in queried] == [0, 1, 0, 1]
    for _index, points, threshold in queried[2:]:
        assert points.shape == (2, 3 * 2, 3)
        torch.testing.assert_close(points[..., 2].amin(1), torch.tensor([0.12, 0.21]))
        assert threshold == 0.003
    assert not diagnostics["terminal_clearance"].requires_grad
    diagnostics["terminal_clearance"].fill_(2.0)
    torch.testing.assert_close(result, without_diagnostics)


def test_withdrawal_reuses_checkers_but_updates_moving_link_pose(fixture) -> None:
    clearance = module._InteractionClearance(fixture.env)
    direction = torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1)
    before = clearance.withdrawal(
        "left_arm", "target", direction=direction, distance=0.04
    )
    fixture.poses["base"][:, 2, 3] = 0.1
    after = clearance.withdrawal(
        "left_arm", "target", direction=direction, distance=0.04
    )
    assert len(fixture.checkers) == 2
    assert fixture.pose_calls == ["base", "moving", "base", "moving"]
    torch.testing.assert_close(before, torch.zeros(2))
    torch.testing.assert_close(after, torch.full((2,), -0.1))


@pytest.mark.parametrize(
    "direction",
    [
        torch.zeros(2, 3),
        torch.ones(3),
        torch.ones(1, 3),
        torch.full((2, 3), float("nan")),
    ],
)
def test_withdrawal_rejects_invalid_direction(fixture, direction) -> None:
    with pytest.raises(ValueError, match="direction"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm", "target", direction=direction, distance=0.04
        )
    assert not fixture.query_calls


@pytest.mark.parametrize("distance", [-0.01, float("nan"), float("inf"), True])
def test_withdrawal_rejects_invalid_distance(fixture, distance) -> None:
    with pytest.raises(ValueError, match="distance"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm", "target", direction=torch.ones(2, 3), distance=distance
        )


@pytest.mark.parametrize("bad_geometry", ["empty", "nonfinite", "no_faces"])
def test_withdrawal_fails_closed_on_missing_target_geometry(
    fixture, bad_geometry
) -> None:
    if bad_geometry == "empty":
        fixture.meshes["moving"] = (
            torch.empty(0, 3),
            torch.empty(0, 3, dtype=torch.int64),
        )
    elif bad_geometry == "nonfinite":
        fixture.meshes["moving"][0][0, 0] = float("nan")
    else:
        fixture.meshes["moving"] = (
            fixture.meshes["moving"][0],
            torch.empty(0, 3, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="geometry"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm", "target", direction=torch.ones(2, 3), distance=0.04
        )


def test_withdrawal_rejects_nonfinite_checker_distances(fixture) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], float("nan"))
    with pytest.raises(ValueError, match="distance"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm", "target", direction=torch.ones(2, 3), distance=0.04
        )


@pytest.mark.parametrize("distance", [0.0029, 0.003, 0.0031])
def test_withdrawal_preserves_signed_distance_at_contact_clearance_threshold(
    fixture, distance
) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], distance)
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.ones(2, 3),
        distance=0.04,
    )
    torch.testing.assert_close(result, torch.full((2,), distance))
    assert all(call[2] == 0.003 for call in fixture.query_calls)


def test_withdrawal_fails_closed_on_absent_target(fixture) -> None:
    with pytest.raises(ValueError, match="geometry"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm",
            "missing",
            direction=torch.ones(2, 3),
            distance=0.04,
        )
    assert not fixture.checkers


def test_withdrawal_does_not_accept_nonfinite_live_link_pose(fixture) -> None:
    fixture.poses["moving"][0, 2, 3] = float("nan")
    with pytest.raises(ValueError, match="poses"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm",
            "target",
            direction=torch.ones(2, 3),
            distance=0.04,
        )


def test_withdrawal_detects_thin_handle_between_old_path_samples(fixture) -> None:
    fixture.articulation.link_names = ["base"]
    # A 2 mm slab at 6 mm is missed by the former 25 mm path spacing.
    fixture.distances = lambda points, index: (points[..., 0] - 0.006).abs() - 0.001
    old_grid = torch.linspace(0.0, 0.1, 5)
    assert ((old_grid - 0.006).abs() - 0.001).min() > 0.003
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=0.1,
    )
    assert (result < 0.0).tolist() == [True, True]
    _, queried, threshold = fixture.query_calls[0]
    assert threshold == 0.003
    assert queried.shape[1] == max(5, math.ceil(0.1 / 0.003) + 1)
    assert (queried[:, 1:, 0] - queried[:, :-1, 0]).max() <= 0.003


@pytest.mark.parametrize("distance", [0.0, 0.001, 0.012, 0.04, 0.1])
def test_withdrawal_bounds_spatial_sample_spacing_without_changing_threshold(
    fixture, distance
) -> None:
    fixture.articulation.link_names = ["base"]
    module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=distance,
    )
    _, points, threshold = fixture.query_calls[0]
    assert points.shape == (2, max(5, math.ceil(distance / 0.003) + 1), 3)
    torch.testing.assert_close(points[:, 0, 0], torch.zeros(2))
    torch.testing.assert_close(points[:, -1, 0], torch.full((2,), distance))
    assert (points[:, 1:, 0] - points[:, :-1, 0]).max() <= 0.003 + 1.0e-7
    assert threshold == 0.003


def _precontact_target(fixture):
    chains = {"base": (), "moving": (SimpleNamespace(name="door_hinge"),)}
    fixture.articulation.get_parent_joint_chain = lambda link: chains[link]
    return chains


def test_precontact_checks_root_and_other_branch_but_excludes_fixed_handle_subtree(
    fixture,
) -> None:
    chains = _precontact_target(fixture)
    chains["fixed_handle"] = (
        SimpleNamespace(name="handle_fixed"),
        SimpleNamespace(name="door_hinge"),
    )
    chains["other_door"] = (SimpleNamespace(name="other_hinge"),)
    fixture.articulation.link_names += ["fixed_handle", "other_door"]
    fixture.poses["other_door"] = torch.eye(4).repeat(2, 1, 1)
    fixture.meshes["other_door"] = (
        fixture.meshes["base"][0].clone(),
        fixture.meshes["base"][1],
    )
    fixture.meshes["other_door"][0][:, 0] = 3.0
    # The excluded subtree must not require a checker or pose at all.
    del fixture.meshes["moving"]
    tcp = torch.eye(4).repeat(2, 3, 1, 1)
    tcp[:, :, 2, 3] = 0.1
    result = module._InteractionClearance(fixture.env).precontact(
        "target",
        "door_hinge",
        tcp,
        torch.zeros(2, 1, 3),
    )
    torch.testing.assert_close(result, torch.full((2,), 0.1))
    assert fixture.pose_calls == ["base", "other_door"]
    assert [call[0] for call in fixture.query_calls] == [0, 3]
    assert all(call[2] == 0.003 for call in fixture.query_calls)
    assert not fixture.geometry_calls


def test_precontact_detects_interior_fk_collision_and_preserves_environment_rows(
    fixture,
) -> None:
    _precontact_target(fixture)
    tcp = torch.eye(4).repeat(2, 3, 1, 1)
    tcp[:, :, 2, 3] = 0.1
    tcp[0, 1, 2, 3] = 0.02
    hand = torch.tensor([[[0.0, 0.0, -0.03]], [[0.0, 0.0, -0.01]]])
    result = module._InteractionClearance(fixture.env).precontact(
        "target", "door_hinge", tcp, hand
    )
    torch.testing.assert_close(result, torch.tensor([-0.01, 0.09]))
    _, points, _ = fixture.query_calls[0]
    assert points.shape == (2, 3, 3)
    torch.testing.assert_close(
        points[:, :, 2], torch.tensor([[0.07, -0.01, 0.07], [0.09, 0.09, 0.09]])
    )


def test_precontact_rotates_hand_and_queries_each_updated_live_frame(fixture) -> None:
    _precontact_target(fixture)
    tcp = torch.eye(4).repeat(2, 2, 1, 1)
    tcp[:, :, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    tcp[:, :, 2, 3] = 0.1
    hand = torch.tensor([[[0.01, 0.0, 0.0]], [[0.02, 0.0, 0.0]]])
    clearance = module._InteractionClearance(fixture.env)
    before = clearance.precontact("target", "door_hinge", tcp, hand)
    fixture.poses["base"][:, 2, 3] = torch.tensor([0.01, 0.03])
    after = clearance.precontact("target", "door_hinge", tcp, hand)
    torch.testing.assert_close(before, torch.tensor([0.09, 0.08]))
    torch.testing.assert_close(after, torch.tensor([0.08, 0.05]))
    assert len(fixture.checkers) == 1
    assert fixture.pose_calls == ["base", "base"]


@pytest.mark.parametrize(
    "invalid",
    [
        "tcp_shape",
        "tcp_empty",
        "tcp_nonfinite",
        "point_batch",
        "point_empty",
        "point_nonfinite",
    ],
)
def test_precontact_rejects_invalid_pose_or_point_inputs(fixture, invalid) -> None:
    _precontact_target(fixture)
    tcp = torch.eye(4).repeat(2, 3, 1, 1)
    points = torch.zeros(2, 2, 3)
    if invalid == "tcp_shape":
        tcp = tcp[:, 0]
    elif invalid == "tcp_empty":
        tcp = tcp[:, :0]
    elif invalid == "tcp_nonfinite":
        tcp[0, 0, 0, 0] = float("nan")
    elif invalid == "point_batch":
        points = points[:1]
    elif invalid == "point_empty":
        points = points[:, :0]
    else:
        points[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Precontact"):
        module._InteractionClearance(fixture.env).precontact(
            "target", "door_hinge", tcp, points
        )
    assert not fixture.query_calls


@pytest.mark.parametrize("target_joint", ["missing", ""])
def test_precontact_rejects_unknown_or_empty_target_joint(
    fixture, target_joint
) -> None:
    _precontact_target(fixture)
    with pytest.raises(ValueError, match="joint"):
        module._InteractionClearance(fixture.env).precontact(
            "target",
            target_joint,
            torch.eye(4).repeat(2, 1, 1, 1),
            torch.zeros(2, 1, 3),
        )


def test_precontact_fails_closed_if_no_non_target_geometry_can_be_audited(
    fixture,
) -> None:
    _precontact_target(fixture)
    fixture.articulation.link_names = ["moving"]
    with pytest.raises(ValueError, match="geometry"):
        module._InteractionClearance(fixture.env).precontact(
            "target",
            "door_hinge",
            torch.eye(4).repeat(2, 1, 1, 1),
            torch.zeros(2, 1, 3),
        )


def test_withdrawal_support_plane_rejects_target_clear_path_into_table(fixture) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], 0.2)
    fixture.tcp[:, 2, 3] = 1.04
    direction = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    clearance = module._InteractionClearance(fixture.env)
    without_table = clearance.withdrawal(
        "left_arm", "target", direction=direction, distance=0.1
    )
    with_table = clearance.withdrawal(
        "left_arm",
        "target",
        direction=direction,
        distance=0.1,
        support_surface_z=torch.tensor(1.0),
    )
    torch.testing.assert_close(without_table, torch.full((2,), 0.2))
    torch.testing.assert_close(with_table, torch.full((2,), -0.06))


@pytest.mark.parametrize(
    "surface", [torch.tensor(1.0), torch.tensor([1.0]), torch.tensor([1.0, 1.0])]
)
def test_withdrawal_upward_path_preserves_support_margin(fixture, surface) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], 0.2)
    fixture.tcp[:, 2, 3] = 1.04
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        distance=0.1,
        support_surface_z=surface,
    )
    torch.testing.assert_close(result, torch.full((2,), 0.04))
    assert (result >= 0.003).tolist() == [True, True]


def test_withdrawal_support_plane_includes_all_hand_samples_without_env_mixing(
    fixture,
) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], 0.2)
    fixture.tcp[:, 2, 3] = torch.tensor([1.04, 2.06])
    fixture.hand = torch.zeros(2, 3, 2, 3)
    fixture.hand[0, 1, 1, 2] = -0.05
    fixture.hand[1, 1, 1, 2] = -0.02
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        distance=0.04,
        hand_target_qpos=fixture.qpos.clone(),
        support_surface_z=torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(result, torch.tensor([-0.01, 0.04]))
    assert (result >= 0.003).tolist() == [False, True]


def test_withdrawal_support_plane_does_not_hide_smaller_target_clearance(
    fixture,
) -> None:
    fixture.distances = lambda points, index: torch.full(points.shape[:2], -0.01)
    fixture.tcp[:, 2, 3] = 1.5
    result = module._InteractionClearance(fixture.env).withdrawal(
        "left_arm",
        "target",
        direction=torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        distance=0.04,
        support_surface_z=torch.tensor([1.0]),
    )
    torch.testing.assert_close(result, torch.full((2,), -0.01))


@pytest.mark.parametrize(
    "surface",
    [
        torch.ones(3),
        torch.ones(2, 1),
        torch.empty(0),
        torch.tensor(float("nan")),
        torch.tensor([1.0, float("inf")]),
        torch.tensor(True),
    ],
)
def test_withdrawal_rejects_invalid_support_surface(fixture, surface) -> None:
    with pytest.raises(ValueError, match="support_surface_z"):
        module._InteractionClearance(fixture.env).withdrawal(
            "left_arm",
            "target",
            direction=torch.ones(2, 3),
            distance=0.04,
            support_surface_z=surface,
        )
