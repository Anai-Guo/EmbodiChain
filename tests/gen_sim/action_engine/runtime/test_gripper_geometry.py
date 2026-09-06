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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.gripper_profiles import get_gripper_profile
from embodichain.gen_sim.action_engine.runtime import gripper_geometry
from embodichain.gen_sim.action_engine.runtime.gripper_geometry import (
    _gripper_points,
    _release_sweep_points,
)


class _Robot:
    def __init__(self, *, scale=(1.0, 1.0, 1.0), num_envs=1) -> None:
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(fpath="fixture.usd", body_scale=scale)
        self.profile = get_gripper_profile("pgi")
        self.joint_names = [
            "right_arm_joint",
            "left_gripper_finger2_joint_1",
            "left_arm_joint",
            "right_gripper_finger1_joint_1",
            "left_gripper_finger1_joint_1",
            "right_gripper_finger2_joint_1",
        ]
        self.qpos = torch.tensor([[0.3, 0.0, -0.4, 0.017, 0.0, 0.017]]).repeat(
            num_envs, 1
        )
        self.link_names = ["root"] + [
            f"{side}_{link}"
            for side in ("left", "right")
            for link in ("mount", "palm", "finger1", "finger2", "pad")
        ]
        self.root_pose = torch.eye(4).repeat(num_envs, 1, 1)
        self.root_pose[:, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.root_pose[:, :3, 3] = torch.tensor([10.0, -5.0, 3.0])
        self.root_pose[:, 0, 3] += torch.arange(num_envs) * 2.0
        self.mesh = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
        self.fk_calls = []
        self.mesh_calls = []

    def get_solver(self, name):
        return SimpleNamespace(
            cfg=SimpleNamespace(end_link_name=name.replace("arm", "mount"))
        )

    def get_qpos(self, name=None, target=False):
        assert target is False
        return self.qpos if name is None else self.qpos[:, self.get_joint_ids(name)]

    def get_joint_ids(self, name):
        return [self.joint_names.index(f"{name}_joint")]

    @property
    def root_link_name(self):
        pytest.fail(
            "Release geometry must not access the broken root_link_name property."
        )

    def get_local_pose(self, to_matrix=False):
        assert to_matrix
        return self.root_pose

    def get_link_pose(self, *_args, **_kwargs):
        pytest.fail("Release geometry must use public get_local_pose for the root.")

    def get_parent_joint_chain(self, link_name):
        if link_name == "root":
            return ()
        side, suffix = link_name.split("_", 1)
        result = [
            SimpleNamespace(
                name=f"{side}_arm_joint",
                parent_link_name="root",
                child_link_name=f"{side}_mount",
            )
        ]
        if suffix != "mount":
            joint_name = {
                "finger1": f"{side}_gripper_finger1_joint_1",
                "finger2": f"{side}_gripper_finger2_joint_1",
                "pad": f"{side}_gripper_finger1_joint_1",
            }.get(suffix, f"{side}_{suffix}_fixed")
            result.insert(
                0,
                SimpleNamespace(
                    name=joint_name,
                    parent_link_name=f"{side}_mount",
                    child_link_name=link_name,
                ),
            )
        return tuple(result)

    def get_link_vert_face(self, link_name):
        self.mesh_calls.append(link_name)
        if link_name.endswith("mount"):
            return torch.empty(0, 3), torch.empty(0, 3, dtype=torch.int64)
        return self.mesh, torch.tensor([[0, 1, 2]])

    @staticmethod
    def _mount_pose():
        pose = torch.eye(4)
        pose[:3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        pose[:3, 3] = torch.tensor([0.5, 0.2, 0.4])
        return pose

    def compute_fk(
        self, qpos, name=None, link_names=None, qpos_joint_names=None, to_matrix=False
    ):
        mount = self._mount_pose()
        if name is not None:
            assert to_matrix
            return self.root_pose @ mount @ torch.tensor(self.profile.tcp_transform)
        assert tuple(qpos_joint_names) == tuple(self.joint_names)
        self.fk_calls.append((qpos.clone(), tuple(link_names)))
        result = torch.eye(4).repeat(qpos.shape[0], len(link_names), 1, 1)
        for i, link in enumerate(link_names):
            side, suffix = link.split("_", 1)
            if suffix in {"finger1", "pad"}:
                result[:, i, 0, 3] = qpos[
                    :, self.joint_names.index(f"{side}_gripper_finger1_joint_1")
                ]
            elif suffix == "finger2":
                result[:, i, 0, 3] = -qpos[
                    :, self.joint_names.index(f"{side}_gripper_finger2_joint_1")
                ]
            if suffix == "pad":
                result[:, i, 2, 3] = 0.04
        return mount @ result

    def set_qpos(self, *_args, **_kwargs):
        pytest.fail("Release envelope must never mutate the robot state.")


@pytest.mark.parametrize("arm", ["left_arm", "right_arm"])
@pytest.mark.parametrize("master", [0.0, 0.55, 0.9])
def test_commanded_robotiq_qpos_expands_master_without_changing_other_joints(
    arm: str, master: float
) -> None:
    robot = _Robot(num_envs=2)
    profile = get_gripper_profile("robotiq")
    robot.joint_names = [
        *reversed(profile.simulated_joint_names("right")),
        "left_arm_joint",
        *reversed(profile.simulated_joint_names("left")),
        "right_arm_joint",
    ]
    robot.qpos = (
        torch.arange(2 * len(robot.joint_names), dtype=torch.float32).reshape(2, -1)
        / 100.0
    )
    initial = robot.qpos.clone()
    master_position = torch.tensor([master, 0.25], requires_grad=True)

    target = gripper_geometry._commanded_gripper_qpos(
        SimpleNamespace(robot=robot), arm, profile, master_position
    )

    side = arm.removesuffix("_arm")
    controls = profile.control_joint_names(side)
    signs = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
    for column, name in enumerate(robot.joint_names):
        expected = (
            master_position.detach() * signs[controls.index(name)]
            if name in controls
            else initial[:, column]
        )
        torch.testing.assert_close(target[:, column], expected)
    assert not target.requires_grad
    torch.testing.assert_close(robot.qpos, initial)
    torch.testing.assert_close(master_position.detach(), torch.tensor([master, 0.25]))


@pytest.mark.parametrize("arm", ["left_arm", "right_arm"])
def test_commanded_pgi_qpos_updates_mimic_outside_control_part(arm: str) -> None:
    robot = _Robot(num_envs=2)
    initial = robot.qpos.clone()
    master_position = torch.tensor([0.015, 0.0])
    side = arm.removesuffix("_arm")
    assert robot.profile.mimic_joint_names(side)[
        0
    ] not in robot.profile.control_joint_names(side)

    target = gripper_geometry._commanded_gripper_qpos(
        SimpleNamespace(robot=robot), arm, robot.profile, master_position
    )

    selected = set(robot.profile.simulated_joint_names(side))
    for column, name in enumerate(robot.joint_names):
        expected = master_position if name in selected else initial[:, column]
        torch.testing.assert_close(target[:, column], expected)
    torch.testing.assert_close(robot.qpos, initial)


def test_commanded_gripper_qpos_uses_supplied_reference_and_profile_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _Robot(num_envs=2)
    profile = replace(robot.profile, mimic_multipliers=(-1.0,), mimic_offsets=(0.01,))
    reference = robot.qpos.clone().requires_grad_()
    master_position = torch.tensor([0.02, 0.03])
    monkeypatch.setattr(
        robot, "get_qpos", lambda: pytest.fail("Use supplied reference.")
    )

    target = gripper_geometry._commanded_gripper_qpos(
        SimpleNamespace(robot=robot),
        "left_arm",
        profile,
        master_position,
        reference_qpos=reference,
    )

    master_id = robot.joint_names.index(profile.control_joint_names("left")[0])
    mimic_id = robot.joint_names.index(profile.mimic_joint_names("left")[0])
    torch.testing.assert_close(target[:, master_id], master_position)
    torch.testing.assert_close(target[:, mimic_id], 0.01 - master_position)
    torch.testing.assert_close(reference, robot.qpos)
    target[:, 0] = 7.0
    assert not torch.equal(reference[:, 0], target[:, 0])


@pytest.mark.parametrize("arm", ["left", "coordinated", "unknown"])
def test_commanded_gripper_qpos_rejects_unknown_arm(arm: str) -> None:
    robot = _Robot()
    with pytest.raises(ValueError, match="arm"):
        gripper_geometry._commanded_gripper_qpos(
            SimpleNamespace(robot=robot), arm, robot.profile, torch.tensor([0.01])
        )


@pytest.mark.parametrize("mutation", ["missing_master", "missing_mimic", "duplicate"])
def test_commanded_gripper_qpos_rejects_incomplete_or_ambiguous_joint_names(
    mutation: str,
) -> None:
    robot = _Robot()
    if mutation == "missing_master":
        robot.joint_names[4] = "unexpected"
    elif mutation == "missing_mimic":
        robot.joint_names[1] = "unexpected"
    else:
        robot.joint_names[0] = robot.joint_names[1]
    with pytest.raises(ValueError, match="joint"):
        gripper_geometry._commanded_gripper_qpos(
            SimpleNamespace(robot=robot),
            "left_arm",
            robot.profile,
            torch.tensor([0.01]),
        )


@pytest.mark.parametrize(
    "master",
    [
        torch.tensor(0.01),
        torch.zeros(1, 1),
        torch.zeros(2),
        torch.tensor([float("nan")]),
    ],
)
def test_commanded_gripper_qpos_rejects_invalid_master_shape_or_value(master) -> None:
    robot = _Robot()
    with pytest.raises(ValueError, match="master_position"):
        gripper_geometry._commanded_gripper_qpos(
            SimpleNamespace(robot=robot), "left_arm", robot.profile, master
        )


@pytest.mark.parametrize("master", [[0.01], torch.tensor([1]), torch.tensor([True])])
def test_commanded_gripper_qpos_requires_floating_tensor_master(master) -> None:
    robot = _Robot()
    with pytest.raises(TypeError, match="master_position"):
        gripper_geometry._commanded_gripper_qpos(
            SimpleNamespace(robot=robot), "left_arm", robot.profile, master
        )


@pytest.mark.parametrize(
    "reference",
    [
        torch.zeros(6),
        torch.zeros(1, 5),
        torch.empty(0, 6),
        torch.full((1, 6), float("nan")),
    ],
)
def test_commanded_gripper_qpos_rejects_invalid_reference(reference) -> None:
    robot = _Robot()
    with pytest.raises(ValueError, match="reference_qpos"):
        gripper_geometry._commanded_gripper_qpos(
            SimpleNamespace(robot=robot),
            "left_arm",
            robot.profile,
            torch.tensor([0.01]),
            reference_qpos=reference,
        )


@pytest.mark.parametrize("arm", ["left_arm", "right_arm"])
def test_release_sweep_uses_named_hand_order_and_keeps_other_joints(arm) -> None:
    robot = _Robot()
    initial = robot.qpos.clone()
    points = _release_sweep_points(SimpleNamespace(robot=robot), arm, robot.profile)
    samples, links = robot.fk_calls[0]
    side = arm.removesuffix("_arm")
    hand_names = robot.profile.simulated_joint_names(side)
    assert samples.shape == (5, len(robot.joint_names))
    for column, name in enumerate(robot.joint_names):
        expected = (
            torch.linspace(0.04, 0.0, 5)
            if name in hand_names
            else initial[:, column].expand(5)
        )
        torch.testing.assert_close(samples[:, column], expected)
    assert set(links) == {
        f"{side}_{link}" for link in ("mount", "palm", "finger1", "finger2", "pad")
    }
    assert all(link.startswith(f"{side}_") for link in robot.mesh_calls)
    assert points.shape == (5 * 4 * robot.mesh.shape[0], 3)
    torch.testing.assert_close(robot.qpos, initial)


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("scale", [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)])
def test_release_sweep_converts_root_fk_and_scaled_mesh_to_tcp_frame(
    num_envs, scale
) -> None:
    robot = _Robot(scale=scale, num_envs=num_envs)
    tcp = torch.tensor(robot.profile.tcp_transform)
    tcp[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    robot.profile = replace(
        robot.profile, tcp_transform=tuple(tuple(row) for row in tcp.tolist())
    )
    points = _release_sweep_points(
        SimpleNamespace(robot=robot), "left_arm", robot.profile
    )
    expected_points = []
    mesh = robot.mesh * torch.tensor(scale)
    for _ in range(num_envs):
        for opening in torch.linspace(0.04, 0.0, 5):
            for offset in (
                [0, 0, 0],
                [opening, 0, 0],
                [-opening, 0, 0],
                [opening, 0, 0.04],
            ):
                expected_points.append(
                    (mesh + torch.tensor(offset) - tcp[:3, 3]) @ tcp[:3, :3]
                )
    expected = torch.cat(expected_points)
    # Compare as sets: implementation may group points by link rather than sample.
    assert points.shape == expected.shape
    distances = torch.cdist(
        points, expected, compute_mode="donot_use_mm_for_euclid_dist"
    )
    assert distances.min(dim=1).values.max() < 2.0e-6
    assert distances.min(dim=0).values.max() < 2.0e-6


def test_release_sweep_rejects_missing_hand_joint() -> None:
    robot = _Robot()
    robot.joint_names[4] = "unexpected_joint"
    with pytest.raises(ValueError, match="joint"):
        _release_sweep_points(SimpleNamespace(robot=robot), "left_arm", robot.profile)


@pytest.mark.parametrize("arm", ["left_arm", "right_arm"])
def test_release_sweep_preserves_robotiq_control_column_order(monkeypatch, arm) -> None:
    robot = _Robot()
    robot.profile = get_gripper_profile("robotiq")
    robot.joint_names = [
        *reversed(robot.profile.simulated_joint_names("right")),
        "left_arm_joint",
        *reversed(robot.profile.simulated_joint_names("left")),
        "right_arm_joint",
    ]
    robot.qpos = torch.linspace(0.1, 0.2, len(robot.joint_names)).unsqueeze(0)
    initial = robot.qpos.clone()

    def compute_fk(
        qpos, name=None, link_names=None, qpos_joint_names=None, to_matrix=False
    ):
        if name is not None:
            return (
                robot.root_pose
                @ robot._mount_pose()
                @ torch.tensor(robot.profile.tcp_transform)
            )
        assert tuple(qpos_joint_names) == tuple(robot.joint_names)
        robot.fk_calls.append((qpos.clone(), tuple(link_names)))
        return robot._mount_pose().repeat(qpos.shape[0], len(link_names), 1, 1)

    monkeypatch.setattr(robot, "compute_fk", compute_fk)
    original_chain = robot.get_parent_joint_chain

    def parent_chain(link_name):
        chain = original_chain(link_name)
        if link_name.endswith(("root", "mount")):
            return chain
        side = link_name.split("_", 1)[0]
        return (
            tuple(
                SimpleNamespace(
                    name=name,
                    parent_link_name=f"{side}_mount",
                    child_link_name=link_name,
                )
                for name in robot.profile.simulated_joint_names(side)
            )
            + chain[-1:]
        )

    monkeypatch.setattr(robot, "get_parent_joint_chain", parent_chain)
    _release_sweep_points(SimpleNamespace(robot=robot), arm, robot.profile)
    samples, _ = robot.fk_calls[0]
    control_names = robot.profile.control_joint_names(arm.removesuffix("_arm"))
    for column, name in enumerate(robot.joint_names):
        if name in control_names:
            profile_column = control_names.index(name)
            expected = torch.linspace(
                robot.profile.close_positions[profile_column],
                robot.profile.open_positions[profile_column],
                5,
            )
        else:
            expected = initial[:, column].expand(5)
        torch.testing.assert_close(samples[:, column], expected)


def test_release_sweep_rejects_empty_gripper_geometry(monkeypatch) -> None:
    robot = _Robot()
    monkeypatch.setattr(
        robot,
        "get_link_vert_face",
        lambda _: (torch.empty(0, 3), torch.empty(0, 3, dtype=torch.int64)),
    )
    with pytest.raises(ValueError, match="geometry"):
        _release_sweep_points(SimpleNamespace(robot=robot), "left_arm", robot.profile)


@pytest.mark.parametrize("opening", [0.0, 0.012])
def test_gripper_points_stationary_uses_observed_qpos_without_closing(
    monkeypatch, opening
) -> None:
    robot = _Robot()
    current = robot.qpos.clone()
    current[:, 1] = current[:, 4] = opening
    monkeypatch.setattr(
        robot,
        "get_qpos",
        lambda **_kwargs: pytest.fail("Use supplied state, not a new qpos read."),
    )
    points = _gripper_points(SimpleNamespace(robot=robot), "left_arm", qpos=current)
    sampled, _ = robot.fk_calls[0]
    torch.testing.assert_close(sampled, current)
    assert points.shape == (1, 1, 12, 3)
    closed = current.clone()
    closed[:, 1] = closed[:, 4] = 0.04
    closed_points = _gripper_points(
        SimpleNamespace(robot=robot), "left_arm", qpos=closed
    )
    assert not torch.allclose(points, closed_points)
    assert points[0, 0, :, 0].max() < 0.03


def test_gripper_points_samples_actual_range_and_preserves_environment_rows() -> None:
    robot = _Robot(num_envs=2)
    current = robot.qpos.clone()
    current[:, 1] = current[:, 4] = torch.tensor([0.005, 0.025])
    target = current.clone()
    target[:, 1] = target[:, 4] = torch.tensor([0.01, 0.02])
    points = _gripper_points(
        SimpleNamespace(robot=robot),
        "left_arm",
        qpos=current,
        target_qpos=target,
        sample_count=3,
    )
    samples, _ = robot.fk_calls[0]
    samples = samples.reshape(2, 3, -1)
    torch.testing.assert_close(samples[:, 0], current)
    torch.testing.assert_close(samples[:, -1], target)
    torch.testing.assert_close(samples[:, 1], (current + target) / 2)
    assert points.shape == (2, 3, 12, 3)
    assert points[0, :, :, 0].max() < 0.021
    assert points[1, :, :, 0].max() > 0.034
    torch.testing.assert_close(robot.qpos, _Robot(num_envs=2).qpos)


@pytest.mark.parametrize("column", [0, 2, 3, 5])
def test_gripper_points_rejects_non_hand_target_motion(column) -> None:
    robot = _Robot()
    target = robot.qpos.clone()
    target[:, column] += 0.1
    with pytest.raises(ValueError, match="hand joints"):
        _gripper_points(
            SimpleNamespace(robot=robot),
            "left_arm",
            qpos=robot.qpos,
            target_qpos=target,
        )
    assert robot.fk_calls == []


@pytest.mark.parametrize("sample_count", [0, 1, -2, 2.5, True])
def test_gripper_points_rejects_invalid_sweep_sample_count(sample_count) -> None:
    robot = _Robot()
    with pytest.raises(ValueError, match="sample_count"):
        _gripper_points(
            SimpleNamespace(robot=robot),
            "left_arm",
            qpos=robot.qpos,
            target_qpos=robot.qpos,
            sample_count=sample_count,
        )


@pytest.mark.parametrize(
    "target", [torch.zeros(1, 5), torch.full((1, 6), float("nan"))]
)
def test_gripper_points_rejects_invalid_target_tensor(target) -> None:
    robot = _Robot()
    with pytest.raises(ValueError, match="target_qpos"):
        _gripper_points(
            SimpleNamespace(robot=robot),
            "left_arm",
            qpos=robot.qpos,
            target_qpos=target,
        )
