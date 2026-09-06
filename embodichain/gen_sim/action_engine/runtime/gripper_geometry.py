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

"""Read-only gripper release geometry for GenSim grasp clearance checks."""

from __future__ import annotations

import math
from typing import Any

import torch

from ..gripper_profiles import GripperProfile
from .articulation import _scaled_link_geometry
from .robot_parts import arm_control_part

__all__: list[str] = []


def _commanded_gripper_qpos(
    env: Any,
    arm: str,
    profile: GripperProfile,
    master_position: torch.Tensor,
    *,
    reference_qpos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand one master command without rewriting the observed hand state.

    The result is a detached full-robot target: only the selected hand's master
    and mimic joints change. Limits remain the caller's responsibility; this
    function neither clamps the master command nor writes to the simulator.
    """
    if arm not in {"left_arm", "right_arm"}:
        raise ValueError("Gripper command requires a left_arm or right_arm.")
    if not isinstance(profile, GripperProfile):
        raise TypeError("profile must be a GripperProfile.")
    side = "left" if arm == "left_arm" else "right"
    joint_names = tuple(env.robot.joint_names)
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Gripper command requires unique robot joint names.")
    controls = profile.control_joint_names(side)
    mimics = profile.mimic_joint_names(side)
    master = controls[0]
    if (
        len(set(controls)) != len(controls)
        or len(set(mimics)) != len(mimics)
        or master in mimics
        or not set(controls[1:]).issubset(mimics)
        or not set((*controls, *mimics)).issubset(joint_names)
    ):
        raise ValueError(
            "Gripper command requires one master and complete mimic joint names."
        )
    reference = env.robot.get_qpos() if reference_qpos is None else reference_qpos
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise TypeError("reference_qpos must be a floating-point tensor.")
    if (
        reference.ndim != 2
        or reference.shape[0] == 0
        or reference.shape[1] != len(joint_names)
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("reference_qpos must have finite full-robot shape (B, D).")
    if (
        not isinstance(master_position, torch.Tensor)
        or not master_position.is_floating_point()
    ):
        raise TypeError("master_position must be a floating-point tensor.")
    if (
        master_position.shape != (reference.shape[0],)
        or not torch.isfinite(master_position).all()
    ):
        raise ValueError("master_position must have finite shape (B,).")
    if (
        master_position.dtype != reference.dtype
        or master_position.device != reference.device
    ):
        raise ValueError("master_position must share reference_qpos dtype and device.")

    result = reference.detach().clone()
    master_position = master_position.detach()
    result[:, joint_names.index(master)] = master_position
    for name, multiplier, offset in zip(
        mimics, profile.mimic_multipliers, profile.mimic_offsets, strict=True
    ):
        if not math.isfinite(multiplier) or not math.isfinite(offset):
            raise ValueError("Mimic joint coefficients must be finite.")
        result[:, joint_names.index(name)] = master_position * multiplier + offset
    return result


def _gripper_points(
    env: Any,
    arm: str,
    *,
    qpos: torch.Tensor,
    target_qpos: torch.Tensor | None = None,
    sample_count: int = 5,
) -> torch.Tensor:
    """Return actual hand geometry as ``(B, S, V, 3)`` TCP-frame vertices.

    Without a target, only the supplied full-robot configuration is evaluated
    (``S=1``). Otherwise, the hand is sampled between the supplied configurations;
    changing joints outside that hand's mount subtree is rejected. No live state
    is written, and no environment rows are combined. These discrete link-mesh
    samples do not constitute a continuous backend collision-shape guarantee.
    """
    robot = env.robot
    control_part = arm_control_part(env, arm)
    solver = robot.get_solver(name=control_part)
    mount = getattr(getattr(solver, "cfg", None), "end_link_name", None)
    if mount not in robot.link_names:
        raise ValueError("Release geometry requires a configured gripper mount link.")
    chains = {link: robot.get_parent_joint_chain(link) for link in robot.link_names}
    links = [
        link
        for link in robot.link_names
        if link == mount
        or any(joint.parent_link_name == mount for joint in chains[link])
    ]
    joint_names = tuple(robot.joint_names)
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Gripper geometry requires unique full-robot joint names.")
    if (
        not isinstance(qpos, torch.Tensor)
        or not qpos.is_floating_point()
        or qpos.ndim != 2
        or qpos.shape[0] == 0
        or qpos.shape[1] != len(joint_names)
        or not torch.isfinite(qpos).all()
    ):
        raise ValueError("Release geometry requires finite full-robot qpos (B, D).")
    qpos = qpos.detach().clone()
    batch_size = qpos.shape[0]
    if target_qpos is None:
        sample_count = 1
        samples = qpos[:, None]
    else:
        if type(sample_count) is not int or sample_count < 2:
            raise ValueError("Gripper sweep sample_count must be an integer >= 2.")
        if (
            not isinstance(target_qpos, torch.Tensor)
            or not target_qpos.is_floating_point()
            or target_qpos.shape != qpos.shape
            or not torch.isfinite(target_qpos).all()
        ):
            raise ValueError("Gripper target_qpos must match finite qpos shape (B, D).")
        target_qpos = target_qpos.detach().to(qpos)
        hand_names = set()
        for link in links:
            for joint in chains[link]:
                if joint.child_link_name == mount:
                    break
                hand_names.add(joint.name)
        fixed_columns = [
            index for index, name in enumerate(joint_names) if name not in hand_names
        ]
        if not torch.equal(qpos[:, fixed_columns], target_qpos[:, fixed_columns]):
            raise ValueError("Gripper sweep may change only the selected hand joints.")
        fraction = torch.linspace(
            0.0, 1.0, sample_count, device=qpos.device, dtype=qpos.dtype
        )
        samples = torch.lerp(
            qpos[:, None], target_qpos[:, None], fraction[None, :, None]
        )

    local_poses = robot.compute_fk(
        qpos=samples.reshape(-1, len(joint_names)),
        link_names=links,
        qpos_joint_names=joint_names,
    )
    root_pose = robot.get_local_pose(to_matrix=True).to(qpos)
    tcp_pose = robot.compute_fk(
        qpos=qpos[:, robot.get_joint_ids(name=control_part)],
        name=control_part,
        to_matrix=True,
    ).to(qpos)
    if (
        local_poses.shape != (batch_size * sample_count, len(links), 4, 4)
        or root_pose.shape != (batch_size, 4, 4)
        or tcp_pose.shape != (batch_size, 4, 4)
        or not all(
            torch.isfinite(pose).all() for pose in (local_poses, root_pose, tcp_pose)
        )
    ):
        raise ValueError("Release geometry requires finite, batch-aligned FK poses.")
    local_poses = local_poses.to(qpos).reshape(
        batch_size, sample_count, len(links), 4, 4
    )
    tcp_from_root = torch.linalg.inv(tcp_pose) @ root_pose
    link_poses = tcp_from_root[:, None, None] @ local_poses
    points = []
    for index, link in enumerate(links):
        vertices, _ = _scaled_link_geometry(robot, link)
        vertices = vertices.to(qpos)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not torch.isfinite(vertices).all()
        ):
            raise ValueError(
                f"Release geometry for link {link!r} must be finite (V, 3)."
            )
        if not vertices.shape[0]:
            continue
        poses = link_poses[:, :, index]
        points.append(
            vertices @ poses[..., :3, :3].transpose(-1, -2) + poses[..., None, :3, 3]
        )
    if not points:
        raise ValueError("Release geometry contains no gripper link vertices.")
    return torch.cat(points, dim=2)


def _release_sweep_points(env: Any, arm: str, profile: GripperProfile) -> torch.Tensor:
    """Retain the legacy full-close-to-open union until callers are migrated."""
    arm_control_part(env, arm)
    side = "left" if arm == "left_arm" else "right"
    joint_names = tuple(env.robot.joint_names)
    if not set(profile.simulated_joint_names(side)).issubset(joint_names):
        raise ValueError("Release geometry requires complete hand joint names.")
    current = env.robot.get_qpos().detach().clone()
    target = current.clone()
    controls = profile.control_joint_names(side)
    for column, name in enumerate(controls):
        current[:, joint_names.index(name)] = profile.close_positions[column]
        target[:, joint_names.index(name)] = profile.open_positions[column]
    for name, multiplier, offset in zip(
        profile.mimic_joint_names(side),
        profile.mimic_multipliers,
        profile.mimic_offsets,
        strict=True,
    ):
        if name not in controls:
            current[:, joint_names.index(name)] = (
                profile.close_positions[0] * multiplier + offset
            )
            target[:, joint_names.index(name)] = (
                profile.open_positions[0] * multiplier + offset
            )
    return _gripper_points(env, arm, qpos=current, target_qpos=target).reshape(-1, 3)
