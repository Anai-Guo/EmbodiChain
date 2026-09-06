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

"""Discrete articulation clearance for gripper approach and withdrawal."""

from __future__ import annotations

import math
from typing import Any

import torch

from embodichain.toolkits.graspkit.pg_grasp import ConvexCollisionChecker

from .articulation import _named_link_geometry, _scaled_link_geometry
from .gripper_geometry import _gripper_points

__all__: list[str] = []

_MIN_WITHDRAWAL_PATH_SAMPLES = 5
_CONTACT_CLEARANCE = 0.003


class _InteractionClearance:
    """Cache target-link checkers for one scene, never their live poses.

    Withdrawal samples a constant-attitude straight path with spacing no greater
    than the contact-clearance margin. Precontact audits supplied TCP samples
    against links outside the intended interaction joint's subtree.
    Withdrawal includes a horizontal support plane only when explicitly supplied.
    Other world objects, the rest of the robot, and continuous collision between
    samples are not covered. Recreate it if scene geometry changes.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._checkers: dict[tuple[str, str], ConvexCollisionChecker] = {}
        self._closure_checkers: dict[
            tuple[str, str, str], ConvexCollisionChecker | None
        ] = {}

    def withdrawal(
        self,
        arm: str,
        target_uid: str,
        *,
        direction: torch.Tensor,
        distance: float,
        hand_target_qpos: torch.Tensor | None = None,
        support_surface_z: torch.Tensor | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return signed path clearance, optionally recording its component bounds.

        Diagnostics retain detached row-local target and support clearances.
        Initial support clearance includes the observed-to-commanded hand shapes
        at the current TCP, before translation along the withdrawal path.
        Terminal clearance includes those same hand shapes at the path endpoint.
        """
        if arm not in {"left_arm", "right_arm"}:
            raise ValueError("Withdrawal requires a specific semantic arm.")
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (float, int))
            or not math.isfinite(distance)
            or distance < 0.0
        ):
            raise ValueError("Withdrawal distance must be finite and non-negative.")
        qpos = self.env.robot.get_qpos()
        batch_size = qpos.shape[0]
        surface = None
        if support_surface_z is not None:
            surface = torch.as_tensor(support_surface_z, device=qpos.device)
            if (
                surface.shape not in {(), (1,), (batch_size,)}
                or surface.dtype == torch.bool
                or surface.is_complex()
                or not torch.isfinite(surface).all()
            ):
                raise ValueError(
                    "Withdrawal support_surface_z must be finite real scalar, (1,), or (B,)."
                )
            surface = surface.to(qpos).reshape(-1).expand(batch_size)
        if (
            not isinstance(direction, torch.Tensor)
            or not direction.is_floating_point()
            or direction.shape != (batch_size, 3)
            or not torch.isfinite(direction).all()
        ):
            raise ValueError("Withdrawal direction must be finite floating (B, 3).")
        direction = direction.to(qpos)
        lengths = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(lengths).all() or torch.any(lengths <= 1.0e-6):
            raise ValueError("Withdrawal direction must have finite non-zero rows.")
        direction = direction / lengths
        target = self.env.sim.get_articulation(target_uid)
        if target is None or not target.link_names:
            raise ValueError("Withdrawal target articulation geometry is unavailable.")
        hand = _gripper_points(self.env, arm, qpos=qpos, target_qpos=hand_target_qpos)
        if (
            hand.ndim != 4
            or hand.shape[0] != batch_size
            or hand.shape[1] == 0
            or hand.shape[2] == 0
            or hand.shape[3] != 3
            or not torch.isfinite(hand).all()
        ):
            raise ValueError("Withdrawal hand geometry must be finite (B, S, V, 3).")
        tcp = self.env.get_current_xpos_agent()[0 if arm == "left_arm" else 1]
        if (
            tcp is None
            or tcp.shape != (batch_size, 4, 4)
            or not torch.isfinite(tcp).all()
        ):
            raise ValueError("Withdrawal requires finite live TCP poses (B, 4, 4).")
        tcp = tcp.to(hand)
        initial_points = hand.flatten(1, 2) @ tcp[:, :3, :3].transpose(-1, -2)
        initial_points += tcp[:, None, :3, 3]
        path_sample_count = max(
            _MIN_WITHDRAWAL_PATH_SAMPLES,
            math.ceil(distance / _CONTACT_CLEARANCE) + 1,
        )
        progress = torch.linspace(
            0.0,
            distance,
            path_sample_count,
            device=hand.device,
            dtype=hand.dtype,
        )
        path_points = (
            initial_points[:, None]
            + progress[None, :, None, None] * direction[:, None, None]
        )
        clearance = self._query_links(
            target_uid,
            target,
            list(target.link_names),
            path_points.reshape(batch_size, -1, 3),
        )
        target_clearance = clearance
        support_clearance = None
        if surface is not None:
            support_clearance = (path_points[..., 2] - surface[:, None, None]).amin(
                dim=(1, 2)
            )
            clearance = torch.minimum(clearance, support_clearance.to(clearance))
        if diagnostics is not None:
            terminal_points = path_points[:, -1]
            terminal_clearance = self._query_links(
                target_uid, target, list(target.link_names), terminal_points
            )
            if surface is not None:
                terminal_support = (terminal_points[..., 2] - surface[:, None]).amin(
                    dim=1
                )
                terminal_clearance = torch.minimum(
                    terminal_clearance, terminal_support.to(terminal_clearance)
                )
            diagnostics.update(
                {
                    "terminal_clearance": terminal_clearance.detach().clone(),
                    "target_clearance": target_clearance.detach().clone(),
                    "support_clearance": (
                        None
                        if support_clearance is None
                        else support_clearance.detach().clone()
                    ),
                    "initial_support_clearance": (
                        None
                        if surface is None
                        else (initial_points[..., 2] - surface[:, None])
                        .amin(dim=1)
                        .detach()
                        .clone()
                    ),
                }
            )
        return clearance

    def precontact(
        self,
        target_uid: str,
        target_joint: str,
        tcp_poses: torch.Tensor,
        hand_points: torch.Tensor,
    ) -> torch.Tensor:
        """Audit supplied TCP poses against the non-target articulation links.

        The target joint's complete child subtree is omitted because it owns
        intended grasp contact. Root links and other joint branches remain in
        scope. Only supplied samples are checked; paths are not interpolated.
        """
        if (
            not isinstance(tcp_poses, torch.Tensor)
            or not tcp_poses.is_floating_point()
            or tcp_poses.ndim != 4
            or tcp_poses.shape[0] == 0
            or tcp_poses.shape[1] == 0
            or tcp_poses.shape[2:] != (4, 4)
            or not torch.isfinite(tcp_poses).all()
        ):
            raise ValueError(
                "Precontact TCP poses must be finite floating (B, T, 4, 4)."
            )
        batch_size = tcp_poses.shape[0]
        if (
            not isinstance(hand_points, torch.Tensor)
            or not hand_points.is_floating_point()
            or hand_points.ndim != 3
            or hand_points.shape[0] != batch_size
            or hand_points.shape[1] == 0
            or hand_points.shape[2] != 3
            or not torch.isfinite(hand_points).all()
        ):
            raise ValueError(
                "Precontact hand points must be finite floating (B, N, 3)."
            )
        if type(target_joint) is not str or not target_joint:
            raise ValueError("Precontact requires a non-empty target joint name.")
        target = self.env.sim.get_articulation(target_uid)
        if target is None or not target.link_names:
            raise ValueError("Precontact target articulation geometry is unavailable.")
        links = []
        found_target_joint = False
        for link in target.link_names:
            if any(
                joint.name == target_joint
                for joint in target.get_parent_joint_chain(link)
            ):
                found_target_joint = True
            else:
                links.append(link)
        if not found_target_joint:
            raise ValueError(f"Precontact target joint {target_joint!r} was not found.")
        if not links:
            raise ValueError(
                "Precontact has no non-target articulation geometry to audit."
            )
        points = hand_points.to(tcp_poses)
        world_points = (
            points[:, None] @ tcp_poses[..., :3, :3].transpose(-1, -2)
            + tcp_poses[:, :, None, :3, 3]
        )
        return self._query_links(
            target_uid, target, links, world_points.reshape(batch_size, -1, 3)
        )

    def grasp_closure(
        self,
        target_uid: str,
        target_link: str,
        handle_mesh: str,
        tcp_pose: torch.Tensor,
        hand_points: torch.Tensor,
        *,
        support_surface_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Audit a fixed-TCP closing sweep, permitting only exact handle contact."""
        target = self.env.sim.get_articulation(target_uid)
        if target is None or target_link not in target.link_names or not handle_mesh:
            raise ValueError(
                "Grasp closure requires an exact moving link and handle mesh."
            )
        if (
            hand_points.ndim != 4
            or hand_points.shape[0] == 0
            or hand_points.shape[1] == 0
            or hand_points.shape[2] == 0
            or hand_points.shape[3] != 3
            or not torch.isfinite(hand_points).all()
            or tcp_pose.shape != (hand_points.shape[0], 4, 4)
            or not torch.isfinite(tcp_pose).all()
        ):
            raise ValueError(
                "Grasp closure requires finite, batch-aligned TCP and hand geometry."
            )
        key = (target_uid, target_link, handle_mesh)
        if key not in self._closure_checkers:
            if (
                _named_link_geometry(target, target_link, mesh_names=(handle_mesh,))
                is None
            ):
                raise ValueError("Grasp closure handle mesh was not found.")
            geometry = _named_link_geometry(
                target, target_link, excluded_mesh_names=(handle_mesh,)
            )
            self._closure_checkers[key] = (
                None if geometry is None else self._mesh_checker(*geometry, target_link)
            )
        partial = self._closure_checkers[key]
        links = [
            link
            for link in target.link_names
            if link != target_link or partial is not None
        ]
        points = hand_points.to(tcp_pose).flatten(1, 2)
        world = points @ tcp_pose[:, :3, :3].transpose(1, 2) + tcp_pose[:, None, :3, 3]
        clearance = self._query_links(
            target_uid,
            target,
            links,
            world,
            checker_overrides={} if partial is None else {target_link: partial},
        )
        if support_surface_z is not None:
            surface = torch.as_tensor(support_surface_z).to(world).reshape(-1)
            if (
                surface.numel() not in {1, world.shape[0]}
                or not torch.isfinite(surface).all()
            ):
                raise ValueError("Grasp support height must be finite and row-local.")
            clearance = torch.minimum(
                clearance, (world[..., 2] - surface[:, None]).amin(1)
            )
        return clearance

    def _query_links(
        self,
        target_uid: str,
        target: Any,
        links: list[str],
        world_points: torch.Tensor,
        *,
        checker_overrides: dict[str, ConvexCollisionChecker] | None = None,
    ) -> torch.Tensor:
        """Query cached link meshes in their current live frames, row by row."""
        batch_size = world_points.shape[0]
        clearance = world_points.new_full((batch_size,), float("inf"))
        for link in links:
            checker = (checker_overrides or {}).get(link)
            if checker is None:
                checker = self._link_checker(target_uid, target, link)
            pose = target.get_link_pose(link, to_matrix=True).to(world_points)
            if pose.shape != (batch_size, 4, 4) or not torch.isfinite(pose).all():
                raise ValueError(
                    f"Interaction link {link!r} requires finite (B,4,4) poses."
                )
            local_points = (world_points - pose[:, None, :3, 3]) @ pose[:, :3, :3]
            _, distances = checker.query_batch_points(
                local_points, collision_threshold=_CONTACT_CLEARANCE, is_visual=False
            )
            if (
                not isinstance(distances, torch.Tensor)
                or distances.shape != local_points.shape[:2]
                or not torch.isfinite(distances).all()
            ):
                raise ValueError(
                    "Interaction checker returned invalid signed distances."
                )
            clearance = torch.minimum(clearance, distances.to(clearance).amin(dim=1))
        return clearance

    def _link_checker(
        self, target_uid: str, target: Any, link: str
    ) -> ConvexCollisionChecker:
        key = (target_uid, link)
        if key not in self._checkers:
            vertices, faces = _scaled_link_geometry(target, link)
            self._checkers[key] = self._mesh_checker(vertices, faces, link)
        return self._checkers[key]

    @staticmethod
    def _mesh_checker(
        vertices: torch.Tensor, faces: torch.Tensor, link: str
    ) -> ConvexCollisionChecker:
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] == 0
            or not torch.isfinite(vertices).all()
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.shape[0] == 0
            or torch.any(faces < 0)
            or torch.any(faces >= vertices.shape[0])
        ):
            raise ValueError(f"Withdrawal target link {link!r} has invalid geometry.")
        return ConvexCollisionChecker(vertices, faces)
