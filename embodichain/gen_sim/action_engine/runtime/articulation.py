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

"""Shared live-articulation resolution for GenSim grounding and verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

__all__: list[str] = []


def _joint_type_name(info: Any) -> str:
    value = getattr(info, "joint_type", None)
    return str(getattr(value, "name", value)).rsplit(".", maxsplit=1)[-1].lower()


def _scene_entity(sim: Any, uid: str) -> Any | None:
    """Resolve a rigid object or articulation without wrong-registry warnings."""
    list_articulations = getattr(sim, "get_articulation_uid_list", None)
    get_articulation = getattr(sim, "get_articulation", None)
    if callable(list_articulations) and callable(get_articulation):
        if uid in {str(value) for value in list_articulations()}:
            return get_articulation(uid)
    list_rigid = getattr(sim, "get_rigid_object_uid_list", None)
    get_rigid = getattr(sim, "get_rigid_object", None)
    if callable(list_rigid) and callable(get_rigid):
        if uid in {str(value) for value in list_rigid()}:
            return get_rigid(uid)
    if not callable(list_rigid) and callable(get_rigid):
        entity = get_rigid(uid)
        if entity is not None:
            return entity
    if not callable(list_articulations) and callable(get_articulation):
        return get_articulation(uid)
    return None


def _active_joint_candidates(
    articulation: Any,
    *,
    joint_types: Iterable[str],
) -> list[tuple[int, str, Any]]:
    backend_entities = getattr(
        articulation,
        "_entities",
        getattr(articulation, "entities", ()),
    )
    if not backend_entities:
        raise ValueError("Articulation backend does not expose joint metadata.")
    backend = backend_entities[0]
    accepted = frozenset(str(value).lower() for value in joint_types)
    result = []
    for raw_joint_id in getattr(
        articulation,
        "active_joint_ids",
        range(len(articulation.joint_names)),
    ):
        joint_id = int(raw_joint_id)
        joint_name = str(articulation.joint_names[joint_id])
        info = backend.get_joint_info(joint_name)
        if _joint_type_name(info) in accepted:
            result.append((joint_id, joint_name, info))
    return result


def _select_joint_candidate(
    candidates: list[tuple[int, str, Any]],
    *,
    preferred_name_tokens: Iterable[str] = (),
    context: str,
) -> tuple[int, str, Any]:
    if len(candidates) == 1:
        return candidates[0]
    tokens = tuple(str(token).casefold() for token in preferred_name_tokens)
    preferred = [
        candidate
        for candidate in candidates
        if any(token in candidate[1].casefold() for token in tokens)
    ]
    if len(preferred) == 1:
        return preferred[0]
    names = [name for _, name, _ in candidates]
    raise ValueError(f"{context} requires one unambiguous joint; found {names}.")


def _select_interaction_joint_candidate(
    articulation: Any,
    candidates: list[tuple[int, str, Any]],
    *,
    agent_config: Mapping[str, Any],
    articulation_uid: str,
    interaction: str,
    preferred_name_tokens: Iterable[str] = (),
    context: str,
) -> tuple[int, str, Any, str | None]:
    """Select one joint and optional target link from generated scene metadata."""
    configured = _configured_interaction_target(
        agent_config,
        articulation_uid=articulation_uid,
        interaction=interaction,
    )
    if configured is None:
        joint_id, joint_name, joint_info = _select_joint_candidate(
            candidates,
            preferred_name_tokens=preferred_name_tokens,
            context=context,
        )
        return joint_id, joint_name, joint_info, None

    configured_joint, configured_link = configured
    matches = [
        candidate for candidate in candidates if candidate[1] == configured_joint
    ]
    if len(matches) != 1:
        available = [name for _, name, _ in candidates]
        raise ValueError(
            f"{context} configured joint {configured_joint!r} is not one of "
            f"the compatible live joints {available}."
        )
    if configured_link not in getattr(articulation, "link_names", ()):
        raise ValueError(
            f"{context} configured link {configured_link!r} is not a live "
            f"articulation link."
        )
    joint_id, joint_name, joint_info = matches[0]
    return joint_id, joint_name, joint_info, configured_link


def _configured_interaction_target(
    agent_config: Mapping[str, Any],
    *,
    articulation_uid: str,
    interaction: str,
) -> tuple[str, str] | None:
    """Read one validated generated joint/link target from an agent snapshot."""
    all_targets = agent_config.get("articulation_interaction_links", {})
    if not isinstance(all_targets, Mapping):
        raise ValueError("articulation_interaction_links must be a mapping.")
    per_articulation = all_targets.get(articulation_uid, {})
    if not isinstance(per_articulation, Mapping):
        raise ValueError(
            "articulation_interaction_links entries must map interactions."
        )
    target = per_articulation.get(interaction)
    if target is None:
        return None
    if (
        not isinstance(target, Mapping)
        or not {
            "joint_name",
            "link_name",
        }.issubset(target)
        or not set(target).issubset({"joint_name", "link_name", "mesh_name"})
    ):
        raise ValueError(
            "articulation interaction targets require joint_name and link_name "
            "with an optional mesh_name."
        )
    joint_name = target.get("joint_name")
    link_name = target.get("link_name")
    if (
        not isinstance(joint_name, str)
        or not joint_name.strip()
        or not isinstance(link_name, str)
        or not link_name.strip()
    ):
        raise ValueError(
            "articulation interaction joint_name and link_name must be non-empty."
        )
    return joint_name.strip(), link_name.strip()


def _configured_interaction_mesh_name(
    agent_config: Mapping[str, Any],
    *,
    articulation_uid: str,
    interaction: str,
) -> str | None:
    """Return the optional exact USD mesh name for one configured interaction."""
    all_targets = agent_config.get("articulation_interaction_links", {})
    if not isinstance(all_targets, Mapping):
        raise ValueError("articulation_interaction_links must be a mapping.")
    per_articulation = all_targets.get(articulation_uid, {})
    if not isinstance(per_articulation, Mapping):
        raise ValueError(
            "articulation_interaction_links entries must map interactions."
        )
    target = per_articulation.get(interaction)
    if target is None:
        return None
    if not isinstance(target, Mapping):
        raise ValueError("articulation interaction target must be a mapping.")
    mesh_name = target.get("mesh_name")
    if mesh_name is None:
        return None
    if not isinstance(mesh_name, str) or not mesh_name.strip():
        raise ValueError("articulation interaction mesh_name must be non-empty.")
    return mesh_name.strip()


def _closed_open_endpoints(limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the zero-nearest closed endpoint and the opposite open endpoint."""
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError("Articulation joint limits must have shape (B, 2).")
    if not torch.isfinite(limits).all() or torch.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Articulation joint limits must be finite and ordered.")
    lower = limits[:, 0]
    upper = limits[:, 1]
    lower_is_closed = torch.abs(lower) <= torch.abs(upper)
    closed = torch.where(lower_is_closed, lower, upper)
    opened = torch.where(lower_is_closed, upper, lower)
    return closed, opened


def _effective_joint_limits(
    articulation: Any,
    joint_id: int,
    joint_info: Any,
) -> torch.Tensor:
    """Return runtime-space limits for one generated articulation joint.

    Generated USD revolute limits are authored in degrees while AtomicAction
    contracts use radians. DexSim's prismatic limit query retains authored
    distances, but physics enforces those distances after the articulation's
    uniform body scale, so runtime targets must use the scaled endpoints.
    """
    limits = torch.as_tensor(
        articulation.get_qpos_limits(joint_ids=[int(joint_id)]),
        dtype=torch.float32,
        device=getattr(articulation, "device", None),
    )[:, 0].clone()
    joint_type = _joint_type_name(joint_info)
    if joint_type == "revolute" and _is_usd_articulation(articulation):
        authored = _usd_revolute_limits(articulation, joint_info)
        if authored is not None:
            return limits.new_tensor(authored).repeat(limits.shape[0], 1)
    if joint_type != "prismatic":
        return limits

    configured_limits = getattr(getattr(articulation, "cfg", None), "qpos_limits", None)
    if configured_limits is not None:
        return limits

    scale = torch.as_tensor(
        getattr(getattr(articulation, "cfg", None), "body_scale", (1.0, 1.0, 1.0)),
        dtype=limits.dtype,
        device=limits.device,
    ).reshape(-1)
    if (
        scale.shape != (3,)
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0.0)
    ):
        raise ValueError(
            "Prismatic articulation body_scale must be finite and positive."
        )
    if not torch.allclose(scale, scale[:1].expand_as(scale), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError(
            "Generated prismatic articulations require a uniform body_scale so "
            "joint distances have one unambiguous runtime scale."
        )
    return limits * scale[0]


def _effective_joint_position(
    articulation: Any,
    joint_id: int,
    joint_info: Any,
) -> torch.Tensor:
    """Return one live joint position in AtomicAction SI/radian units."""
    position = torch.as_tensor(
        articulation.get_qpos(),
        dtype=torch.float32,
        device=getattr(articulation, "device", None),
    )[:, int(joint_id)].clone()
    if _joint_type_name(joint_info) == "revolute" and _is_usd_articulation(
        articulation
    ):
        offsets = getattr(articulation, "_gen_sim_revolute_qpos_offsets", {})
        offset = offsets.get(int(joint_id)) if isinstance(offsets, dict) else None
        if offset is not None:
            offset = torch.as_tensor(
                offset,
                dtype=position.dtype,
                device=position.device,
            ).reshape_as(position)
            position = position - offset
        position = torch.atan2(torch.sin(position), torch.cos(position))
    return position


def _is_usd_articulation(articulation: Any) -> bool:
    path = Path(str(getattr(getattr(articulation, "cfg", None), "fpath", "")))
    return path.suffix.lower() in {".usd", ".usda", ".usdc"}


def _scaled_link_geometry(
    articulation: Any,
    link_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return link-local mesh geometry in the runtime articulation scale."""
    vertices, triangles = articulation.get_link_vert_face(link_name)
    device = getattr(articulation, "device", None)
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    triangles = torch.as_tensor(triangles, dtype=torch.int64, device=device)
    if not _is_usd_articulation(articulation):
        return vertices, triangles
    scale = torch.as_tensor(
        getattr(getattr(articulation, "cfg", None), "body_scale", (1.0, 1.0, 1.0)),
        dtype=vertices.dtype,
        device=vertices.device,
    ).reshape(-1)
    if (
        scale.shape != (3,)
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0.0)
    ):
        raise ValueError("USD articulation body_scale must be finite and positive.")
    return vertices * scale, triangles


class _InitialInteractionGeometryProvider:
    """Expose a scaled live USD snapshot through the Atomic Action protocol."""

    def __init__(
        self,
        articulation: Any,
        target_link_name: str,
        *,
        target_vertices: torch.Tensor,
        target_triangles: torch.Tensor,
        owning_link_vertices: torch.Tensor,
        owning_link_triangles: torch.Tensor,
    ) -> None:
        physical_link_names = tuple(
            str(link_name) for link_name in articulation.link_names
        )
        if target_link_name not in physical_link_names:
            raise ValueError(f"Unknown articulation link {target_link_name!r}.")
        self.device = torch.device(getattr(articulation, "device", "cpu"))
        self.target_link_name = f"__gensim_interaction__:{target_link_name}"
        if self.target_link_name in physical_link_names:
            raise ValueError("Generated interaction link name collides with live USD.")
        self.link_names = (*physical_link_names, self.target_link_name)
        self.initial_qpos_joint_names = tuple(
            str(joint_name) for joint_name in articulation.joint_names
        )
        qpos = torch.as_tensor(
            articulation.get_qpos(),
            dtype=torch.float32,
            device=self.device,
        )
        if qpos.ndim != 2 or qpos.shape[0] < 1:
            raise ValueError("Generated articulation qpos must have shape (B, J).")
        self.initial_qpos = qpos[0].clone()
        self._articulation = articulation
        self._physical_target_link_name = target_link_name
        self._target_geometry = (
            torch.as_tensor(
                target_vertices,
                dtype=torch.float32,
                device=self.device,
            ).clone(),
            torch.as_tensor(
                target_triangles,
                dtype=torch.int64,
                device=self.device,
            ).clone(),
        )
        self._owning_link_geometry = (
            torch.as_tensor(
                owning_link_vertices,
                dtype=torch.float32,
                device=self.device,
            ).clone(),
            torch.as_tensor(
                owning_link_triangles,
                dtype=torch.int64,
                device=self.device,
            ).clone(),
        )
        self._link_poses = {
            link_name: self._snapshot_link_pose(link_name)
            for link_name in physical_link_names
        }
        self._link_poses[self.target_link_name] = self._link_poses[
            target_link_name
        ].clone()

    def _snapshot_link_pose(self, link_name: str) -> torch.Tensor:
        pose = torch.as_tensor(
            self._articulation.get_link_pose(link_name, to_matrix=True),
            dtype=torch.float32,
            device=self.device,
        )
        if pose.ndim == 3:
            pose = pose[0]
        if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all().item()):
            raise ValueError(
                f"Articulation link {link_name!r} pose must be finite with "
                "shape (4, 4)."
            )
        return pose.clone()

    def compute_fk(
        self,
        qpos: torch.Tensor,
        *,
        link_names: Sequence[str],
        qpos_joint_names: Sequence[str],
    ) -> torch.Tensor:
        """Return the captured reset snapshot for the explicitly named state."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        if qpos.ndim != 2 or qpos.shape[0] != 1:
            raise ValueError("Initial interaction FK requires one qpos row.")
        if tuple(qpos_joint_names) != self.initial_qpos_joint_names:
            raise ValueError("Initial interaction joint names changed after reset.")
        if not torch.allclose(qpos[0], self.initial_qpos, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("Initial interaction qpos changed after snapshot capture.")
        requested_links = tuple(str(link_name) for link_name in link_names)
        if any(link_name not in self._link_poses for link_name in requested_links):
            raise ValueError("Initial interaction FK requested an unknown link.")
        return torch.stack(
            [self._link_poses[link_name] for link_name in requested_links],
            dim=0,
        ).unsqueeze(0)

    def get_link_vert_face(
        self,
        link_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled physical geometry or the exact interaction submesh."""
        if link_name == self.target_link_name:
            return self._target_geometry
        if link_name == self._physical_target_link_name:
            return self._owning_link_geometry
        return _scaled_link_geometry(self._articulation, link_name)

    def get_parent_joint_chain(self, link_name: str) -> tuple[Any, ...]:
        """Delegate the synthetic handle to its owning moving link."""
        physical_link_name = (
            self._physical_target_link_name
            if link_name == self.target_link_name
            else link_name
        )
        return tuple(self._articulation.get_parent_joint_chain(physical_link_name))


def _sample_interaction_point_clouds(
    articulation: Any,
    target_link_name: str,
    *,
    target_vertices: torch.Tensor,
    target_triangles: torch.Tensor,
    owning_link_vertices: torch.Tensor,
    owning_link_triangles: torch.Tensor,
    articulation_point_count: int = 100_000,
    target_point_count: int = 5_000,
) -> dict[str, torch.Tensor]:
    """Adapt a generated USD handle to the tutorial geometry contract."""
    from embodichain.lab.sim.atomic_actions import (
        sample_initial_articulation_geometry,
    )

    provider = _InitialInteractionGeometryProvider(
        articulation,
        target_link_name,
        target_vertices=target_vertices,
        target_triangles=target_triangles,
        owning_link_vertices=owning_link_vertices,
        owning_link_triangles=owning_link_triangles,
    )
    return sample_initial_articulation_geometry(
        provider,
        provider.target_link_name,
        initial_qpos=provider.initial_qpos,
        initial_qpos_joint_names=provider.initial_qpos_joint_names,
        body_scale=(1.0, 1.0, 1.0),
        articulation_point_count=articulation_point_count,
        target_point_count=target_point_count,
    ).to_object_geometry()


def _usd_revolute_limits(
    articulation: Any,
    joint_info: Any,
) -> tuple[float, float] | None:
    """Read authored USD degree limits and return their radian equivalent."""
    path = Path(str(getattr(getattr(articulation, "cfg", None), "fpath", "")))
    if not path.is_file():
        return None
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(path.as_posix())
    except (ImportError, RuntimeError):
        return None
    if stage is None:
        return None
    requested_name = str(getattr(joint_info, "name", "")).strip()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        authored_name = prim.GetAttribute("articraft:name").Get()
        joint_name = str(authored_name or prim.GetName()).strip()
        if requested_name and joint_name != requested_name:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            return None
        limits = torch.deg2rad(torch.tensor([float(lower), float(upper)]))
        if not torch.isfinite(limits).all() or limits[0] >= limits[1]:
            return None
        return float(limits[0]), float(limits[1])
    return None


def _named_link_geometry(
    articulation: Any,
    link_name: str,
    *,
    name_tokens: Iterable[str] = (),
    mesh_names: Iterable[str] = (),
    excluded_mesh_names: Iterable[str] = (),
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract selected USD shape meshes in the owning articulation-link frame."""
    source = Path(str(getattr(getattr(articulation, "cfg", None), "fpath", "")))
    if source.suffix.lower() not in {".usd", ".usda", ".usdc"} or not source.is_file():
        return None
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(source.as_posix())
    if stage is None:
        return None
    link_prim = next(
        (
            prim
            for prim in stage.Traverse()
            if prim.GetName() == link_name and "/rigid_bodies/" in str(prim.GetPath())
        ),
        None,
    )
    if link_prim is None:
        return None
    exact_names = {str(name).strip() for name in mesh_names if str(name).strip()}
    excluded_names = {
        str(name).strip() for name in excluded_mesh_names if str(name).strip()
    }
    if exact_names & excluded_names:
        raise ValueError("Selected and excluded USD mesh names must be disjoint.")
    tokens = tuple(str(token).casefold() for token in name_tokens)
    mesh_prims = []
    for prim in Usd.PrimRange(link_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.GetName() in excluded_names:
            continue
        if exact_names:
            if prim.GetName() in exact_names:
                mesh_prims.append(prim)
        elif not tokens or any(token in prim.GetName().casefold() for token in tokens):
            mesh_prims.append(prim)
    if exact_names:
        if {str(prim.GetName()) for prim in mesh_prims} != exact_names:
            return None
    elif tokens:
        primary_mesh_prims = [
            prim
            for prim in mesh_prims
            if not any(
                token in prim.GetName().casefold()
                for token in ("mount", "bracket", "base", "support", "hinge")
            )
        ]
        if primary_mesh_prims:
            mesh_prims = primary_mesh_prims
    if not mesh_prims:
        return None

    cache = UsdGeom.XformCache()
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    scale = torch.as_tensor(
        getattr(getattr(articulation, "cfg", None), "body_scale", (1.0, 1.0, 1.0)),
        dtype=torch.float32,
    ).reshape(3)
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or ()
        counts = mesh.GetFaceVertexCountsAttr().Get() or ()
        indices = mesh.GetFaceVertexIndicesAttr().Get() or ()
        if not points or not counts or not indices:
            continue
        transform, _ = cache.ComputeRelativeTransform(prim, link_prim)
        offset = len(vertices)
        vertices.extend(
            [float(value) for value in transform.Transform(Gf.Vec3d(point))]
            for point in points
        )
        cursor = 0
        for count in counts:
            face = [int(index) + offset for index in indices[cursor : cursor + count]]
            cursor += int(count)
            triangles.extend(
                [face[0], face[index], face[index + 1]]
                for index in range(1, len(face) - 1)
            )
    if not vertices or not triangles:
        return None
    vertex_tensor = torch.tensor(vertices, dtype=torch.float32) * scale
    triangle_tensor = torch.tensor(triangles, dtype=torch.int64)
    return vertex_tensor, triangle_tensor
