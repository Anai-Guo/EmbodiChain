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

from collections.abc import Iterable
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
    name_tokens: Iterable[str],
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
    tokens = tuple(str(token).casefold() for token in name_tokens)
    mesh_prims = [
        prim
        for prim in Usd.PrimRange(link_prim)
        if prim.IsA(UsdGeom.Mesh)
        and any(token in prim.GetName().casefold() for token in tokens)
    ]
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
