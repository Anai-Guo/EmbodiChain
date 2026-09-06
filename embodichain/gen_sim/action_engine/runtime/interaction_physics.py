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

"""Observe articulation transitions and preserve configured GenSim physics."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch

__all__: list[str] = []


def _joint_motion_evidence(
    *,
    initial: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    executed: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    """Freeze row-local physical completion before releasing an articulation.

    Numeric lists retain finite measurements and use ``None`` for unavailable
    values, so failed observations do not turn into fabricated zeros or JSON NaN.
    Boolean masks stay on the input device for executor gating.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number.")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    for name, value in (
        ("initial", initial),
        ("target", target),
        ("observed", observed),
        ("executed", executed),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor.")
        if name == "executed":
            if value.dtype != torch.bool:
                raise TypeError("executed must be a boolean tensor.")
        elif not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor.")
        if value.ndim != 1 or not value.numel() or value.shape != initial.shape:
            raise ValueError("Joint motion tensors must share non-empty shape (B,).")
        if value.device != initial.device:
            raise ValueError("Joint motion tensors must share the same device.")

    initial = initial.detach().clone()
    target = target.detach().clone()
    observed = observed.detach().clone()
    executed = executed.detach().clone()
    requested_delta = target - initial
    delta = observed - initial
    target_error = torch.abs(observed - target)
    initial_finite = (
        torch.isfinite(initial)
        & torch.isfinite(target)
        & torch.isfinite(requested_delta)
    )
    finite = (
        initial_finite
        & torch.isfinite(observed)
        & torch.isfinite(delta)
        & torch.isfinite(target_error)
    )
    initial_satisfied = initial_finite & (torch.abs(requested_delta) <= tolerance)
    direction_ok = finite & (
        ((requested_delta > 0.0) & (delta > 0.0))
        | ((requested_delta < 0.0) & (delta < 0.0))
    )
    success = (
        executed
        & finite
        & ~initial_satisfied
        & direction_ok
        & (target_error <= tolerance)
    )

    def numeric_values(value: torch.Tensor) -> list[float | None]:
        return [item if math.isfinite(item) else None for item in value.cpu().tolist()]

    return {
        "phase": "core_terminal_before_release",
        "initial": numeric_values(initial),
        "target": numeric_values(target),
        "observed": numeric_values(observed),
        "delta": numeric_values(delta),
        "target_error": numeric_values(target_error),
        "tolerance": tolerance,
        "finite": finite.detach().clone(),
        "initial_satisfied": initial_satisfied.detach().clone(),
        "direction_ok": direction_ok.detach().clone(),
        "executed": executed,
        "success": success.detach().clone(),
    }


def _restore_gravity_contract(
    articulation: Any,
    *,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> dict[str, Any] | None:
    """Reapply a generated USD's explicit gravity flag and verify live attributes.

    The general physical-attribute initialization can overwrite the earlier
    articulation gravity setting. Only mismatching environment rows are changed;
    this does not choose a new gravity policy or write any joint state.
    """
    cfg = getattr(articulation, "cfg", None)
    if Path(str(getattr(cfg, "fpath", ""))).suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    } or not hasattr(cfg, "enable_gravity"):
        return None
    configured = cfg.enable_gravity
    if type(configured) is not bool:
        raise TypeError("USD enable_gravity must be an explicit boolean.")
    qpos = articulation.get_qpos()
    links = list(articulation.link_names)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or not links:
        raise ValueError("Gravity contract requires non-empty environments and links.")
    if env_ids is None:
        env_ids = list(range(qpos.shape[0]))
    else:
        indices = torch.as_tensor(env_ids)
        if indices.ndim != 1 or (
            indices.numel() and indices.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("Gravity env_ids must be a one-dimensional integer list.")
        env_ids = indices.cpu().tolist()
        if len(set(env_ids)) != len(env_ids) or any(
            index < 0 or index >= qpos.shape[0] for index in env_ids
        ):
            raise ValueError("Gravity env_ids must be unique and in range.")

    def observed_flags() -> list[list[bool]]:
        if not env_ids:
            return []
        attrs = articulation.get_link_physical_attr(link_names=links, env_ids=env_ids)
        expected_count = len(env_ids) * len(links)
        if len(attrs) != expected_count:
            raise ValueError(
                f"Gravity attribute count must be {expected_count}, got {len(attrs)}."
            )
        flags = [getattr(attr, "has_gravity", None) for attr in attrs]
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("Observed link has_gravity flags must be boolean.")
        return [
            flags[index * len(links) : (index + 1) * len(links)]
            for index in range(len(env_ids))
        ]

    before = observed_flags()
    changed_env_ids = [
        env_ids[index]
        for index, row in enumerate(before)
        if any(flag != configured for flag in row)
    ]
    if changed_env_ids:
        articulation.set_gravity(configured, env_ids=changed_env_ids)
    after = observed_flags()
    if any(flag != configured for row in after for flag in row):
        raise RuntimeError(
            f"USD gravity contract was not restored: configured={configured}, "
            f"observed={after}."
        )
    return {
        "configured": configured,
        "before": before,
        "after": after,
        "changed": bool(changed_env_ids),
        "env_ids": changed_env_ids,
        "link_names": links,
        "direct_qpos_write": False,
    }
