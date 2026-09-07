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

"""Branch-seeded numerical FEP IK for seven-revolute-joint URDF chains.

Follows HolisticMotion's FEPKinematics numerical facade, including its joint-sign
configuration convention. This is not a closed-form analytical FEP solver.
"""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import product
from typing import Any, List

import numpy as np
import torch

from embodichain.lab.sim.solvers.base_solver import BaseSolver, SolverCfg
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from embodichain.utils import configclass

# np and List also resolve inherited SolverCfg annotations on the generated init.

__all__ = ["FEPSolverCfg", "FEPSolver"]

_METHODS = (
    "seeded_numerical",
    "configuration",
    "all_configurations",
    "nearest_redundancy",
)


@configclass
class FEPSolverCfg(SolverCfg):
    """Configure a seven-axis FEP numerical solver.

    Geometry comes from the selected URDF serial chain, including fixed joints.
    ``auto`` uses Warp on CPU and CUDA. Both backends use double precision during
    correction and return float32 joints through the BaseSolver API.
    """

    class_type: str = "FEPSolver"
    backend: str = "auto"
    """Execution backend: ``auto``, ``python`` (PyTorch), or ``warp``."""

    solve_method: str = "seeded_numerical"
    """Seeded solve, configuration-constrained solve, eight branches, or radial search."""

    max_iterations: int = 200
    """Maximum damped least-squares updates per candidate."""

    pos_eps: float = 1e-5
    """TCP position tolerance in metres."""

    rot_eps: float = 1e-5
    """TCP rotation tolerance in radians."""

    damp: float = 0.01
    """Maximum damping lambda, or fixed lambda when adaptive damping is disabled."""

    adaptive_damping: bool = True
    """Reduce damping with pose residual, down to one percent of ``damp``."""

    step_size: float = 1.0
    """Multiplier applied to each damped least-squares update."""

    max_step: float = 0.35
    """Maximum absolute joint change per iteration, in radians."""

    redundancy_step: float = math.pi / 36.0
    """Radial increment for joint-seven seed search, in radians."""

    redundancy_samples: int = 37
    """Number of radial levels, including zero; each nonzero level tries both signs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in ("auto", "python", "warp"):
            raise ValueError(f"Unknown FEP backend: {self.backend!r}")
        if self.solve_method not in _METHODS:
            raise ValueError(f"Unknown FEP solve method: {self.solve_method!r}")
        if not isinstance(self.adaptive_damping, bool):
            raise ValueError("adaptive_damping must be a boolean")
        for name in ("max_iterations", "redundancy_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "pos_eps",
            "rot_eps",
            "damp",
            "step_size",
            "max_step",
            "redundancy_step",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def init_solver(
        self, device: str | torch.device = torch.device("cpu"), **kwargs: Any
    ) -> FEPSolver:
        """Construct the solver and apply its configured TCP.

        Args:
            device: PyTorch execution device.
            **kwargs: BaseSolver initialization arguments, including a serial chain.

        Returns:
            Initialized FEP solver.
        """
        solver = FEPSolver(self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class FEPSolver(BaseSolver):
    """Solve offset 7R chains with configuration seeds and numerical correction.

    Args:
        cfg: Solver settings and URDF chain selection.
        device: CPU or CUDA device.
        **kwargs: Optional ``pk_serial_chain`` and other BaseSolver arguments.

    Raises:
        ValueError: The chain is not seven revolute joints, or joint order differs.
    """

    def __init__(
        self, cfg: FEPSolverCfg, device: str | torch.device = "cpu", **kwargs: Any
    ) -> None:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        chain = kwargs.pop("pk_serial_chain", None)
        if chain is None:
            chain = create_pk_serial_chain(
                urdf_path=cfg.urdf_path,
                end_link_name=cfg.end_link_name,
                root_link_name=cfg.root_link_name,
                device=device,
            )
        else:
            # Preserve the caller's chain dtype/device while satisfying inherited FK.
            chain = deepcopy(chain).to(dtype=torch.float32, device=device)
        names = chain.get_joint_parameter_names()
        moving = [f for f in chain._serial_frames if f.joint.joint_type != "fixed"]
        if len(moving) != 7 or any(f.joint.joint_type != "revolute" for f in moving):
            raise ValueError("FEP requires exactly seven revolute/continuous joints")
        if cfg.joint_names is not None and list(cfg.joint_names) != names:
            raise ValueError("FEP joint_names must match the URDF serial-chain order")
        resolved_cfg = cfg.copy()
        resolved_cfg.joint_names = names
        super().__init__(resolved_cfg, device, pk_serial_chain=chain, **kwargs)
        self.dof = 7
        # BaseSolver's injected-chain path leaves compilation to its caller.
        self.compiled_fk = chain.forward_kinematics_tensor
        self._chain = deepcopy(chain).to(dtype=torch.float64, device=device)
        self.backend = "warp" if cfg.backend == "auto" else cfg.backend
        self._origins, self._axes = self._pack_chain()
        self._reach_bound = torch.linalg.vector_norm(
            self._origins[1:7, :3, 3], dim=-1
        ).sum()
        x, y, z = self._axes.unbind(-1)
        zero = torch.zeros_like(x)
        self._axis_skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
        ).reshape(7, 3, 3)
        self._axis_outer = self._axes[:, :, None] * self._axes[:, None, :]
        self._identity3 = torch.eye(3, dtype=torch.float64, device=device)
        if (
            not torch.isfinite(self._origins).all()
            or not torch.isfinite(self._axes).all()
        ):
            raise ValueError("FEP chain transforms and axes must be finite")
        self._warp_model = None

    def _pack_chain(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold fixed frames into seven pre-motion origins and one flange origin."""
        pending = torch.eye(4, dtype=torch.float64, device=self.device)
        origins, axes = [], []
        for frame in self._chain._serial_frames:
            for offset in (frame.link.offset, frame.joint.offset):
                if offset is not None:
                    pending = pending @ offset.get_matrix()[0]
            if frame.joint.joint_type != "fixed":
                origins.append(pending)
                axes.append(frame.joint.axis)
                pending = torch.eye(4, dtype=torch.float64, device=self.device)
        origins.append(pending)
        return torch.stack(origins).contiguous(), torch.stack(axes).contiguous()

    @staticmethod
    def get_configuration(qpos: torch.Tensor) -> torch.Tensor:
        """Extract HolisticMotion's shoulder/elbow/wrist signs and redundancy seed.

        Args:
            qpos: Joint angles, shape ``(..., 7)``.

        Returns:
            Tensor ``(..., 4)``: signs of joints 2, 4, 6 (zero maps to +1),
            followed by joint 7. These are joint-coordinate branch labels.
        """
        if qpos.shape[-1] != 7:
            raise ValueError("FEP configurations require seven joint angles")
        signs = torch.where(qpos[..., [1, 3, 5]] >= 0, 1.0, -1.0)
        return torch.cat((signs, qpos[..., 6:7]), dim=-1)

    @torch.no_grad()
    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        return_all_solutions: bool = False,
        *,
        solve_method: str | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IK for TCP poses relative to the selected chain root.

        Args:
            target_xpos: Homogeneous poses, shape ``(4, 4)`` or ``(N, 4, 4)``.
            qpos_seed: Shape ``(7,)`` (broadcast) or ``(N, 7)``; defaults to zero.
            return_all_solutions: Enumerate eight branch seeds and return all
                valid, distinct candidates sorted by weighted periodic distance.
            solve_method: Override the configured method for this call.
            **kwargs: Reserved BaseSolver arguments.

        Returns:
            Boolean validity ``(N,)`` and float32 joints ``(N, 7)``; with all
            solutions, shapes are ``(N, 8)`` and ``(N, 8, 7)``. Invalid slots
            preserve the original input seed. Numerical enumeration is not
            exhaustive and redundancy is a seed value, not a locked joint.

        Raises:
            ValueError: Input shapes, finite values, rigid transforms, method,
                or effective joint limits are invalid.
        """
        method = self.cfg.solve_method if solve_method is None else solve_method
        if method not in _METHODS:
            raise ValueError(f"Unknown FEP solve method: {method!r}")
        if return_all_solutions:
            method = "all_configurations"
        target = torch.as_tensor(target_xpos, dtype=torch.float64, device=self.device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.ndim != 3 or target.shape[1:] != (4, 4):
            raise ValueError("target_xpos must have shape (4, 4) or (N, 4, 4)")
        n = target.shape[0]
        seed = (
            torch.zeros((n, 7), dtype=torch.float64, device=self.device)
            if qpos_seed is None
            else torch.as_tensor(qpos_seed, dtype=torch.float64, device=self.device)
        )
        if seed.shape == (7,):
            seed = seed.expand(n, -1)
        if seed.shape != (n, 7):
            raise ValueError("qpos_seed must have shape (7,) or (N, 7)")
        tcp = torch.as_tensor(self.tcp_xpos, dtype=torch.float64, device=self.device)
        self._validate_poses(torch.cat((target, tcp.unsqueeze(0))))
        lower = self.lower_qpos_limits.to(dtype=torch.float64)
        upper = self.upper_qpos_limits.to(dtype=torch.float64)
        # URDF limits remain hard bounds even after a runtime limit update.
        lower = torch.maximum(lower, self._chain.low)
        upper = torch.minimum(upper, self._chain.high)
        if (
            lower.shape != (7,)
            or upper.shape != (7,)
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or torch.any(lower > upper)
        ):
            raise ValueError("FEP requires finite, nonempty effective joint limits")
        if not torch.isfinite(seed).all():
            raise ValueError("qpos_seed must be finite")
        weights = torch.as_tensor(self.ik_nearest_weight, device=self.device)
        if (
            weights.shape != (7,)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(
                "FEP nearest weights must be seven finite nonnegative values"
            )
        within_reach = self._within_reach_bound(target, tcp)
        if within_reach.all():
            return self._solve_validated(
                target, seed, tcp, lower, upper, weights, method, return_all_solutions
            )
        shape = (n, 8) if return_all_solutions else (n,)
        validity = torch.zeros(shape, dtype=torch.bool, device=self.device)
        result = (
            seed[:, None].expand(-1, 8, -1).clone()
            if return_all_solutions
            else seed.clone()
        ).float()
        if within_reach.any():
            validity[within_reach], result[within_reach] = self._solve_validated(
                target[within_reach],
                seed[within_reach],
                tcp,
                lower,
                upper,
                weights,
                method,
                return_all_solutions,
            )
        return validity, result

    def _within_reach_bound(
        self, target: torch.Tensor, tcp: torch.Tensor
    ) -> torch.Tensor:
        """Conservative first-to-last joint distance bound, including TCP tolerance.

        Undo the fixed tip using the requested orientation. The distance between
        the first and last moving joint cannot exceed the intervening lengths.
        Passing this necessary condition does not establish reachability.
        """
        tip = self._origins[7] @ tcp
        offset = tip[:3, :3].T @ tip[:3, 3]
        wrist = target[:, :3, 3] - (target[:, :3, :3] @ offset[:, None]).squeeze(-1)
        tip_length = torch.linalg.vector_norm(tip[:3, 3])
        # Include permitted pose residuals and float32 URDF/FK roundoff.
        margin = (
            self.cfg.pos_eps
            + self.cfg.rot_eps * tip_length
            + 1e-5 * (1 + self._reach_bound + tip_length)
        )
        distance = torch.linalg.vector_norm(wrist - self._origins[0, :3, 3], dim=-1)
        return distance <= self._reach_bound + margin

    def _solve_validated(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        tcp: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        method: str,
        return_all_solutions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the validated targets inside the conservative reach bound."""
        n = len(target)
        if method == "nearest_redundancy":
            return self._search_redundancy(target, seed, tcp, lower, upper, weights)
        candidates, branches = self._make_seeds(seed, method)
        count = candidates.shape[1]
        if n == 0:
            validity = torch.empty((0, count), dtype=torch.bool, device=self.device)
            joints = candidates.float()
            return (
                (validity, joints)
                if return_all_solutions
                else (validity[:, 0], joints[:, 0])
            )
        flat_target = (
            target[:, None].expand(-1, count, -1, -1).reshape(-1, 4, 4).contiguous()
        )
        valid, joints = self._correct_candidates(
            flat_target, candidates.reshape(-1, 7), tcp, lower, upper
        )
        joints = joints.reshape(n, count, 7)
        valid = valid.reshape(n, count)
        if branches is not None:
            valid &= (self.get_configuration(joints)[..., :3] == branches).all(-1)
        if count == 1:
            return valid[:, 0], torch.where(valid, joints[:, 0], seed).float()
        delta = self._wrapped(joints - seed[:, None])
        costs = ((delta * weights) ** 2).sum(-1).masked_fill(~valid, float("inf"))
        order = torch.argsort(costs, dim=1, stable=True)
        joints = joints.gather(1, order[..., None].expand(-1, -1, 7))
        valid = valid.gather(1, order)
        if return_all_solutions:
            for i in range(1, count):
                duplicate = (
                    self._wrapped(joints[:, i : i + 1] - joints[:, :i]) ** 2
                ).sum(-1) < 1e-12
                valid[:, i] &= ~(duplicate & valid[:, :i]).any(-1)
        joints = torch.where(valid[..., None], joints, seed[:, None]).float()
        return (valid, joints) if return_all_solutions else (valid[:, 0], joints[:, 0])

    def _correct_candidates(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        tcp: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one candidate batch and validate its actual float32 output."""
        flat_seed = seed.clamp(lower, upper).contiguous()
        if self.backend == "python":
            valid, joints = self._solve_python(target, flat_seed, tcp, lower, upper)
        else:
            valid, joints = self._solve_warp(target, flat_seed, tcp, lower, upper)
        # Validate the actual returned float32 joints against shared URDF FK.
        joints = joints.float().double()
        actual = self._chain.forward_kinematics_tensor(joints)[-1] @ tcp
        error = self._pose_error(target, actual)
        valid &= torch.linalg.vector_norm(error[:, :3], dim=-1) <= self.cfg.pos_eps
        valid &= torch.linalg.vector_norm(error[:, 3:], dim=-1) <= self.cfg.rot_eps
        valid &= torch.isfinite(joints).all(-1)
        valid &= ((joints >= lower - 1e-7) & (joints <= upper + 1e-7)).all(-1)
        return valid, joints

    def _search_redundancy(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        tcp: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search successive radii only for targets that have not yet converged."""
        result = seed.clone()
        success = torch.zeros(len(seed), dtype=torch.bool, device=self.device)
        remaining = torch.arange(len(seed), device=self.device)
        for level in range(self.cfg.redundancy_samples):
            radius = level * self.cfg.redundancy_step
            if len(remaining) == 0 or radius > math.pi + 1e-12:
                break
            offsets = seed.new_tensor([0.0] if level == 0 else [-radius, radius])
            count = len(offsets)
            trials = seed[remaining, None].repeat(1, count, 1)
            trials[..., 6] += offsets
            targets = (
                target[remaining, None]
                .expand(-1, count, -1, -1)
                .reshape(-1, 4, 4)
                .contiguous()
            )
            valid, joints = self._correct_candidates(
                targets, trials.reshape(-1, 7), tcp, lower, upper
            )
            valid = valid.reshape(-1, count)
            joints = joints.reshape(-1, count, 7)
            delta = self._wrapped(joints - seed[remaining, None])
            costs = ((delta * weights) ** 2).sum(-1).masked_fill(~valid, float("inf"))
            selected = costs.argmin(-1)
            rows = torch.arange(len(remaining), device=self.device)
            solved = valid[rows, selected]
            result[remaining[solved]] = joints[rows, selected][solved]
            success[remaining[solved]] = True
            remaining = remaining[~solved]
        return success, result.float()

    @staticmethod
    def _wrapped(delta: torch.Tensor) -> torch.Tensor:
        return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi

    @staticmethod
    def _validate_poses(poses: torch.Tensor) -> None:
        if not torch.isfinite(poses).all():
            raise ValueError("FEP target and TCP transforms must be finite")
        rotation = poses[:, :3, :3]
        eye = torch.eye(3, dtype=poses.dtype, device=poses.device).expand_as(rotation)
        row = poses.new_tensor([0, 0, 0, 1]).expand_as(poses[:, 3])
        if (
            not torch.allclose(
                rotation.transpose(-1, -2) @ rotation, eye, atol=1e-5, rtol=0
            )
            or not torch.allclose(
                torch.linalg.det(rotation),
                poses.new_ones(len(poses)),
                atol=1e-5,
                rtol=0,
            )
            or not torch.allclose(poses[:, 3], row, atol=1e-7, rtol=0)
        ):
            raise ValueError("FEP target and TCP must be rigid homogeneous transforms")

    def _make_seeds(
        self, seed: torch.Tensor, method: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        branches = None
        candidates = seed[:, None].clone()
        if method in ("configuration", "all_configurations"):
            branches = (
                seed.new_tensor(list(product((-1, 1), repeat=3)))[None].expand(
                    len(seed), -1, -1
                )
                if method == "all_configurations"
                else self.get_configuration(seed)[:, None, :3]
            )
            candidates = seed[:, None].repeat(1, branches.shape[1], 1)
            candidates[..., [1, 3, 5]] = branches * seed[
                :, None, [1, 3, 5]
            ].abs().clamp_min(0.35)
        return candidates, branches

    @staticmethod
    def _pose_error(target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        from pytorch_kinematics.transforms import (
            matrix_to_axis_angle,
            quaternion_to_axis_angle,
        )

        rotation = target[:, :3, :3] @ actual[:, :3, :3].transpose(-1, -2)
        if (
            rotation.is_cuda
            or (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) <= 2.0).any()
        ):
            rotation_error = matrix_to_axis_angle(rotation)
        else:
            # Use precisely the reference's positive-real quaternion branch
            # below 60 degrees, without allocating its other three candidates.
            # Preserve its arithmetic and axis-angle conversion, including the
            # small-angle series, to avoid perturbing difficult trajectories.
            m00, m11, m22 = rotation[:, 0, 0], rotation[:, 1, 1], rotation[:, 2, 2]
            real = (1.0 + m00 + m11 + m22).sqrt()
            vector = torch.stack(
                (
                    real**2,
                    rotation[:, 2, 1] - rotation[:, 1, 2],
                    rotation[:, 0, 2] - rotation[:, 2, 0],
                    rotation[:, 1, 0] - rotation[:, 0, 1],
                ),
                dim=-1,
            )
            quaternion = vector / (2.0 * real[:, None])
            rotation_error = quaternion_to_axis_angle(quaternion)
        return torch.cat((target[:, :3, 3] - actual[:, :3, 3], rotation_error), dim=-1)

    def _solve_python(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        tcp: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joints = seed.clone()
        valid = torch.zeros(len(seed), dtype=torch.bool, device=self.device)
        active = torch.arange(len(seed), device=self.device)
        identity = torch.eye(6, dtype=seed.dtype, device=self.device)
        for iteration in range(self.cfg.max_iterations + 1):
            # Keep updates in float64, but solve for the representable output.
            # Otherwise rounding a just-converged candidate can invalidate it.
            evaluated = joints[active].float().double()
            jacobian, actual = self._evaluate(evaluated, tcp)
            error = self._pose_error(target[active], actual)
            converged = (
                torch.linalg.vector_norm(error[:, :3], dim=-1) <= self.cfg.pos_eps
            ) & (torch.linalg.vector_norm(error[:, 3:], dim=-1) <= self.cfg.rot_eps)
            valid[active[converged]] = True
            active = active[~converged]
            if len(active) == 0 or iteration == self.cfg.max_iterations:
                break
            jacobian, error = jacobian[~converged], error[~converged]
            damping_squared = error.new_full((len(active),), self.cfg.damp**2)
            if self.cfg.adaptive_damping:
                # Residual-scaled regularization preserves damping far from the
                # goal and avoids slow convergence along weak Jacobian modes.
                damping_squared *= (
                    10.0 * torch.linalg.vector_norm(error, dim=-1)
                ).clamp(1e-4, 1.0)
            system = (
                jacobian @ jacobian.transpose(1, 2)
                + damping_squared[:, None, None] * identity
            )
            update = (
                jacobian.transpose(1, 2)
                @ torch.linalg.solve(system, error.unsqueeze(-1))
            ).squeeze(-1) * self.cfg.step_size
            scale = (
                self.cfg.max_step / update.abs().amax(-1).clamp_min(1e-15)
            ).clamp_max(1)
            joints[active] = (joints[active] + scale[:, None] * update).clamp(
                lower, upper
            )
        return valid, joints

    def _evaluate(
        self, joints: torch.Tensor, tcp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute only the seven joint frames and TCP needed by correction.

        Public FK and final acceptance continue to use the shared URDF chain.
        The iteration path folds fixed frames and avoids storing full 4x4
        transforms for every intermediate URDF link in every batch row.
        """
        cosine = joints.cos()[..., None, None]
        sine = joints.sin()[..., None, None]
        motion = (
            cosine * self._identity3
            + (1 - cosine) * self._axis_outer
            + sine * self._axis_skew
        )
        rotation = self._identity3.expand(len(joints), -1, -1)
        position = joints.new_zeros((len(joints), 3))
        positions, axes = [], []
        for i in range(7):
            position = position + (rotation @ self._origins[i, :3, 3:4]).squeeze(-1)
            rotation = rotation @ self._origins[i, :3, :3]
            positions.append(position)
            axes.append((rotation @ self._axes[i, :, None]).squeeze(-1))
            rotation = rotation @ motion[:, i]
        tip = self._origins[7] @ tcp
        position = position + (rotation @ tip[:3, 3:4]).squeeze(-1)
        rotation = rotation @ tip[:3, :3]
        actual = torch.eye(4, dtype=joints.dtype, device=joints.device).repeat(
            len(joints), 1, 1
        )
        actual[:, :3, :3], actual[:, :3, 3] = rotation, position
        angular = torch.stack(axes, dim=1)
        linear = torch.linalg.cross(
            angular, position[:, None] - torch.stack(positions, dim=1), dim=-1
        )
        return torch.cat((linear, angular), dim=-1).transpose(1, 2), actual

    def _solve_warp(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        tcp: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import warp as wp

        from embodichain.utils.device_utils import standardize_device_string
        from embodichain.utils.warp.kinematics.fep_solver import FEPParam, fep_ik

        wp.init()
        if self._warp_model is None:
            self._warp_model = (
                wp.from_torch(self._origins, dtype=wp.mat44d),
                wp.from_torch(self._axes, dtype=wp.vec3d),
            )
        params = FEPParam()
        params.max_iterations = self.cfg.max_iterations
        params.adaptive_damping = self.cfg.adaptive_damping
        params.pos_eps, params.rot_eps = self.cfg.pos_eps, self.cfg.rot_eps
        params.damp, params.step_size, params.max_step = (
            self.cfg.damp,
            self.cfg.step_size,
            self.cfg.max_step,
        )
        joints = torch.empty_like(seed)
        valid = torch.zeros(len(seed), dtype=torch.int32, device=self.device)
        device = standardize_device_string(self.device)
        # Share the caller's PyTorch stream, so inputs and outputs are ordered.
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        wp.launch(
            fep_ik,
            dim=len(seed),
            inputs=[
                params,
                *self._warp_model,
                wp.from_torch(tcp[None].contiguous(), dtype=wp.mat44d),
                wp.from_torch(target, dtype=wp.mat44d),
                wp.from_torch(seed),
                wp.from_torch(lower.contiguous()),
                wp.from_torch(upper.contiguous()),
            ],
            outputs=[wp.from_torch(valid), wp.from_torch(joints)],
            device=device,
            stream=stream,
        )
        return valid.bool(), joints
