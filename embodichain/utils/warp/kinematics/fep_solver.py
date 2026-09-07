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

"""Double-precision, per-candidate numerical FEP correction on Warp CPU/CUDA."""

from __future__ import annotations

import warp as wp

__all__ = ["FEPParam", "fep_ik"]

_Vec6 = wp.types.vector(length=6, dtype=wp.float64)
_Vec7 = wp.types.vector(length=7, dtype=wp.float64)
_Mat37 = wp.types.matrix(shape=(3, 7), dtype=wp.float64)
_Mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_Mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float64)


@wp.struct
class FEPParam:
    """Iteration cap, pose tolerances, damping and step bounds for ``fep_ik``."""

    max_iterations: int
    adaptive_damping: bool
    pos_eps: wp.float64
    rot_eps: wp.float64
    damp: wp.float64
    step_size: wp.float64
    max_step: wp.float64


@wp.func
def _rotation(axis: wp.vec3d, angle: wp.float64) -> wp.mat33d:
    return wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle))


@wp.func
def _position(pose: wp.mat44d) -> wp.vec3d:
    return wp.vec3d(pose[0, 3], pose[1, 3], pose[2, 3])


@wp.func
def _orientation(pose: wp.mat44d) -> wp.mat33d:
    result = wp.mat33d()
    for row in range(3):
        for col in range(3):
            result[row, col] = pose[row, col]
    return result


@wp.func
def _rotation_error(target: wp.mat33d, actual: wp.mat33d) -> wp.vec3d:
    rotation = target * wp.transpose(actual)
    quat = wp.normalize(wp.quat_from_matrix(rotation))
    if quat[3] < wp.float64(0.0):
        quat = -quat
    vector = wp.vec3d(quat[0], quat[1], quat[2])
    sine = wp.length(vector)
    scale = wp.float64(2.0)
    if sine > wp.float64(1e-12):
        scale = wp.float64(2.0) * wp.atan2(sine, quat[3]) / sine
    return scale * vector


@wp.func
def _solve_spd(system: _Mat66, rhs: _Vec6) -> _Vec6:
    """Cholesky solve of the six-dimensional damped normal system."""
    lower = _Mat66()
    for row in range(6):
        for col in range(row + 1):
            value = system[row, col]
            for k in range(col):
                value -= lower[row, k] * lower[col, k]
            if row == col:
                lower[row, col] = wp.sqrt(wp.max(value, wp.float64(1e-30)))
            else:
                lower[row, col] = value / lower[col, col]
    forward = _Vec6()
    for row in range(6):
        value = rhs[row]
        for col in range(row):
            value -= lower[row, col] * forward[col]
        forward[row] = value / lower[row, row]
    result = _Vec6()
    for offset in range(6):
        row = 5 - offset
        value = forward[row]
        for col in range(row + 1, 6):
            value -= lower[col, row] * result[col]
        result[row] = value / lower[row, row]
    return result


@wp.kernel
def fep_ik(
    params: FEPParam,
    origins: wp.array(dtype=wp.mat44d),
    axes: wp.array(dtype=wp.vec3d),
    tcp: wp.array(dtype=wp.mat44d),
    targets: wp.array(dtype=wp.mat44d),
    seeds: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    validity: wp.array(dtype=wp.int32),
    solutions: wp.array2d(dtype=wp.float64),
):
    """Correct one seven-axis seed per thread using bounded damped least squares.

    Args:
        params: Iteration and convergence settings.
        origins: Seven pre-motion transforms followed by the fixed flange transform.
        axes: Seven normalized joint axes in their local joint frames.
        tcp: One flange-to-TCP transform.
        targets: Root-relative TCP targets, one per candidate.
        seeds: Candidate joints, shape ``(N, 7)``.
        lower: Seven lower joint bounds.
        upper: Seven upper joint bounds.
        validity: Output success flags, shape ``(N,)``.
        solutions: Output corrected joints, shape ``(N, 7)``.
    """
    tid = wp.tid()
    joints = _Vec7()
    for joint in range(7):
        joints[joint] = wp.clamp(seeds[tid, joint], lower[joint], upper[joint])
    validity[tid] = 0
    tip = origins[7] * tcp[0]
    tip_position = _position(tip)
    tip_rotation = _orientation(tip)
    target_position = _position(targets[tid])
    target_rotation = _orientation(targets[tid])
    for iteration in range(params.max_iterations + 1):
        rotation = wp.identity(n=3, dtype=wp.float64)
        position = wp.vec3d()
        positions = _Mat37()
        world_axes = _Mat37()
        for joint in range(7):
            position = position + rotation * _position(origins[joint])
            rotation = rotation * _orientation(origins[joint])
            axis = rotation * axes[joint]
            for row in range(3):
                positions[row, joint] = position[row]
                world_axes[row, joint] = axis[row]
            # FK uses the same quantized joints that the public API returns.
            angle = wp.float64(wp.float32(joints[joint]))
            rotation = rotation * _rotation(axes[joint], angle)
        actual_position = position + rotation * tip_position
        actual_rotation = rotation * tip_rotation
        position_error = target_position - actual_position
        rotation_error = _rotation_error(target_rotation, actual_rotation)
        if (
            wp.length(position_error) <= params.pos_eps
            and wp.length(rotation_error) <= params.rot_eps
        ):
            validity[tid] = 1
            break
        if iteration == params.max_iterations:
            break
        jacobian = _Mat67()
        error = _Vec6()
        for row in range(3):
            error[row] = position_error[row]
            error[row + 3] = rotation_error[row]
        for joint in range(7):
            axis = wp.vec3d(
                world_axes[0, joint], world_axes[1, joint], world_axes[2, joint]
            )
            position = wp.vec3d(
                positions[0, joint], positions[1, joint], positions[2, joint]
            )
            linear = wp.cross(axis, actual_position - position)
            for row in range(3):
                jacobian[row, joint] = linear[row]
                jacobian[row + 3, joint] = axis[row]
        system = jacobian * wp.transpose(jacobian)
        damping_squared = params.damp * params.damp
        if params.adaptive_damping:
            damping_squared *= wp.clamp(
                wp.float64(10.0) * wp.length(error), wp.float64(1e-4), wp.float64(1.0)
            )
        for row in range(6):
            system[row, row] += damping_squared
        update = wp.transpose(jacobian) * _solve_spd(system, error) * params.step_size
        largest = wp.float64(1e-15)
        finite = True
        for joint in range(7):
            largest = wp.max(largest, wp.abs(update[joint]))
            finite = finite and wp.isfinite(update[joint])
        if not finite:
            break
        scale = wp.min(wp.float64(1.0), params.max_step / largest)
        for joint in range(7):
            joints[joint] = wp.clamp(
                joints[joint] + scale * update[joint], lower[joint], upper[joint]
            )
    for joint in range(7):
        solutions[tid, joint] = joints[joint]
