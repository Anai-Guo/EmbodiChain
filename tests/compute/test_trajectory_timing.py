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

"""Numerical contracts for timed joint trajectories."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute import trajectory


def test_nonuniform_quadratic_derivative() -> None:
    # q=t² has derivative 2t; endpoints use first-order one-sided slopes.
    t = torch.tensor([[0.0, 0.1, 0.4, 1.0]], dtype=torch.float64)
    dt = torch.cat([t[:, :1], t.diff(dim=1)], dim=1)
    q = t.square().unsqueeze(-1)
    velocity = trajectory.differentiate_positions(q, dt)
    torch.testing.assert_close(velocity[:, 1:-1, 0], 2 * t[:, 1:-1])
    assert velocity.dtype == q.dtype
    torch.testing.assert_close(
        velocity[:, [0, -1], 0], torch.tensor([[0.1, 1.4]], dtype=q.dtype)
    )


def test_batched_duplicates_padding_and_stationary_rows() -> None:
    q = torch.tensor([[[0.0], [1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0], [2.0]]])
    dt = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    velocity = trajectory.differentiate_positions(q, dt)
    torch.testing.assert_close(velocity[0, :, 0], torch.tensor([2.0, 2.0, 0.0, 0.0]))
    assert torch.equal(velocity[1], torch.zeros_like(velocity[1]))


@pytest.mark.parametrize("count", [0, 1, 2])
def test_short_constant_trajectories(count: int) -> None:
    q = torch.ones(2, count, 3)
    dt = torch.zeros(2, count)
    assert torch.equal(trajectory.differentiate_positions(q, dt), torch.zeros_like(q))


def test_zero_time_position_jump_rejected() -> None:
    with pytest.raises(ValueError, match="zero.*time|zero.*interval"):
        trajectory.differentiate_positions(
            torch.tensor([[[0.0], [1.0]]]), torch.zeros(1, 2)
        )


@pytest.mark.parametrize(
    "dt", [torch.tensor([[0.0, -1.0]]), torch.tensor([[0.0, float("nan")]])]
)
def test_invalid_intervals_rejected(dt: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        trajectory.differentiate_positions(torch.zeros(1, 2, 1), dt)


def test_resampling_preserves_time_profile_and_duration() -> None:
    # Spending 75% of time on the first half must survive resampling.
    q = torch.tensor([[[0.0], [1.0], [2.0]]])
    dt = torch.tensor([[0.0, 0.75, 0.25]])
    positions, intervals = trajectory.resample_in_time(q, dt, 5)
    torch.testing.assert_close(
        positions[0, :, 0], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0, 2.0])
    )
    torch.testing.assert_close(intervals, torch.tensor([[0.0, 0.25, 0.25, 0.25, 0.25]]))


def test_resampling_handles_repeated_time_and_zero_duration() -> None:
    q = torch.tensor([[[0.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]])
    dt = torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    positions, intervals = trajectory.resample_in_time(q, dt, 4)
    torch.testing.assert_close(positions[0, :, 0], torch.linspace(0, 1, 4))
    assert torch.equal(positions[1], torch.full((4, 1), 2.0))
    torch.testing.assert_close(intervals.sum(dim=1), dt.sum(dim=1))
