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

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import interaction_physics
from embodichain.gen_sim.action_engine.runtime.interaction_physics import (
    _restore_gravity_contract,
)


class _Articulation:
    def __init__(self, flags, configured=False) -> None:
        self.cfg = SimpleNamespace(fpath="generated.usdc", enable_gravity=configured)
        self.link_names = ["base", "moving_link"]
        self.flags = [list(row) for row in flags]
        self.calls = []

    def get_qpos(self):
        return torch.zeros(len(self.flags), 1)

    def get_link_physical_attr(self, link_names, env_ids):
        self.calls.append(("read", list(link_names), list(env_ids)))
        return [
            SimpleNamespace(has_gravity=self.flags[env][self.link_names.index(link)])
            for env in env_ids
            for link in link_names
        ]

    def set_gravity(self, enabled, env_ids):
        self.calls.append(("set_gravity", enabled, list(env_ids)))
        for env in env_ids:
            self.flags[env] = [enabled] * len(self.link_names)

    def set_qpos(self, *_args, **_kwargs):
        pytest.fail("Gravity contract restoration must not write joint positions.")


@pytest.mark.parametrize("configured", [False, True])
def test_gravity_contract_restores_only_mismatched_environments(configured) -> None:
    articulation = _Articulation(
        [[configured, configured], [configured, not configured]], configured
    )

    evidence = _restore_gravity_contract(articulation)

    assert evidence == {
        "configured": configured,
        "before": [[configured, configured], [configured, not configured]],
        "after": [[configured, configured], [configured, configured]],
        "changed": True,
        "env_ids": [1],
        "link_names": ["base", "moving_link"],
        "direct_qpos_write": False,
    }
    assert articulation.calls == [
        ("read", articulation.link_names, [0, 1]),
        ("set_gravity", configured, [1]),
        ("read", articulation.link_names, [0, 1]),
    ]


def test_gravity_contract_is_idempotent_when_config_already_matches() -> None:
    articulation = _Articulation([[False, False]])
    evidence = _restore_gravity_contract(articulation)
    assert evidence["changed"] is False
    assert evidence["env_ids"] == []
    assert evidence["before"] == evidence["after"] == [[False, False]]
    assert all(call[0] == "read" for call in articulation.calls)


@pytest.mark.parametrize("suffix", [".usd", ".usda", ".usdc", ".USD"])
def test_gravity_contract_accepts_usd_suffixes(suffix) -> None:
    articulation = _Articulation([[True, True]])
    articulation.cfg.fpath = "generated" + suffix
    assert _restore_gravity_contract(articulation)["changed"] is True


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        SimpleNamespace(fpath="robot.urdf", enable_gravity=False),
        SimpleNamespace(fpath="generated.usdc"),
    ],
)
def test_gravity_contract_skips_unsupported_or_unspecified_configs(cfg) -> None:
    articulation = SimpleNamespace(cfg=cfg)
    assert _restore_gravity_contract(articulation) is None


@pytest.mark.parametrize("configured", [None, 0, 1, "false"])
def test_gravity_contract_rejects_non_boolean_configuration(configured) -> None:
    articulation = _Articulation([[True, True]], configured)
    with pytest.raises(TypeError, match="enable_gravity"):
        _restore_gravity_contract(articulation)
    assert articulation.calls == []


def test_gravity_contract_rejects_incomplete_attribute_batch_before_writing(
    monkeypatch,
) -> None:
    articulation = _Articulation([[True, True], [True, True]])
    monkeypatch.setattr(
        articulation,
        "get_link_physical_attr",
        lambda **_kwargs: [SimpleNamespace(has_gravity=True)],
    )
    with pytest.raises(ValueError, match="attribute count"):
        _restore_gravity_contract(articulation)
    assert articulation.calls == []


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_gravity_contract_rejects_untyped_observed_flags(flag) -> None:
    articulation = _Articulation([[False, flag]])
    with pytest.raises(TypeError, match="has_gravity"):
        _restore_gravity_contract(articulation)
    assert all(call[0] == "read" for call in articulation.calls)


def test_gravity_contract_fails_closed_if_backend_does_not_apply_flag(
    monkeypatch,
) -> None:
    articulation = _Articulation([[True, True]])
    monkeypatch.setattr(articulation, "set_gravity", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="gravity contract"):
        _restore_gravity_contract(articulation)


def test_gravity_contract_does_not_reuse_mutable_before_snapshot() -> None:
    articulation = _Articulation([[True, True]])
    evidence = _restore_gravity_contract(articulation)
    articulation.flags[0][0] = True
    assert evidence["before"] == [[True, True]]
    assert evidence["after"] == [[False, False]]


def test_gravity_contract_restores_only_selected_reset_rows() -> None:
    articulation = _Articulation([[True, True], [True, True], [False, False]])
    evidence = _restore_gravity_contract(articulation, env_ids=torch.tensor([2, 1]))
    assert articulation.flags == [[True, True], [False, False], [False, False]]
    assert evidence["before"] == [[False, False], [True, True]]
    assert evidence["after"] == [[False, False], [False, False]]
    assert evidence["env_ids"] == [1]
    assert articulation.calls[0] == ("read", articulation.link_names, [2, 1])


@pytest.mark.parametrize("env_ids", [[True], [0.0], [[0]], [0, 0], [-1], [2]])
def test_gravity_contract_rejects_invalid_reset_rows_without_writing(env_ids) -> None:
    articulation = _Articulation([[True, True], [True, True]])
    with pytest.raises(ValueError, match="env_ids"):
        _restore_gravity_contract(articulation, env_ids=env_ids)
    assert articulation.calls == []


def test_gravity_contract_empty_reset_does_not_touch_other_rows() -> None:
    articulation = _Articulation([[True, True]])
    evidence = _restore_gravity_contract(articulation, env_ids=[])
    assert evidence["before"] == evidence["after"] == evidence["env_ids"] == []
    assert evidence["changed"] is False
    assert articulation.calls == []


def test_generated_reset_restores_all_gravity_flags_before_native_physics() -> None:
    from embodichain.gen_sim.action_engine.environment.agent_env import ActionEngineEnv

    targets = {
        "door": _Articulation([[True, True], [True, True]]),
        "drawer": _Articulation([[True, True], [True, True]]),
    }
    resets = []

    def native_reset(name, env_ids):
        assert list(env_ids) == [1]
        # A native articulation reset advances the whole physics world.
        assert all(not any(target.flags[1]) for target in targets.values())
        resets.append(name)

    for name, target in targets.items():
        target.reset = lambda env_ids, name=name: native_reset(name, env_ids)
    unrelated = SimpleNamespace(cfg=SimpleNamespace(fpath="robot.urdf"))
    assets = {**targets, "unrelated": unrelated}
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_articulation_uid_list=lambda: list(assets),
            get_articulation=assets.get,
        )
    )
    ActionEngineEnv._restore_generated_usd_articulation_reset_state(env, [1])
    assert resets == ["door", "drawer"]
    assert all(target.flags[0] == [True, True] for target in targets.values())
    assert set(env._generated_usd_reset_gravity) == set(targets)
    assert all(value["changed"] for value in env._generated_usd_reset_gravity.values())


@pytest.fixture
def joint_motion_inputs() -> dict[str, Any]:
    return {
        "initial": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "target": torch.tensor([-0.08, 1.38], dtype=torch.float64),
        "observed": torch.tensor([-0.079, 1.379], dtype=torch.float64),
        "executed": torch.tensor([True, True]),
        "tolerance": 0.005,
    }


def test_joint_motion_evidence_distinguishes_transition_direction_and_execution() -> (
    None
):
    evidence = interaction_physics._joint_motion_evidence(
        initial=torch.tensor(
            [0.0, 0.0, 0.998, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
        ),
        target=torch.tensor(
            [-0.08, 1.38, 1.0, -0.08, -0.08, -0.08, -0.08], dtype=torch.float64
        ),
        observed=torch.tensor(
            [-0.079, 1.379, 1.0, 0.0, 0.03, -0.05, -0.08], dtype=torch.float64
        ),
        executed=torch.tensor([True, True, True, True, True, True, False]),
        tolerance=0.005,
    )

    assert evidence["phase"] == "core_terminal_before_release"
    assert evidence["success"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert evidence["initial_satisfied"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert evidence["direction_ok"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert evidence["finite"].tolist() == [True] * 7
    assert evidence["executed"].tolist() == [True] * 6 + [False]
    assert evidence["delta"] == pytest.approx(
        [-0.079, 1.379, 0.002, 0.0, 0.03, -0.05, -0.08]
    )
    assert evidence["target_error"] == pytest.approx(
        [0.001, 0.001, 0.0, 0.08, 0.11, 0.03, 0.0]
    )
    assert evidence["tolerance"] == 0.005


def test_joint_motion_evidence_target_tolerance_boundary_is_inclusive() -> None:
    # Binary-exact tolerance avoids an ambiguous floating-point equality boundary.
    tolerance = 0.125
    evidence = interaction_physics._joint_motion_evidence(
        initial=torch.zeros(4, dtype=torch.float64),
        target=torch.ones(4, dtype=torch.float64),
        observed=torch.tensor([0.875, 1.125, 0.874999, 1.125001], dtype=torch.float64),
        executed=torch.ones(4, dtype=torch.bool),
        tolerance=tolerance,
    )
    assert evidence["success"].tolist() == [True, True, False, False]


@pytest.mark.parametrize("field", ["initial", "target", "observed"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_joint_motion_evidence_invalid_values_fail_closed_and_serialize_as_null(
    joint_motion_inputs: dict[str, Any], field: str, value: float
) -> None:
    from embodichain.gen_sim.action_engine.runtime.recording import _jsonable

    joint_motion_inputs[field][0] = value
    evidence = interaction_physics._joint_motion_evidence(**joint_motion_inputs)

    assert evidence["finite"].tolist() == [False, True]
    assert evidence["success"].tolist() == [False, True]
    assert evidence[field][0] is None
    for other in {"initial", "target", "observed"} - {field}:
        assert evidence[other] == joint_motion_inputs[other].tolist()
    serialized = json.dumps(_jsonable(evidence), allow_nan=False)
    assert json.loads(serialized)[field][0] is None


@pytest.mark.parametrize("tolerance", [0.0, -0.005, float("nan"), float("inf")])
def test_joint_motion_evidence_rejects_invalid_tolerance(
    joint_motion_inputs: dict[str, Any], tolerance: float
) -> None:
    joint_motion_inputs["tolerance"] = tolerance
    with pytest.raises(ValueError, match="tolerance"):
        interaction_physics._joint_motion_evidence(**joint_motion_inputs)


@pytest.mark.parametrize("tolerance", [True, "0.005"])
def test_joint_motion_evidence_rejects_non_numeric_tolerance(
    joint_motion_inputs: dict[str, Any], tolerance: Any
) -> None:
    joint_motion_inputs["tolerance"] = tolerance
    with pytest.raises(TypeError, match="tolerance"):
        interaction_physics._joint_motion_evidence(**joint_motion_inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial", torch.tensor(0.0)),
        ("target", torch.zeros(2, 1)),
        ("observed", torch.zeros(1)),
        ("executed", torch.ones(2, 1, dtype=torch.bool)),
        ("initial", torch.empty(0)),
    ],
)
def test_joint_motion_evidence_rejects_implicit_broadcasting(
    joint_motion_inputs: dict[str, Any], field: str, value: torch.Tensor
) -> None:
    joint_motion_inputs[field] = value
    with pytest.raises(ValueError, match="shape"):
        interaction_physics._joint_motion_evidence(**joint_motion_inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial", [0.0, 0.0]),
        ("target", torch.tensor([0, 1])),
        ("executed", torch.ones(2)),
    ],
)
def test_joint_motion_evidence_requires_typed_numeric_and_boolean_tensors(
    joint_motion_inputs: dict[str, Any], field: str, value: Any
) -> None:
    joint_motion_inputs[field] = value
    with pytest.raises(TypeError, match=field):
        interaction_physics._joint_motion_evidence(**joint_motion_inputs)


def test_joint_motion_evidence_is_an_independent_detached_snapshot(
    joint_motion_inputs: dict[str, Any],
) -> None:
    for name in ("initial", "target", "observed"):
        joint_motion_inputs[name].requires_grad_()
    evidence = interaction_physics._joint_motion_evidence(**joint_motion_inputs)
    with torch.no_grad():
        for name in ("initial", "target", "observed"):
            joint_motion_inputs[name].fill_(3.0)
    joint_motion_inputs["executed"].fill_(False)

    assert evidence["initial"] == [0.0, 0.0]
    assert evidence["target"] == [-0.08, 1.38]
    assert evidence["observed"] == [-0.079, 1.379]
    assert evidence["delta"] == pytest.approx([-0.079, 1.379])
    for name in ("finite", "initial_satisfied", "direction_ok", "executed", "success"):
        assert evidence[name].shape == (2,)
        assert evidence[name].dtype == torch.bool
        assert not evidence[name].requires_grad
    assert evidence["executed"].tolist() == [True, True]
    assert evidence["success"].tolist() == [True, True]
