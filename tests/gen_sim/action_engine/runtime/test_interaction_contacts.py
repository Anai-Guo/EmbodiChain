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

"""CPU-only contracts for observed articulation release contacts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.interaction_contacts import (
    _InteractionContacts,
)


class _Entity:
    def __init__(self, uid: str, ids: dict[str, int], batch_size: int = 2) -> None:
        self.uid = uid
        self.link_names = list(ids)
        self.ids = ids
        self.batch_size = batch_size

    def get_user_ids(self, link_name: str | None = None) -> torch.Tensor:
        base = (
            self.ids[link_name]
            if link_name is not None
            else next(iter(self.ids.values()))
        )
        return torch.arange(self.batch_size) * 100 + base


class _Robot(_Entity):
    def __init__(self) -> None:
        super().__init__(
            "robot",
            {
                "base": 1,
                "left_arm_link": 2,
                "left_mount": 3,
                "left_palm": 4,
                "left_finger": 5,
                "left_fixed_pad": 6,
                "right_arm_link": 12,
                "right_mount": 13,
                "right_palm": 14,
                "right_finger": 15,
                "right_fixed_pad": 16,
            },
        )

    def get_solver(self, *, name: str) -> SimpleNamespace:
        assert name in {"physical_left_arm", "physical_right_arm"}
        return SimpleNamespace(
            cfg=SimpleNamespace(
                end_link_name=name.removeprefix("physical_").replace("arm", "mount")
            )
        )

    def get_parent_joint_chain(self, link: str) -> tuple[SimpleNamespace, ...]:
        if link == "base":
            return ()
        side, suffix = link.split("_", 1)
        if suffix in {"arm_link", "mount"}:
            return (SimpleNamespace(parent_link_name="base", child_link_name=link),)
        return (
            SimpleNamespace(parent_link_name=f"{side}_palm", child_link_name=link),
            SimpleNamespace(
                parent_link_name=f"{side}_mount", child_link_name=f"{side}_palm"
            ),
        )

    def set_qpos(self, *_args, **_kwargs) -> None:
        pytest.fail("Contact observation must not command robot state.")


class _Sensor:
    def __init__(self, cfg, batch_size: int = 2) -> None:
        self.cfg = cfg
        self.updates = 0
        size = (batch_size, cfg.max_contacts_per_env)
        self.data = {
            "is_valid": torch.zeros(size, dtype=torch.bool),
            "user_ids": torch.zeros((*size, 2), dtype=torch.int32),
            "distance": torch.zeros(size),
            "impulse": torch.zeros(size),
            "position": torch.zeros((*size, 3)),
            "normal": torch.zeros((*size, 3)),
        }

    def update(self) -> None:
        self.updates += 1

    def get_data(self):
        return self.data

    def contact(self, row: int, slot: int, bodies: tuple[int, int], **fields) -> None:
        self.data["is_valid"][row, slot] = True
        self.data["user_ids"][row, slot] = torch.tensor(bodies)
        for key, value in fields.items():
            self.data[key][row, slot] = torch.as_tensor(value)


class _Sim:
    def __init__(self, robot: _Robot) -> None:
        self.robot = robot
        self.rigids = {
            uid: _Entity(uid, {"": body_id})
            for uid, body_id in (("table", 40), ("book", 41))
        }
        self.articulations = {
            "drawer": _Entity("drawer", {"cabinet": 30, "drawer_link": 31}),
            "other_articulation": _Entity(
                "other_articulation", {"base": 50, "door": 51}
            ),
        }
        self.sensors = {}
        self.created = []

    def get_rigid_object_uid_list(self):
        return ["table", "book", "table"]

    def get_rigid_object(self, uid):
        return self.rigids.get(uid)

    def get_articulation_uid_list(self):
        return ["robot", "drawer", "drawer", "other_articulation"]

    def get_articulation(self, uid):
        return self.articulations.get(uid)

    def get_sensor_uid_list(self):
        return list(self.sensors)

    def get_sensor(self, uid):
        return self.sensors.get(uid)

    def add_sensor(self, cfg):
        assert cfg.uid not in self.sensors
        sensor = _Sensor(cfg)
        sensor.item_user_ids = torch.cat(
            [self.rigids[uid].get_user_ids() for uid in cfg.rigid_uid_list]
            + [
                (
                    self.robot
                    if entry.articulation_uid == "robot"
                    else self.articulations[entry.articulation_uid]
                ).get_user_ids(link)
                for entry in cfg.articulation_cfg_list
                for link in entry.link_name_list
            ]
        )
        sensor.item_env_ids = torch.arange(2).repeat(sensor.item_user_ids.numel() // 2)
        self.sensors[cfg.uid] = sensor
        self.created.append(cfg)
        return sensor

    def update(self, *_args, **_kwargs) -> None:
        pytest.fail("Contact observation must not advance simulation.")


@pytest.fixture
def env():
    robot = _Robot()
    return SimpleNamespace(
        robot=robot,
        sim=_Sim(robot),
        num_envs=2,
        device="cpu",
        get_agent_arm_control_part=lambda is_left: (
            "physical_left_arm" if is_left else "physical_right_arm"
        ),
        step=lambda *_args: pytest.fail(
            "Contact observer must not step the environment."
        ),
    )


def _observer(env):
    observer = _InteractionContacts(env, ["drawer"])
    return observer, next(iter(env.sim.sensors.values()))


def test_sensor_registration_deduplicates_entities_and_reuses_owned_sensor(env):
    observer, sensor = _observer(env)
    other = _InteractionContacts(env, ["drawer"])

    assert observer is not other
    assert len(env.sim.created) == 1
    cfg = env.sim.created[0]
    assert cfg.filter_need_both_actor is True
    assert set(cfg.rigid_uid_list) == {"table", "book"}
    assert len(cfg.rigid_uid_list) == 2
    entries = {
        item.articulation_uid: item.link_name_list for item in cfg.articulation_cfg_list
    }
    assert len(entries) == len(cfg.articulation_cfg_list) == 3
    assert entries["robot"] == env.robot.link_names
    assert set(entries["drawer"]) == {"cabinet", "drawer_link"}
    assert sensor.updates == 0


@pytest.mark.parametrize("pair", [(5, 31), (31, 5), (6, 30)])
def test_target_contacts_cover_all_target_links_and_either_actor_order(env, pair):
    observer, sensor = _observer(env)
    sensor.contact(
        0, 0, pair, distance=0.0002, impulse=0.03, position=[1, 2, 3], normal=[0, 0, 1]
    )

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["known"].tolist() == [True, True]
    assert result["target_contact"].tolist() == [True, False]
    assert result["obstacle_contact"].tolist() == [False, False]
    assert result["non_target_robot_world_contact"].tolist() == [False, False]
    entry = result["pairs"][0][0]
    assert entry["distance"] == pytest.approx(0.0002)
    assert entry["impulse"] == pytest.approx(0.03)
    assert entry["position"] == [1.0, 2.0, 3.0]
    assert {body["entity_uid"] for body in entry["bodies"]} == {"robot", "drawer"}
    assert all("link_name" in body and "user_id" in body for body in entry["bodies"])


@pytest.mark.parametrize("other", [40, 41, 51, 15, 2])
def test_same_hand_self_contact_is_ignored_but_world_and_other_arm_are_obstacles(
    env, other
):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 6), impulse=0.1)
    sensor.contact(1, 0, (105, other + 100), impulse=0.1)

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["known"].tolist() == [True, True]
    assert result["target_contact"].tolist() == [False, False]
    assert result["obstacle_contact"].tolist() == [False, True]
    assert result["pairs"][0] == []


def test_other_robot_links_contact_world_without_claiming_selected_hand_contact(env):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (2, 40), distance=-0.001, impulse=0.1)

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["robot_world_contact"].tolist() == [True, False]
    assert result["obstacle_contact"].tolist() == [False, False]
    assert result["target_contact"].tolist() == [False, False]


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_target_link_separates_moving_handle_link_from_parent_cabinet(
    env, reverse
):
    observer, sensor = _observer(env)
    pairs = [(5, 31), (105, 130)]
    for row, pair in enumerate(pairs):
        sensor.contact(row, 0, tuple(reversed(pair)) if reverse else pair)

    result = observer.sample("left_arm", "drawer", tick=1.0, target_link="drawer_link")

    assert result["known"].tolist() == [True, True]
    assert result["target_contact"].tolist() == [True, False]
    assert result["obstacle_contact"].tolist() == [False, True]
    assert result["robot_world_contact"].tolist() == [True, True]
    assert result["non_target_robot_world_contact"].tolist() == [False, True]
    assert result["pairs"][0][0]["non_target_robot_world_contact"] is False
    assert result["pairs"][1][0]["non_target_robot_world_contact"] is True


@pytest.mark.parametrize("target_link", [None, "drawer_link"])
def test_mixed_allowed_grasp_and_other_arm_world_contact_cannot_cancel_obstacle_flag(
    env, target_link
):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 31))
    sensor.contact(0, 1, (12, 40))
    sensor.contact(1, 0, (105, 131))

    result = observer.sample("left_arm", "drawer", tick=1.0, target_link=target_link)

    assert result["known"].tolist() == [True, True]
    assert result["target_contact"].tolist() == [True, True]
    assert result["obstacle_contact"].tolist() == [False, False]
    assert result["non_target_robot_world_contact"].tolist() == [True, False]


def test_switching_target_link_does_not_create_a_second_fresh_sample_at_same_tick(env):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 30))
    first = observer.sample("left_arm", "drawer", tick=1.0)
    stale = observer.sample("left_arm", "drawer", tick=1.0, target_link="drawer_link")
    fresh = observer.sample("left_arm", "drawer", tick=2.0, target_link="drawer_link")

    assert first["target_contact"].tolist() == [True, False]
    assert not stale["known"].any()
    assert stale["non_target_robot_world_contact"].tolist() == [False, False]
    assert fresh["target_contact"].tolist() == [False, False]
    assert fresh["obstacle_contact"].tolist() == [True, False]
    assert sensor.updates == 2


def test_unknown_target_link_fails_instead_of_claiming_contact_free(env):
    observer, sensor = _observer(env)

    with pytest.raises(ValueError, match="target link"):
        observer.sample("left_arm", "drawer", tick=1.0, target_link="missing_handle")
    assert sensor.updates == 0


def test_repeated_or_older_tick_is_unknown_while_other_arm_reuses_same_snapshot(env):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (15, 31))
    first = observer.sample("left_arm", "drawer", tick=1.0)
    repeated = observer.sample("left_arm", "drawer", tick=1.0)
    older = observer.sample("left_arm", "drawer", tick=0.5)
    right = observer.sample("right_arm", "drawer", tick=1.0)

    assert first["known"].all()
    assert not repeated["known"].any()
    assert not older["known"].any()
    assert right["known"].all()
    assert right["target_contact"].tolist() == [True, False]
    assert sensor.updates == 1
    assert observer.sample("left_arm", "drawer", tick=2.0)["known"].all()
    assert sensor.updates == 2


def test_unknown_user_id_and_cross_environment_id_never_become_contact_free(env):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 999))
    sensor.contact(1, 0, (105, 31))

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["known"].tolist() == [False, False]


@pytest.mark.parametrize("field", ["distance", "impulse", "position", "normal"])
def test_nonfinite_valid_contact_is_unknown_only_in_its_environment(env, field):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 31))
    sensor.data[field][0, 0] = float("nan")
    sensor.data[field][1, 1] = float("nan")

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["known"].tolist() == [False, True]


def test_saturated_contact_buffer_is_unknown_and_masks_are_row_local(env):
    observer, sensor = _observer(env)
    for slot in range(sensor.cfg.max_contacts_per_env):
        sensor.contact(0, slot, (5, 40))

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert result["known"].tolist() == [False, True]
    assert result["obstacle_contact"].tolist() == [True, False]


def test_returned_snapshot_does_not_alias_sensor_buffers_or_other_samples(env):
    observer, sensor = _observer(env)
    sensor.contact(0, 0, (5, 31), impulse=0.1)
    first = observer.sample("left_arm", "drawer", tick=1.0)
    sensor.data["is_valid"].zero_()
    sensor.data["impulse"].zero_()
    observer.sample("right_arm", "drawer", tick=1.0)
    first["known"].zero_()

    again = observer.sample("left_arm", "drawer", tick=2.0)

    assert first["pairs"][0][0]["impulse"] == pytest.approx(0.1)
    assert first["target_contact"].tolist() == [True, False]
    assert again["known"].all()
    assert not again["target_contact"].any()


def test_missing_target_registration_fails_before_constructing_sensor(env):
    with pytest.raises(ValueError, match="target"):
        _InteractionContacts(env, ["missing"])
    assert env.sim.created == []


def test_reused_sensor_cannot_silently_filter_out_recreated_body_ids(env):
    _observer(env)
    env.sim.articulations["drawer"].ids["drawer_link"] = 32

    with pytest.raises(ValueError, match="user IDs"):
        _InteractionContacts(env, ["drawer"])


def test_reused_sensor_cannot_mix_rows_when_user_ids_move_between_environments(
    env, monkeypatch
):
    _observer(env)
    entity = env.sim.articulations["drawer"]
    original = entity.get_user_ids
    monkeypatch.setattr(entity, "get_user_ids", lambda link: original(link).flip(0))

    with pytest.raises(ValueError, match="user IDs"):
        _InteractionContacts(env, ["drawer"])


def test_duplicate_body_ids_across_environments_are_rejected(env):
    env.sim.articulations["drawer"].ids["drawer_link"] = 105

    with pytest.raises(ValueError, match="unique"):
        _InteractionContacts(env, ["drawer"])


def test_malformed_buffer_is_unknown_without_counting_it_as_empty_contacts(env):
    observer, sensor = _observer(env)
    sensor.data["user_ids"] = sensor.data["user_ids"][:, :1]

    result = observer.sample("left_arm", "drawer", tick=1.0)

    assert not result["known"].any()


@pytest.mark.parametrize("tick", [float("nan"), float("inf")])
def test_nonfinite_tick_never_produces_a_fresh_contact_free_sample(env, tick):
    observer, sensor = _observer(env)

    result = observer.sample("left_arm", "drawer", tick=tick)

    assert not result["known"].any()
    assert sensor.updates == 0
