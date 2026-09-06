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

"""Observed body/link contacts for GenSim articulation cleanup."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch

from embodichain.lab.sim.sensors.contact_sensor import (
    ArticulationContactFilterCfg,
    ContactSensorCfg,
)

from .robot_parts import arm_control_part

__all__: list[str] = []

_SENSOR_UID = "gen_sim_interaction_contacts"
_CONTACT_CAPACITY = 512


def _filter_signature(cfg: ContactSensorCfg) -> tuple[Any, ...]:
    return (
        tuple(sorted(set(cfg.rigid_uid_list))),
        tuple(
            sorted(
                (item.articulation_uid, tuple(sorted(set(item.link_name_list))))
                for item in cfg.articulation_cfg_list
            )
        ),
        cfg.filter_need_both_actor,
        cfg.max_contacts_per_env,
    )


class _InteractionContacts:
    """Read fresh contact buffers without commanding or stepping the simulator.

    All contact classification is body/link based, never handle-submesh based.
    Missing or saturated observations are unknown, not proof of release. The
    caller owns physics ticks and must pass a monotonic tick for each sample.
    """

    def __init__(self, env: Any, target_uids: Iterable[str]) -> None:
        self.env = env
        self.device = torch.device(env.device)
        self.num_envs = int(env.num_envs)
        if isinstance(target_uids, str):
            raise TypeError("Interaction contact targets must be an iterable of UIDs.")
        self._targets = frozenset(target_uids)
        if not self._targets or any(
            not isinstance(uid, str) or not uid for uid in self._targets
        ):
            raise ValueError("Interaction contact targets must be non-empty UIDs.")
        sim = env.sim
        robot = env.robot
        self._robot_uid = str(robot.uid)
        articulations = {self._robot_uid: robot}
        for uid in dict.fromkeys(sim.get_articulation_uid_list()):
            if uid not in articulations:
                entity = sim.get_articulation(uid)
                if entity is None:
                    raise ValueError(f"Contact articulation {uid!r} is unavailable.")
                articulations[str(uid)] = entity
        if not self._targets.issubset(set(articulations) - {self._robot_uid}):
            raise ValueError("Interaction contact target articulation is unavailable.")
        self._target_links = {
            uid: frozenset(articulations[uid].link_names) for uid in self._targets
        }
        rigid_uids = list(dict.fromkeys(sim.get_rigid_object_uid_list()))
        if set(rigid_uids).intersection(articulations):
            raise ValueError("Contact entity UIDs must be unambiguous.")

        self._hands: dict[str, set[str]] = {}
        chains = {link: robot.get_parent_joint_chain(link) for link in robot.link_names}
        for arm in ("left_arm", "right_arm"):
            solver = robot.get_solver(name=arm_control_part(env, arm))
            mount = getattr(getattr(solver, "cfg", None), "end_link_name", None)
            if mount not in robot.link_names:
                raise ValueError(f"Contact observation requires a mount for {arm}.")
            self._hands[arm] = {
                link
                for link in robot.link_names
                if link == mount
                or any(joint.parent_link_name == mount for joint in chains[link])
            }
        if self._hands["left_arm"].intersection(self._hands["right_arm"]):
            raise ValueError("Contact hand link sets must be disjoint.")

        self._labels: list[dict[int, dict[str, Any]]] = [
            {} for _ in range(self.num_envs)
        ]
        registered_ids: set[int] = set()

        def register(uid: str, link: str | None, ids: torch.Tensor) -> None:
            values = torch.as_tensor(ids).detach().cpu().reshape(-1)
            if (
                values.dtype not in {torch.int32, torch.int64}
                or values.shape != (self.num_envs,)
                or bool((values < 0).any())
            ):
                raise ValueError(
                    "Contact user IDs must be batch-aligned non-negative integers."
                )
            for env_id, user_id in enumerate(values.tolist()):
                if user_id in registered_ids:
                    raise ValueError(
                        "Contact user IDs must be unique across bodies and environments."
                    )
                registered_ids.add(user_id)
                self._labels[env_id][user_id] = {
                    "user_id": user_id,
                    "entity_uid": uid,
                    "link_name": link,
                }

        filters = []
        for uid, entity in articulations.items():
            links = list(dict.fromkeys(entity.link_names))
            if not links:
                raise ValueError(f"Contact articulation {uid!r} has no links.")
            filters.append(
                ArticulationContactFilterCfg(articulation_uid=uid, link_name_list=links)
            )
            for link in links:
                register(uid, link, entity.get_user_ids(link))
        for uid in rigid_uids:
            entity = sim.get_rigid_object(uid)
            if entity is None:
                raise ValueError(f"Contact rigid object {uid!r} is unavailable.")
            register(str(uid), None, entity.get_user_ids())

        cfg = ContactSensorCfg(
            uid=_SENSOR_UID,
            rigid_uid_list=rigid_uids,
            articulation_cfg_list=filters,
            filter_need_both_actor=True,
            max_contacts_per_env=_CONTACT_CAPACITY,
        )
        if _SENSOR_UID in sim.get_sensor_uid_list():
            sensor = sim.get_sensor(_SENSOR_UID)
            if sensor is None or _filter_signature(sensor.cfg) != _filter_signature(
                cfg
            ):
                raise ValueError(
                    "Existing GenSim contact sensor has incompatible filters."
                )
        else:
            sensor = sim.add_sensor(cfg)
        if sensor is None:
            raise ValueError("GenSim contact sensor could not be registered.")
        sensor_ids = torch.as_tensor(sensor.item_user_ids).detach().cpu().reshape(-1)
        sensor_envs = torch.as_tensor(sensor.item_env_ids).detach().cpu().reshape(-1)
        expected_envs = {
            user_id: env_id
            for env_id, labels in enumerate(self._labels)
            for user_id in labels
        }
        if (
            sensor_ids.shape != sensor_envs.shape
            or dict(zip(sensor_ids.tolist(), sensor_envs.tolist())) != expected_envs
        ):
            raise ValueError(
                "GenSim contact sensor user IDs do not match the live scene."
            )
        self._sensor = sensor
        self._tick: float | None = None
        self._seen: dict[tuple[str, str], float] = {}
        self._data: dict[str, torch.Tensor] | None = None

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "known": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "target_contact": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "obstacle_contact": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "robot_world_contact": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "non_target_robot_world_contact": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "pairs": [[] for _ in range(self.num_envs)],
            "reason": [reason for _ in range(self.num_envs)],
        }

    def sample(
        self,
        arm: str,
        target_uid: str,
        *,
        tick: float,
        target_link: str | None = None,
    ) -> dict[str, Any]:
        """Observe a target link, or the whole articulation when omitted.

        Changing the link binding does not make the same physics tick fresh.
        Only the selected hand's contact with the bound target is permitted;
        other robot-world contacts remain visible even alongside a valid grasp.
        """
        if arm not in self._hands or target_uid not in self._targets:
            raise ValueError("Contact sample requires a registered arm and target.")
        if target_link is not None and (
            not isinstance(target_link, str)
            or target_link not in self._target_links[target_uid]
        ):
            raise ValueError(
                "Contact sample target link must exist on its articulation."
            )
        if not math.isfinite(tick):
            return self._empty("nonfinite_tick")
        key = (arm, target_uid)
        if tick <= self._seen.get(key, -math.inf) or (
            self._tick is not None and tick < self._tick
        ):
            return self._empty("stale_tick")
        self._seen[key] = tick
        if self._tick is None or tick > self._tick:
            self._sensor.update()
            raw = self._sensor.get_data()
            expected = {
                "is_valid": (self.num_envs, _CONTACT_CAPACITY),
                "user_ids": (self.num_envs, _CONTACT_CAPACITY, 2),
                "distance": (self.num_envs, _CONTACT_CAPACITY),
                "impulse": (self.num_envs, _CONTACT_CAPACITY),
                "position": (self.num_envs, _CONTACT_CAPACITY, 3),
                "normal": (self.num_envs, _CONTACT_CAPACITY, 3),
            }
            self._tick = tick
            self._data = None
            if (
                all(
                    isinstance(raw.get(name), torch.Tensor)
                    and tuple(raw[name].shape) == shape
                    for name, shape in expected.items()
                )
                and raw["is_valid"].dtype == torch.bool
                and raw["user_ids"].dtype
                in {
                    torch.int32,
                    torch.int64,
                }
                and all(
                    raw[name].is_floating_point()
                    for name in ("distance", "impulse", "position", "normal")
                )
            ):
                self._data = {
                    name: raw[name].detach().cpu().clone() for name in expected
                }
        if self._data is None:
            return self._empty("invalid_buffer")
        data = self._data
        result = self._empty("observed")
        result["known"].fill_(True)
        hand = self._hands[arm]
        for env_id in range(self.num_envs):
            valid = torch.nonzero(data["is_valid"][env_id], as_tuple=False).flatten()
            if len(valid) >= _CONTACT_CAPACITY:
                result["known"][env_id] = False
                result["reason"][env_id] = "saturated_buffer"
            for index in valid.tolist():
                ids = data["user_ids"][env_id, index].tolist()
                bodies = [self._labels[env_id].get(user_id) for user_id in ids]
                if any(body is None for body in bodies):
                    result["known"][env_id] = False
                    result["reason"][env_id] = "unknown_body"
                    continue
                if not all(
                    bool(torch.isfinite(data[name][env_id, index]).all())
                    for name in ("distance", "impulse", "position", "normal")
                ):
                    result["known"][env_id] = False
                    result["reason"][env_id] = "nonfinite_contact"
                    continue
                robots = [body["entity_uid"] == self._robot_uid for body in bodies]
                selected = [
                    is_robot and body["link_name"] in hand
                    for is_robot, body in zip(robots, bodies, strict=True)
                ]
                if all(selected):
                    continue
                target = any(selected) and any(
                    body["entity_uid"] == target_uid
                    and (target_link is None or body["link_name"] == target_link)
                    for body in bodies
                )
                obstacle = any(selected) and not target
                robot_world = any(robots) and not all(robots)
                non_target_robot_world = robot_world and not target
                if not (target or obstacle or robot_world):
                    continue
                result["target_contact"][env_id] |= target
                result["obstacle_contact"][env_id] |= obstacle
                result["robot_world_contact"][env_id] |= robot_world
                result["non_target_robot_world_contact"][
                    env_id
                ] |= non_target_robot_world
                result["pairs"][env_id].append(
                    {
                        "bodies": [dict(body) for body in bodies],
                        "target_contact": target,
                        "obstacle_contact": obstacle,
                        "robot_world_contact": robot_world,
                        "non_target_robot_world_contact": non_target_robot_world,
                        **{
                            name: data[name][env_id, index].tolist()
                            for name in ("distance", "impulse", "position", "normal")
                        },
                    }
                )
        return result
