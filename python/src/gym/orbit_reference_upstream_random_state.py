"""Deterministic rollout of Kaggle upstream Orbit RNG without stepping fleet physics.

Uses ``generate_planets`` / ``generate_comet_paths`` from
``python/cpp/orbit_wars/reference_kaggle_upstream_github_no_edit/orbit_wars.py``.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import random
from typing import Any

_VALID_NUM_AGENTS = (2, 4)

_REF_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "cpp"
    / "orbit_wars"
    / "reference_kaggle_upstream_github_no_edit"
    / "orbit_wars.py"
)


def _upstream_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "orbit_wars_reference_upstream_rng",
        _REF_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None, _REF_MODULE_PATH
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_UP = _upstream_module()


def orbit_reference_upstream_random_derived_dict(
    *,
    seed: int,
    num_agents: int,
    comet_speed: float = 4.0,
) -> dict[str, Any]:
    """Replay upstream Random usage for one episode: startup layout + comet spawns.

    Parameters match the reference interpreter: ``init_rng = random.Random(seed)``,
    home planets for ``num_agents``, then for each value ``s`` in ``COMET_SPAWN_STEPS``
    the comet RNG is ``random.Random(f"orbit_wars-comet-{seed}-{s}")`` with
    ``spawn_step=s`` passed to ``generate_comet_paths`` (same as ``step+1`` in the
    interpreter when ``step == s - 1``).

    Returns a dict with ``seed``, ``angular_velocity``, ``planets`` (after home
    assignment), ``initial_planets_start`` (deep copy before homes), ``comets`` (one
    entry per scheduled spawn),     ``comet_sync_updates`` (ordered payloads for Python
    ``orbit_cpp_env_apply_comet_sync_update_one``: ``episode_step`` must match ``cpp_env.episode_step()`` when applied,
    each payload includes ``planets`` (snapshot list) and ``comet_planet_ids``, ``comets_groups``
    with ``path_index`` only — comet ship counts are read from ``planets`` rows by id),
    and
    ``working_initial_planets_after_comets`` (snapshot
    after all successful spawns in order — mirrors append-only growth without expiry).
    """
    assert isinstance(seed, int), type(seed)
    na = int(num_agents)
    assert na in _VALID_NUM_AGENTS, na
    assert comet_speed > 0.0, comet_speed

    generate_planets = _UP.generate_planets
    generate_comet_paths = _UP.generate_comet_paths
    comet_spawn_steps = list(_UP.COMET_SPAWN_STEPS)
    comet_radius = float(_UP.COMET_RADIUS)
    comet_production = int(_UP.COMET_PRODUCTION)

    init_rng = random.Random(seed)
    angular_velocity = float(init_rng.uniform(0.025, 0.05))
    planets_raw = generate_planets(init_rng)
    initial_planets_start = [copy.deepcopy(p) for p in planets_raw]
    planets = copy.deepcopy(planets_raw)

    num_groups = len(planets) // 4
    if num_groups > 0:
        home_group = init_rng.randint(0, num_groups - 1)
        base = home_group * 4
        if na == 2:
            planets[base][1] = 0
            planets[base][5] = 10
            planets[base + 3][1] = 1
            planets[base + 3][5] = 10
        else:
            for j in range(4):
                planets[base + j][1] = j
                planets[base + j][5] = 10

    working_initial = copy.deepcopy(initial_planets_start)
    working_planets = copy.deepcopy(planets)
    comet_planet_ids: list[int] = []
    active_comet_groups: list[dict[str, Any]] = []

    comet_sync_updates: list[dict[str, Any]] = []
    comet_records: list[dict[str, Any]] = []
    for spawn_step in comet_spawn_steps:
        expired_comet_pids: list[int] = []
        for group in active_comet_groups:
            group_spawn_step = int(group["spawn_step"])
            path_index = int(spawn_step) - group_spawn_step - 1
            planet_ids = group["planet_ids"]
            paths = group["paths"]
            assert isinstance(planet_ids, list)
            assert isinstance(paths, list)
            for i, pid in enumerate(planet_ids):
                if path_index >= len(paths[i]):
                    expired_comet_pids.append(int(pid))
        if expired_comet_pids:
            expired = set(expired_comet_pids)
            working_planets = [p for p in working_planets if int(p[0]) not in expired]
            working_initial = [p for p in working_initial if int(p[0]) not in expired]
            comet_planet_ids = [pid for pid in comet_planet_ids if int(pid) not in expired]
            next_active: list[dict[str, Any]] = []
            for group in active_comet_groups:
                planet_ids = group["planet_ids"]
                paths = group["paths"]
                assert isinstance(planet_ids, list)
                assert isinstance(paths, list)
                kept_ids: list[int] = []
                kept_paths: list[Any] = []
                for pid, path in zip(planet_ids, paths, strict=True):
                    if int(pid) not in expired:
                        kept_ids.append(int(pid))
                        kept_paths.append(path)
                if kept_ids:
                    next_active.append(
                        {
                            "spawn_step": int(group["spawn_step"]),
                            "planet_ids": kept_ids,
                            "paths": kept_paths,
                            "ships": float(group.get("ships", 0.0)),
                        }
                    )
            active_comet_groups = next_active
        comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = generate_comet_paths(
            working_initial,
            angular_velocity,
            int(spawn_step),
            comet_planet_ids,
            float(comet_speed),
            rng=comet_rng,
        )
        rec: dict[str, Any] = {
            "spawn_step": int(spawn_step),
            "observation_step_before_spawn": int(spawn_step) - 1,
            "paths": paths,
            "comet_ships": None,
            "planet_ids": [],
        }
        if paths:
            next_id = max(int(p[0]) for p in working_planets) + 1
            comet_ships = int(
                min(
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                )
            )
            rec["comet_ships"] = comet_ships
            pids: list[int] = []
            for i, _p_path in enumerate(paths):
                pid = next_id + i
                pids.append(pid)
                planet = [
                    pid,
                    -1,
                    -99,
                    -99,
                    comet_radius,
                    comet_ships,
                    comet_production,
                ]
                working_planets.append(planet)
                working_initial.append(planet[:])
                comet_planet_ids.append(pid)
            rec["planet_ids"] = pids
            active_comet_groups.append(
                {
                    "spawn_step": int(spawn_step),
                    "planet_ids": list(pids),
                    "paths": paths,
                    "ships": float(comet_ships),
                }
            )
        comet_records.append(rec)
        groups_plain: list[dict[str, Any]] = []
        for g in active_comet_groups:
            assert isinstance(g["planet_ids"], list)
            assert isinstance(g["paths"], list)
            group_spawn_step = int(g["spawn_step"])
            path_index = int(spawn_step) - group_spawn_step - 1
            groups_plain.append(
                {
                    "planet_ids": [int(x) for x in g["planet_ids"]],
                    "paths": copy.deepcopy(g["paths"]),
                    "path_index": int(path_index),
                }
            )
        comet_sync_updates.append(
            {
                "episode_step": int(spawn_step) - 1,
                "comet_planet_ids": [int(x) for x in comet_planet_ids],
                "comets_groups": groups_plain,
                "planets": copy.deepcopy(working_planets),
            }
        )

    return {
        "seed": int(seed),
        "num_agents": na,
        "comet_speed": float(comet_speed),
        "angular_velocity": angular_velocity,
        "planets": planets,
        "initial_planets_start": initial_planets_start,
        "comet_spawn_steps": comet_spawn_steps,
        "comets": comet_records,
        "comet_sync_updates": comet_sync_updates,
        "working_initial_planets_after_comets": working_initial,
        "working_planets_after_comets": working_planets,
        "comet_planet_ids_after_comets": list(comet_planet_ids),
    }
