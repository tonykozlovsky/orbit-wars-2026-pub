"""Maps policy obs physical channels to logical feature names."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..gym.obs_wrapper import (
    ORBIT_EDGE_BASE_FEATURES,
    ORBIT_EDGE_FEATURES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_BASE_FEATURES,
    ORBIT_PLANET_FEATURES,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLAYER_AXIS_SLOTS,
)


def _stat_token_ok(name: str) -> None:
    assert isinstance(name, str) and len(name) > 0, name
    for ch in name:
        assert ch.isascii(), name
        assert ch.isalnum() or ch == "_", name


def _coerce_name_tuple(raw: Any, *, label: str) -> tuple[str, ...]:
    assert isinstance(raw, (list, tuple)), (label, type(raw))
    out: list[str] = []
    for i, x in enumerate(raw):
        assert isinstance(x, str), (label, i, type(x))
        _stat_token_ok(x)
        out.append(x)
    return tuple(out)


def planet_edge_physical_to_logical_from_layout(
    layout: Mapping[str, Any],
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor, tuple[str, ...]]:
    pb = _coerce_name_tuple(layout["planet_base_feature_names"], label="planet_base_feature_names")
    pr = _coerce_name_tuple(
        layout["planet_player_block_feature_names"],
        label="planet_player_block_feature_names",
    )
    eb = _coerce_name_tuple(layout["edge_base_feature_names"], label="edge_base_feature_names")
    er = _coerce_name_tuple(
        layout["edge_player_block_feature_names"],
        label="edge_player_block_feature_names",
    )
    assert len(pb) == ORBIT_PLANET_BASE_FEATURES, (len(pb), ORBIT_PLANET_BASE_FEATURES)
    assert len(pr) == ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER, (
        len(pr),
        ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    )
    assert len(eb) == ORBIT_EDGE_BASE_FEATURES, (len(eb), ORBIT_EDGE_BASE_FEATURES)
    assert len(er) == ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER, (
        len(er),
        ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    )

    planet_logical_names = pb + pr
    assert len(set(planet_logical_names)) == len(planet_logical_names), planet_logical_names

    edge_logical_names = eb + er
    assert len(set(edge_logical_names)) == len(edge_logical_names), edge_logical_names

    planet_phys: list[int] = []
    for c in range(ORBIT_PLANET_FEATURES):
        if c < ORBIT_PLANET_BASE_FEATURES:
            planet_phys.append(c)
        else:
            off = c - ORBIT_PLANET_PLAYER_FEATURE_OFFSET
            player_width = ORBIT_PLAYER_AXIS_SLOTS * ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
            assert 0 <= off < player_width, (c, off, player_width)
            k = off % ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER
            planet_phys.append(ORBIT_PLANET_BASE_FEATURES + k)

    edge_phys: list[int] = []
    for c in range(ORBIT_EDGE_FEATURES):
        if c < ORBIT_EDGE_BASE_FEATURES:
            edge_phys.append(c)
        else:
            off = c - ORBIT_EDGE_PLAYER_FEATURE_OFFSET
            player_width = ORBIT_PLAYER_AXIS_SLOTS * ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
            assert 0 <= off < player_width, (c, off, player_width)
            k = off % ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER
            edge_phys.append(ORBIT_EDGE_BASE_FEATURES + k)

    return (
        torch.tensor(planet_phys, dtype=torch.long),
        planet_logical_names,
        torch.tensor(edge_phys, dtype=torch.long),
        edge_logical_names,
    )


def logical_feature_normalization_keys(prefix: str, logical_names: tuple[str, ...]) -> tuple[str, ...]:
    _stat_token_ok(prefix)
    return tuple(f"continuous.{prefix}.{name}" for name in logical_names)
