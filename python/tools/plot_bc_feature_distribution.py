"""Plot one BC observation feature distribution by feature-importance name."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PY_ROOT = Path(__file__).resolve().parents[1]
_SUPERVISED_ROOT = _PY_ROOT / "supervised_learning"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))
if str(_SUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUPERVISED_ROOT))

from behavior_cloning import (  # noqa: E402
    BcEpisodePoolIterable,
    N_EPISODES,
    ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
    ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
    ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
    ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
    _OBS_EMBEDDING_FEATURE_KEY_BY_BASE,
    _bc_perm_feature_catalog,
)

_STAT_BATCH_CACHE_KEYS: tuple[str, ...] = (
    "orbit_planet_features",
    "orbit_planet_arrival_features",
    "orbit_planet_pairwise_features",
    "orbit_planet_mask",
    "orbit_planet_pairwise_mask",
    "orbit_enemy_mask",
    "player_mask",
)
_SPIKE_EPSILON = 1e-9


def _n_statistic_batches_default() -> int:
    raw = os.environ.get("N_STATISTIC_BATCHES", "").strip()
    if not raw:
        return 100
    out = int(raw)
    assert out >= 1, out
    return out


def _bc_episode_pt_files(data_dir: Path) -> list[Path]:
    root = data_dir.expanduser().resolve()
    assert root.is_dir(), root
    out = sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".pt")
    assert len(out) >= 1, (root, "expected at least one .pt episode")
    return out


def _stat_batch_cache_metadata(
    *,
    data_dir: Path,
    paths: list[Path],
    n_batches: int,
    batch_size: int,
    pool_episodes: int,
    loader_workers: int,
) -> dict[str, Any]:
    assert len(paths) >= 1, paths
    return {
        "data_dir": str(data_dir.expanduser().resolve()),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "pool_episodes": int(pool_episodes),
        "loader_workers": int(loader_workers),
        "cache_keys": _STAT_BATCH_CACHE_KEYS,
    }


def _stat_batch_cache_path(metadata: dict[str, Any]) -> Path:
    raw = repr(metadata).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:24]
    root = _PY_ROOT.parent / "outputs" / "plot_bc_feature_distribution_cache"
    return root / f"stat_batches_{digest}.pt"


def _cacheable_stat_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in _STAT_BATCH_CACHE_KEYS:
        assert key in batch, (key, sorted(batch.keys()))
        value = batch[key]
        assert isinstance(value, torch.Tensor), (key, type(value))
        out[key] = value.detach().cpu().clone()
    return out


def _load_or_build_stat_batches(
    *,
    data_dir: Path,
    paths: list[Path],
    n_batches: int,
    batch_size: int,
    pool_episodes: int,
    loader_workers: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    metadata = _stat_batch_cache_metadata(
        data_dir=data_dir,
        paths=paths,
        n_batches=n_batches,
        batch_size=batch_size,
        pool_episodes=pool_episodes,
        loader_workers=loader_workers,
    )
    cache_path = _stat_batch_cache_path(metadata)
    if cache_path.is_file():
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        assert isinstance(payload, dict), type(payload)
        assert payload["metadata"] == metadata, (payload["metadata"], metadata)
        batches = payload["batches"]
        assert isinstance(batches, tuple), type(batches)
        assert len(batches) == n_batches, (len(batches), n_batches)
        print(f"loaded stat batch cache: {cache_path}", flush=True)
        return batches

    dataset = BcEpisodePoolIterable(
        paths,
        pool_episodes=pool_episodes,
        batch_size=batch_size,
        infinite=True,
        deterministic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=loader_workers,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
    )
    batches: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batches.append(_cacheable_stat_batch(batch))
        if len(batches) >= n_batches:
            break
    assert len(batches) == n_batches, (len(batches), n_batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "batches": tuple(batches),
    }
    torch.save(payload, str(cache_path))
    print(f"wrote stat batch cache: {cache_path}", flush=True)
    return tuple(batches)


def _batch_with_embedding_aliases(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(batch)
    for base_key, embedding_key in _OBS_EMBEDDING_FEATURE_KEY_BY_BASE.items():
        assert base_key in batch, (base_key, sorted(batch.keys()))
        out[embedding_key] = batch[base_key]
    return out


def _expanded_mask_for_value_view(
    mask: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    out = mask
    while out.ndim < value.ndim:
        out = out.unsqueeze(-1)
    return out.expand_as(value)


def _player_block_valid_for_channel(
    *,
    batch: dict[str, torch.Tensor],
    channel: int,
    player_feature_offset: int,
    player_features_per_player: int,
) -> torch.Tensor:
    ch = int(channel)
    off = int(player_feature_offset)
    width = int(player_features_per_player)
    assert width >= 1, width
    player_active = batch["player_mask"] > 0.5
    if ch < off:
        return player_active
    block = (ch - off) // width
    assert block >= 0, (ch, off, width)
    if block == 0:
        return player_active
    enemy_idx = block - 1
    enemy_mask = batch["orbit_enemy_mask"] > 0.5
    assert 0 <= enemy_idx < int(enemy_mask.shape[-1]), (enemy_idx, tuple(enemy_mask.shape))
    return player_active & enemy_mask[..., enemy_idx]


def _feature_view_valid_mask(
    *,
    batch: dict[str, torch.Tensor],
    spec: Any,
    channel_index: tuple[int | slice, ...],
    view: torch.Tensor,
) -> torch.Tensor:
    if spec.obs_key == "orbit_planet_features":
        assert len(channel_index) == 1, (spec.name, channel_index)
        channel = channel_index[0]
        assert isinstance(channel, int), (spec.name, channel_index)
        token_mask = (batch["orbit_planet_mask"] > 0.5) & (batch["player_mask"] > 0.5).unsqueeze(-1)
        feature_mask = _player_block_valid_for_channel(
            batch=batch,
            channel=int(channel),
            player_feature_offset=ORBIT_PLANET_PLAYER_FEATURE_OFFSET,
            player_features_per_player=ORBIT_PLANET_PLAYER_FEATURES_PER_PLAYER,
        ).unsqueeze(-1)
        return _expanded_mask_for_value_view(token_mask & feature_mask, view)
    if spec.obs_key == "orbit_planet_pairwise_features":
        assert len(channel_index) == 1, (spec.name, channel_index)
        channel = channel_index[0]
        assert isinstance(channel, int), (spec.name, channel_index)
        token_mask = (batch["orbit_planet_pairwise_mask"] > 0.5) & (batch["player_mask"] > 0.5).unsqueeze(-1)
        feature_mask = _player_block_valid_for_channel(
            batch=batch,
            channel=int(channel),
            player_feature_offset=ORBIT_EDGE_PLAYER_FEATURE_OFFSET,
            player_features_per_player=ORBIT_EDGE_PLAYER_FEATURES_PER_PLAYER,
        ).unsqueeze(-1)
        return _expanded_mask_for_value_view(token_mask & feature_mask, view)
    if spec.obs_key == "orbit_planet_arrival_features":
        assert len(channel_index) == 3, (spec.name, channel_index)
        player_block = channel_index[1]
        assert isinstance(player_block, int), (spec.name, channel_index)
        token_mask = (batch["orbit_planet_mask"] > 0.5) & (batch["player_mask"] > 0.5).unsqueeze(-1)
        if int(player_block) == 0:
            feature_mask = batch["player_mask"] > 0.5
        else:
            enemy_idx = int(player_block) - 1
            enemy_mask = batch["orbit_enemy_mask"] > 0.5
            assert 0 <= enemy_idx < int(enemy_mask.shape[-1]), (enemy_idx, tuple(enemy_mask.shape))
            feature_mask = (batch["player_mask"] > 0.5) & enemy_mask[..., enemy_idx]
        return _expanded_mask_for_value_view(token_mask & feature_mask.unsqueeze(-1), view)
    if spec.obs_key in ("orbit_planet_mask", "orbit_enemy_mask", "orbit_planet_pairwise_mask"):
        return torch.ones_like(view, dtype=torch.bool)
    raise AssertionError((spec.name, spec.obs_key))


def _masked_feature_values(
    batch: dict[str, torch.Tensor],
    spec: Any,
) -> torch.Tensor:
    obs_key = spec.obs_key
    if spec.feature_input == "embedding":
        obs_key = _OBS_EMBEDDING_FEATURE_KEY_BY_BASE[spec.obs_key]
    else:
        assert spec.feature_input in ("continuous", "shared"), spec.feature_input
    aliased_batch = _batch_with_embedding_aliases(batch)
    tensor = aliased_batch[obs_key]
    assert isinstance(tensor, torch.Tensor), (obs_key, type(tensor))
    views = tuple(
        (tensor if len(spec.channel_indices) == 0 else tensor[(Ellipsis, *channel_index)])
        for channel_index in spec.channel_indices
    )
    if len(spec.channel_indices) == 0:
        views = (tensor,)
    values: list[torch.Tensor] = []
    channel_indices = spec.channel_indices if len(spec.channel_indices) > 0 else ((),)
    for channel_index, view in zip(channel_indices, views, strict=True):
        assert isinstance(view, torch.Tensor), type(view)
        mask = _feature_view_valid_mask(
            batch=batch,
            spec=spec,
            channel_index=channel_index,
            view=view,
        )
        values.append(view[mask].detach().to(dtype=torch.float32).reshape(-1))
    assert len(values) >= 1, spec.name
    return torch.cat(values, dim=0)


def _shared_feature_name(name: str) -> str:
    assert "@" not in name or name.startswith(("continuous.", "embedding.")), name
    return name.split("@", maxsplit=1)[0]


def _shared_feature_specs(*, per_horizon_arrival: bool) -> tuple[Any, ...]:
    specs = _bc_perm_feature_catalog(per_horizon_arrival=per_horizon_arrival)
    by_name: dict[str, Any] = {}
    channel_indices_by_name: dict[str, list[tuple[int | slice, ...]]] = {}
    for spec in specs:
        name = _shared_feature_name(spec.name)
        if name not in by_name:
            by_name[name] = spec
            channel_indices_by_name[name] = []
        first = by_name[name]
        assert first.obs_key == spec.obs_key, (name, first.obs_key, spec.obs_key)
        assert first.feature_input == spec.feature_input, (name, first.feature_input, spec.feature_input)
        channel_indices_by_name[name].extend(spec.channel_indices)

    out: list[Any] = []
    for name, spec in by_name.items():
        out.append(
            type(spec)(
                name=name,
                obs_key=spec.obs_key,
                channel_indices=tuple(channel_indices_by_name[name]),
                feature_input=spec.feature_input,
            )
        )
    names = tuple(spec.name for spec in out)
    assert len(names) == len(set(names)), "duplicate shared feature names"
    return tuple(out)


def _feature_spec_by_name(feature_name: str, *, per_horizon_arrival: bool) -> Any:
    assert "@" not in feature_name, feature_name
    specs = _shared_feature_specs(per_horizon_arrival=per_horizon_arrival)
    by_name = {spec.name: spec for spec in specs}
    assert len(by_name) == len(specs), "duplicate feature names"
    if feature_name not in by_name:
        names = "\n".join(sorted(by_name))
        raise SystemExit(f"unknown feature {feature_name!r}; available names:\n{names}")
    return by_name[feature_name]


def _normalization_feature_key(feature_name: str) -> str:
    assert feature_name.startswith("continuous."), feature_name
    assert "@" not in feature_name, feature_name
    return feature_name


def _filtered_spike_values(values: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    assert values.ndim == 1, tuple(values.shape)
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    assert unique.ndim == counts.ndim == 1, (unique.shape, counts.shape)
    assert int(unique.numel()) >= 1, tuple(values.shape)
    spike_idx = int(torch.argmax(counts).item())
    spike_value = float(unique[spike_idx].item())
    spike_mask = torch.abs(values - spike_value) <= _SPIKE_EPSILON
    spike_count = int(spike_mask.sum().item())
    filtered = values[~spike_mask]
    assert int(filtered.numel()) >= 1, (spike_value, spike_count, int(values.numel()))
    return filtered, spike_value, spike_count


def _iteratively_filter_spikes(
    values: torch.Tensor,
    *,
    n_spikes: int,
) -> tuple[torch.Tensor, tuple[tuple[float, int], ...]]:
    assert values.ndim == 1, tuple(values.shape)
    n = int(n_spikes)
    assert n >= 0, n_spikes
    current = values
    spikes: list[tuple[float, int]] = []
    for _ in range(n):
        filtered, spike_value, spike_count = _filtered_spike_values(current)
        spikes.append((spike_value, spike_count))
        current = filtered
    return current, tuple(spikes)


def _top_spike_value(values: torch.Tensor) -> tuple[float, int]:
    assert values.ndim == 1, tuple(values.shape)
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    assert unique.ndim == counts.ndim == 1, (unique.shape, counts.shape)
    assert int(unique.numel()) >= 1, tuple(values.shape)
    spike_idx = int(torch.argmax(counts).item())
    return float(unique[spike_idx].item()), int(counts[spike_idx].item())


def _print_top_exact_spikes(values: torch.Tensor, *, top_k: int) -> None:
    assert values.ndim == 1, tuple(values.shape)
    k = int(top_k)
    assert k >= 1, top_k
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    assert unique.ndim == counts.ndim == 1, (unique.shape, counts.shape)
    assert int(unique.numel()) >= 1, tuple(values.shape)
    n_show = min(k, int(unique.numel()))
    top_counts, top_indices = torch.topk(counts, k=n_show, largest=True, sorted=True)
    total = int(values.numel())
    assert total >= 1, total
    print("top_exact_spikes:", flush=True)
    for rank, (count, idx) in enumerate(zip(top_counts.tolist(), top_indices.tolist(), strict=True), start=1):
        value = float(unique[int(idx)].item())
        count_i = int(count)
        frac = float(count_i) / float(total)
        print(f"  {rank:02d}: value={value:.10g} count={count_i} frac={frac:.6f}", flush=True)


def _standardized_values(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    assert values.ndim == 1, tuple(values.shape)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    assert std > 0.0, (mean, std, int(values.numel()))
    return (values - mean) / std, mean, std


def _clip_quantiles_from_arg(raw: str) -> tuple[float, float]:
    text = raw.strip()
    assert text.startswith("(") and text.endswith(")"), raw
    parts = text[1:-1].split(",")
    assert len(parts) == 2, raw
    lo_q = float(parts[0].strip())
    hi_q = float(parts[1].strip())
    assert 0.0 <= lo_q <= hi_q <= 1.0, (lo_q, hi_q)
    return lo_q, hi_q


def _quantile_clipped_values(
    values: torch.Tensor,
    *,
    clip_down_quantile: float,
    clip_up_quantile: float,
) -> tuple[torch.Tensor, float, float]:
    assert values.ndim == 1, tuple(values.shape)
    n = int(values.numel())
    assert n >= 1, tuple(values.shape)
    lo_q = float(clip_down_quantile)
    hi_q = float(clip_up_quantile)
    assert 0.0 <= lo_q <= hi_q <= 1.0, (clip_down_quantile, clip_up_quantile)
    lo_k = int(lo_q * float(n - 1)) + 1
    hi_k = int(hi_q * float(n - 1)) + 1
    assert 1 <= lo_k <= hi_k <= n, (lo_k, hi_k, n)
    clip_down = float(torch.kthvalue(values, lo_k).values.item())
    clip_up = float(torch.kthvalue(values, hi_k).values.item())
    assert clip_down <= clip_up, (clip_down, clip_up)
    return torch.clamp(values, min=clip_down, max=clip_up), clip_down, clip_up


def _filter_clipped_values(
    clipped: torch.Tensor,
    *,
    clip_down: float,
    clip_up: float,
) -> tuple[torch.Tensor, int, int]:
    assert clipped.ndim == 1, tuple(clipped.shape)
    lo_mask = clipped == float(clip_down)
    hi_mask = clipped == float(clip_up)
    keep = ~(lo_mask | hi_mask)
    out = clipped[keep]
    assert int(out.numel()) >= 1, (clip_down, clip_up, int(clipped.numel()))
    return out, int(lo_mask.sum().item()), int(hi_mask.sum().item())


def _plot_histograms(
    *,
    feature_name: str,
    panels: tuple[tuple[str, torch.Tensor], ...],
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    assert n_panels >= 1, n_panels
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.suptitle(feature_name)
    flat_axes = axes.reshape(-1)
    for ax, (title, values) in zip(flat_axes[:n_panels], panels, strict=True):
        assert values.ndim == 1, (title, tuple(values.shape))
        ax.hist(values.numpy(), bins=int(bins))
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    for ax in flat_axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot one BC feature distribution by importance feature name.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with BC episode .pt files.")
    ap.add_argument("--feature", required=False, default=None, help="Feature name, e.g. continuous.edge.edge_distance.")
    ap.add_argument("--list-features", action="store_true", help="Print available feature names and exit.")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pool-episodes", type=int, default=N_EPISODES)
    ap.add_argument("--loader-workers", type=int, default=4)
    ap.add_argument("--n-statistic-batches", type=int, default=_n_statistic_batches_default())
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--clip-tail-frac", type=str, default="(0.001,0.999)")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--disable", action="store_true")
    ap.add_argument("--n-spikes", type=int, default=1)
    ap.add_argument("--feature-importance-per-horizon", action="store_true")
    args = ap.parse_args()

    specs = _shared_feature_specs(per_horizon_arrival=bool(args.feature_importance_per_horizon))
    if bool(args.list_features):
        for spec in specs:
            print(spec.name)
        return
    if args.feature is None:
        raise SystemExit("--feature is required unless --list-features is passed")

    feature_name = str(args.feature)
    if "@" in feature_name:
        raise SystemExit("--feature must be a shared logical feature name without '@...' suffix")
    spec = _feature_spec_by_name(
        feature_name,
        per_horizon_arrival=bool(args.feature_importance_per_horizon),
    )
    n_batches = int(args.n_statistic_batches)
    assert n_batches >= 1, n_batches
    batch_size = int(args.batch_size)
    assert batch_size >= 1, batch_size
    bins = int(args.bins)
    assert bins >= 1, bins
    enabled = not bool(args.disable)
    clip_enabled = enabled and not bool(args.no_clip)
    norm_enabled = enabled and not bool(args.no_norm)
    if clip_enabled:
        clip_down_quantile, clip_up_quantile = _clip_quantiles_from_arg(str(args.clip_tail_frac))
    n_spikes = int(args.n_spikes)
    assert n_spikes >= 0, n_spikes
    if not enabled:
        n_spikes = 0

    paths = _bc_episode_pt_files(args.data_dir)
    loader_workers = int(args.loader_workers)
    assert loader_workers >= 1, loader_workers
    assert loader_workers <= len(paths), (loader_workers, len(paths))
    assert loader_workers <= int(args.pool_episodes), (loader_workers, int(args.pool_episodes))
    batches = _load_or_build_stat_batches(
        data_dir=args.data_dir,
        paths=paths,
        n_batches=n_batches,
        batch_size=batch_size,
        pool_episodes=int(args.pool_episodes),
        loader_workers=loader_workers,
    )

    chunks: list[torch.Tensor] = []
    for batch in batches:
        chunks.append(_masked_feature_values(batch, spec))
    assert len(chunks) >= 1, (feature_name, n_batches)
    values = torch.cat(chunks, dim=0)
    values = values[torch.isfinite(values)]
    assert int(values.numel()) >= 1, feature_name
    assert spec.feature_input == "continuous", (feature_name, spec.feature_input)

    _print_top_exact_spikes(values, top_k=10)
    filtered_values, spikes = _iteratively_filter_spikes(values, n_spikes=n_spikes)
    if clip_enabled:
        clipped, clip_down, clip_up = _quantile_clipped_values(
            filtered_values,
            clip_down_quantile=clip_down_quantile,
            clip_up_quantile=clip_up_quantile,
        )
        clip_filtered, clip_down_count, clip_up_count = _filter_clipped_values(
            clipped,
            clip_down=clip_down,
            clip_up=clip_up,
        )
        norm_input = clip_filtered
    else:
        clip_down = 0.0
        clip_up = 0.0
        norm_input = filtered_values
    if norm_enabled:
        normalized_values, mean, std = _standardized_values(norm_input)
    else:
        normalized_values = norm_input
        mean = 0.0
        std = 1.0
    norm_key = _normalization_feature_key(feature_name)
    if clip_enabled:
        print(
            f"clip_exact_spikes: down={clip_down:.10g} count={clip_down_count} "
            f"up={clip_up:.10g} count={clip_up_count}",
            flush=True,
        )
    else:
        print("clip_exact_spikes: disabled", flush=True)
    print(f"{norm_key!r}: _obs_feature_norm(", flush=True)
    print(f"    mean={mean:.10g},", flush=True)
    print(f"    std={std:.10g},", flush=True)
    print(f"    clip_down={clip_down:.10g},", flush=True)
    print(f"    clip_up={clip_up:.10g},", flush=True)
    spike_values_text = ", ".join(f"{spike_value:.10g}" for spike_value, _count in spikes)
    if len(spikes) == 1:
        spike_values_text = f"{spike_values_text},"
    print(f"    spike_values=({spike_values_text}),", flush=True)
    print(f"    enabled={enabled},", flush=True)
    print(f"    norm_enabled={norm_enabled},", flush=True)
    print(f"    clip_enabled={clip_enabled},", flush=True)
    print("),", flush=True)
    panel_list: list[tuple[str, torch.Tensor]] = [(f"raw n={int(values.numel())}", values)]
    current = values
    for spike_i, (spike_value, spike_count) in enumerate(spikes, start=1):
        current = current[torch.abs(current - spike_value) > _SPIKE_EPSILON]
        panel_list.append(
            (
                f"after spike {spike_i}: value={spike_value:.6g} "
                f"spike_n={spike_count} n={int(current.numel())}",
                current,
            )
        )
    if clip_enabled:
        panel_list.extend(
            (
                (
                    f"raw clipped p{100.0 * clip_down_quantile:.4g}={clip_down:.6g} "
                    f"p{100.0 * clip_up_quantile:.4g}={clip_up:.6g}",
                    clipped,
                ),
                (
                    f"after clipped endpoints down_n={clip_down_count} "
                    f"up_n={clip_up_count} n={int(clip_filtered.numel())}",
                    clip_filtered,
                ),
            )
        )
    else:
        norm_input = filtered_values
    if norm_enabled:
        panel_list.append(
            (
                f"z-score after exclusions mean={mean:.6g} std={std:.6g}",
                normalized_values,
            )
        )
    elif not enabled:
        panel_list.append((f"disabled n={int(values.numel())}", values))
    else:
        panel_list.append((f"raw after exclusions n={int(norm_input.numel())}", normalized_values))
    panels = tuple(panel_list)
    _plot_histograms(feature_name=feature_name, panels=panels, bins=bins)


if __name__ == "__main__":
    main()
