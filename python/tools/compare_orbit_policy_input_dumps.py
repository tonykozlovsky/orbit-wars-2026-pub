from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

_EXPECTED_FORMAT = "orbit_policy_network_inputs_dump_v1"
_IMPALA_PROJECT_ROOT_RAW = os.environ.get(
    "IMPALA_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[2]),
).strip()
assert _IMPALA_PROJECT_ROOT_RAW, "IMPALA_PROJECT_ROOT must be non-empty"
_IMPALA_PROJECT_ROOT = Path(_IMPALA_PROJECT_ROOT_RAW).expanduser().resolve()
_DEFAULT_LEFT = _IMPALA_PROJECT_ROOT / "replays/replay_rl_vis_inputs.pt"
_DEFAULT_RIGHT = _IMPALA_PROJECT_ROOT / "replays/replay_submission_inputs.pt"


def _load_dump(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    assert resolved.is_file(), resolved
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict), type(payload)
    assert payload["format"] == _EXPECTED_FORMAT, payload["format"]
    input_keys = payload["input_keys"]
    samples = payload["samples"]
    samples_per_step = payload["samples_per_step"]
    assert isinstance(input_keys, tuple), type(input_keys)
    assert len(input_keys) >= 1, input_keys
    assert all(isinstance(k, str) and len(k) > 0 for k in input_keys), input_keys
    assert samples_per_step == 1, samples_per_step
    assert isinstance(samples, list), type(samples)
    assert len(samples) >= 1, path
    for sample_index, sample in enumerate(samples):
        assert isinstance(sample, tuple), (path, sample_index, type(sample))
        assert len(sample) == len(input_keys), (path, sample_index, len(sample), len(input_keys))
        for key, tensor in zip(input_keys, sample, strict=True):
            assert isinstance(tensor, torch.Tensor), (path, sample_index, key, type(tensor))
            assert not tensor.is_cuda, (path, sample_index, key, tensor.device)
            assert tensor.is_contiguous(), (path, sample_index, key, tuple(tensor.stride()))
    return payload


def _first_true_index(mask: torch.Tensor) -> tuple[int, ...]:
    assert isinstance(mask, torch.Tensor), type(mask)
    assert mask.dtype == torch.bool, mask.dtype
    assert bool(mask.any().item()), tuple(mask.shape)
    flat_idx = int(torch.argmax(mask.reshape(-1).to(dtype=torch.int64)).item())
    return tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_idx), mask.shape))


def _float_mismatch_mask(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> torch.Tensor:
    left_f = left.to(dtype=torch.float32)
    right_f = right.to(dtype=torch.float32)
    finite_pair = torch.isfinite(left_f) & torch.isfinite(right_f)
    same_nonfinite = (~finite_pair) & (left_f == right_f)
    close_finite = torch.isclose(left_f, right_f, rtol=float(rtol), atol=float(atol))
    return ~(torch.where(finite_pair, close_finite, same_nonfinite))


def _compare_tensor(
    *,
    key: str,
    sample_index: int,
    left_name: str,
    right_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    atol: float,
    rtol: float,
) -> str | None:
    assert isinstance(key, str) and len(key) > 0, key
    assert isinstance(left, torch.Tensor), (key, sample_index, type(left))
    assert isinstance(right, torch.Tensor), (key, sample_index, type(right))
    assert tuple(left.shape) == tuple(right.shape), (key, sample_index, tuple(left.shape), tuple(right.shape))

    if torch.is_floating_point(left) or torch.is_floating_point(right):
        assert torch.is_floating_point(left) and torch.is_floating_point(right), (
            key,
            sample_index,
            left.dtype,
            right.dtype,
        )
        mismatch = _float_mismatch_mask(left, right, atol=atol, rtol=rtol)
        num_bad = int(mismatch.to(dtype=torch.int64).sum().item())
        if num_bad == 0:
            return None
        abs_diff = (left.to(dtype=torch.float32) - right.to(dtype=torch.float32)).abs()
        first_idx = _first_true_index(mismatch)
        return (
            f"sample={sample_index} key={key} {left_name}_dtype={left.dtype} {right_name}_dtype={right.dtype} "
            f"shape={tuple(left.shape)} "
            f"num_not_close={num_bad}/{left.numel()} "
            f"max_abs={float(abs_diff.max().item()):.9g} "
            f"mean_abs={float(abs_diff.mean().item()):.9g} "
            f"first_idx={first_idx} "
            f"{left_name}={float(left[first_idx].to(dtype=torch.float32).item())!r} "
            f"{right_name}={float(right[first_idx].to(dtype=torch.float32).item())!r}"
        )

    assert left.dtype == right.dtype, (key, sample_index, left.dtype, right.dtype)
    mismatch = left != right
    num_bad = int(mismatch.to(dtype=torch.int64).sum().item())
    if num_bad == 0:
        return None
    first_idx = _first_true_index(mismatch)
    return (
        f"sample={sample_index} key={key} dtype={left.dtype} shape={tuple(left.shape)} "
        f"num_diff={num_bad}/{left.numel()} "
        f"first_idx={first_idx} "
        f"{left_name}={left[first_idx].item()!r} {right_name}={right[first_idx].item()!r}"
    )


def _compare_dumps(
    *,
    left_name: str,
    left_payload: dict[str, Any],
    right_name: str,
    right_payload: dict[str, Any],
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> list[str]:
    assert left_payload["format"] == right_payload["format"], (
        left_payload["format"],
        right_payload["format"],
    )
    left_keys = tuple(left_payload["input_keys"])
    right_keys = tuple(right_payload["input_keys"])
    assert left_keys == right_keys, (left_keys, right_keys)
    assert left_payload["samples_per_step"] == right_payload["samples_per_step"], (
        left_payload["samples_per_step"],
        right_payload["samples_per_step"],
    )
    left_samples = left_payload["samples"]
    right_samples = right_payload["samples"]
    assert len(left_samples) == len(right_samples), (len(left_samples), len(right_samples))
    assert max_mismatches >= 1, max_mismatches

    mismatches: list[str] = []
    for sample_index, (left_sample, right_sample) in enumerate(zip(left_samples, right_samples, strict=True)):
        assert isinstance(left_sample, tuple), (sample_index, type(left_sample))
        assert isinstance(right_sample, tuple), (sample_index, type(right_sample))
        assert len(left_sample) == len(left_keys), (sample_index, len(left_sample), len(left_keys))
        assert len(right_sample) == len(left_keys), (sample_index, len(right_sample), len(left_keys))
        for key, left_tensor, right_tensor in zip(left_keys, left_sample, right_sample, strict=True):
            mismatch = _compare_tensor(
                key=key,
                sample_index=sample_index,
                left_name=left_name,
                right_name=right_name,
                left=left_tensor,
                right=right_tensor,
                atol=atol,
                rtol=rtol,
            )
            if mismatch is not None:
                mismatches.append(mismatch)
                if len(mismatches) >= max_mismatches:
                    return mismatches
    return mismatches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Orbit policy network input dumps.")
    parser.add_argument("--left", type=Path, default=_DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=_DEFAULT_RIGHT)
    parser.add_argument("--bc", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-mismatches", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert float(args.atol) >= 0.0, args.atol
    assert float(args.rtol) >= 0.0, args.rtol
    assert int(args.max_mismatches) >= 1, args.max_mismatches

    dumps: list[tuple[str, dict[str, Any]]] = [
        ("left", _load_dump(args.left)),
        ("right", _load_dump(args.right)),
    ]
    if args.bc is not None:
        dumps.append(("bc", _load_dump(args.bc)))

    mismatches: list[str] = []
    for left_index in range(len(dumps)):
        for right_index in range(left_index + 1, len(dumps)):
            left_name, left_payload = dumps[left_index]
            right_name, right_payload = dumps[right_index]
            remaining = int(args.max_mismatches) - len(mismatches)
            if remaining <= 0:
                break
            pair_mismatches = _compare_dumps(
                left_name=left_name,
                left_payload=left_payload,
                right_name=right_name,
                right_payload=right_payload,
                atol=float(args.atol),
                rtol=float(args.rtol),
                max_mismatches=remaining,
            )
            mismatches.extend(f"{left_name}_vs_{right_name}: {line}" for line in pair_mismatches)

    sample_count = len(dumps[0][1]["samples"])
    key_count = len(dumps[0][1]["input_keys"])
    pair_count = len(dumps) * (len(dumps) - 1) // 2
    print(
        f"Compared dumps={len(dumps)} pairs={pair_count} samples={sample_count} "
        f"keys={key_count} atol={float(args.atol)} rtol={float(args.rtol)}"
    )
    if len(mismatches) == 0:
        print("DUMPS_MATCH")
        return
    print(f"DUMPS_DIFFER shown_mismatches={len(mismatches)}")
    for line in mismatches:
        print(line)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
