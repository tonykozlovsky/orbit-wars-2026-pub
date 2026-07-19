from collections.abc import Callable
from copy import copy
import logging
from typing import Any, Union

import torch


def tree_to_device(tree: Any, device: Union[str, torch.device]) -> Any:
    """Recursively ``.to(device)`` on ``Tensor`` leaves. ``non_blocking`` is ``True`` only for CUDA targets."""
    d = torch.device(device) if isinstance(device, str) else device
    non_blocking = d.type == "cuda"
    if isinstance(tree, dict):
        return {k: tree_to_device(v, d) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.to(d, non_blocking=non_blocking)
    apply_fn = getattr(tree, "apply", None)
    if callable(apply_fn):
        return apply_fn(lambda t: t.to(d, non_blocking=non_blocking))
    return tree


def take_first_batch_in_tree(obj: Any, *, batch_size: int) -> Any:
    """Slice tensors along dim 0 to batch size 1 where dim 0 equals ``batch_size`` (env batch)."""
    if isinstance(obj, dict):
        return {k: take_first_batch_in_tree(v, batch_size=batch_size) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        if obj.ndim > 0 and int(obj.shape[0]) == int(batch_size):
            return obj[0:1].contiguous()
        return obj
    return obj

Buffers = list[dict[str, Union[dict, torch.Tensor]]]


def fill_buffers_inplace(buffers, fill_vals, non_blocking=False):
    if isinstance(fill_vals, dict):
        assert set(buffers.keys()) == set(fill_vals.keys()), (
            "fill_buffers_inplace keys differ: "
            f"{set(buffers.keys())} != {set(fill_vals.keys())}"
        )
        for key, val in fill_vals.items():
            assert key in buffers
            fill_buffers_inplace(buffers[key], val, non_blocking=non_blocking)
    else:
        # logging.info("SHAPES:", buffers.shape, fill_vals.shape)
        buffers.copy_(fill_vals, non_blocking=non_blocking)


def fill_buffers_inplace_2(
    buffers: Union[dict, torch.Tensor],
    fill_vals: Union[dict, torch.Tensor],
    step: int,
    key_path: str = "",
):
    if isinstance(fill_vals, dict):
        assert isinstance(buffers, dict), f"Expected dict buffers for dict fill_vals, got {type(buffers)}"
        for key, val in fill_vals.items():
            assert key in buffers, f"fill_buffers_inplace_2 missing destination key: {key}"
            child_path = key if not key_path else f"{key_path}.{key}"
            fill_buffers_inplace_2(buffers[key], val, step, key_path=child_path)
    else:
        assert isinstance(fill_vals, torch.Tensor), (
            f"Expected tensor fill values, got {type(fill_vals)}"
        )
        dst = buffers[step, ...]
        assert tuple(dst.shape) == tuple(fill_vals.shape), (
            f"fill_buffers_inplace_2 shape mismatch at key_path={key_path!r} step={step}: "
            f"dst={tuple(dst.shape)} src={tuple(fill_vals.shape)}"
        )
        dst.copy_(fill_vals, non_blocking=False)


def copy_infer_slot_src_slice_to_dst(
    dst: Union[dict, torch.Tensor],
    src: Union[dict, torch.Tensor],
    a: int,
    b: int,
) -> None:
    if isinstance(src, dict) and isinstance(dst, dict):
        for k in src:
            if k in dst:
                copy_infer_slot_src_slice_to_dst(dst[k], src[k], a, b)
        return
    assert isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor)
    sl = src[a:b].to("cpu", non_blocking=False)
    assert tuple(sl.shape) == tuple(dst.shape), (
        "inference copy shape mismatch: "
        f"dst={tuple(dst.shape)} sl={tuple(sl.shape)}"
    )
    dst.copy_(sl, non_blocking=False)


def copy_matching_tree_into(dst: Any, src: Any) -> None:
    if isinstance(src, dict):
        assert isinstance(dst, dict), (
            f"copy_matching_tree_into: dict src requires dict dst, got {type(dst)}"
        )
        for k in src:
            if k in dst:
                copy_matching_tree_into(dst[k], src[k])
        return
    assert isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor)
    dst.copy_(src, non_blocking=False)


def fill_buffers_inplace_3(
    buffers: Union[dict, torch.Tensor],
    fill_vals: Union[dict, torch.Tensor],
    a,
    b,
    key_path: str = "",
):
    if isinstance(fill_vals, dict):
        assert isinstance(buffers, dict), (
            f"Expected dict destination for dict source at key_path='{key_path}', "
            f"got {type(buffers)}"
        )
        for key, val in fill_vals.items():
            if key not in buffers:
                continue
            child_key_path = key if not key_path else f"{key_path}.{key}"
            fill_buffers_inplace_3(buffers[key], val, a, b, child_key_path)
    else:
        assert isinstance(fill_vals, torch.Tensor), (
            f"Expected tensor fill_vals for key_path='{key_path}', got {type(fill_vals)}"
        )
        dst_slice = buffers[:, a:b, ...]
        assert tuple(dst_slice.shape) == tuple(fill_vals.shape), (
            "fill_buffers_inplace_3 shape mismatch "
            f"key_path='{key_path}' a={a} b={b}: dst_slice={tuple(dst_slice.shape)} "
            f"src={tuple(fill_vals.shape)} dst_full={tuple(buffers.shape)}"
        )
        dst_slice.copy_(fill_vals, non_blocking=True)


def stack_buffers(
    buffers: Buffers,
    dim: int,
    *,
    _path: str = "",
) -> dict[str, Union[dict, torch.Tensor]]:
    stacked_buffers = {}
    for key, val in copy(buffers[0]).items():
        path = f"{_path}.{key}" if _path else str(key)
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim, _path=path)
        else:
            parts = [b[key] for b in buffers]
            try:
                stacked_buffers[key] = torch.cat(parts, dim=dim)
            except RuntimeError:
                shape_info = [
                    tuple(t.shape) if isinstance(t, torch.Tensor) else repr(type(t))
                    for t in parts
                ]
                logging.error(
                    "stack_buffers torch.cat failed path=%r dim=%s shapes=%s dtypes=%s",
                    path,
                    dim,
                    shape_info,
                    [
                        str(t.dtype) if isinstance(t, torch.Tensor) else None
                        for t in parts
                    ],
                )
                raise
    return stacked_buffers


def stack_learn_timestep_template_along_time(
    sample_one_timestep: dict,
    *,
    time_steps: int,
) -> dict:
    """
    Buffer allocation template: prepend time axis ``dim=0`` with length ``time_steps``.
    Actor later fills ``buffers[step, ...]`` in-place per ``step`` (see ``fill_buffers_inplace_2``).
    """
    assert int(time_steps) >= 1, f"time_steps must be >= 1, got {time_steps}"
    cpu = buffers_apply(
        sample_one_timestep,
        lambda x: x.to("cpu", non_blocking=False),
    )
    parts = [
        buffers_apply(cpu, lambda x: x.unsqueeze(0).clone())
        for _ in range(int(time_steps))
    ]
    return stack_buffers(parts, dim=0)


def split_buffers(
    buffers: dict[str, Union[dict, torch.Tensor]],
    split_size_or_sections: Union[int, list[int]],
    dim: int,
    contiguous: bool,
) -> list[Union[dict, torch.Tensor]]:
    buffers_split = None
    for key, val in copy(buffers).items():
        if isinstance(val, dict):
            bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
        else:
            # Handle torch.Tensor splitting (existing behavior)
            bufs = torch.split(val, split_size_or_sections, dim=dim)
            if contiguous:
                bufs = [b.contiguous() for b in bufs]

        if buffers_split is None:
            buffers_split = [{} for _ in range(len(bufs))]
        assert len(bufs) == len(buffers_split)
        buffers_split = [dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split, strict=False)]
    return buffers_split


def buffers_apply(
    buffers: Union[dict, torch.Tensor], func: Callable[[torch.Tensor], Any]
) -> Union[dict, torch.Tensor]:
    if isinstance(buffers, dict):
        return {key: buffers_apply(val, func) for key, val in copy(buffers).items()}
    else:
        # Handle torch.Tensor (existing behavior)
        return func(buffers)


def copy_buffers(
    buffers: Union[dict, torch.Tensor, Any]
) -> Union[dict, torch.Tensor, Any]:
    if isinstance(buffers, dict):
        return {key: copy_buffers(val) for key, val in copy(buffers).items()}
    else:
        # torch.Tensor: return a cloned tensor
        return buffers.clone()

def get_buffers_with_tag(buffers: Union[dict, torch.Tensor], device=None, tag=None, keep=False):
    assert tag is not None
    if isinstance(buffers, dict):
        result = {
            key: get_buffers_with_tag(val, device, tag, keep or tag in key)
            for key, val in copy(buffers).items()
        }
        keys = list(result.keys())
        for key in keys:
            if result[key] is None:
                result.pop(key)
        if len(result) == 0:
            return None
        return result
    else:
        # Handle torch.Tensor (existing behavior)
        if keep:
            return buffers.to(device, non_blocking=False)

        return None