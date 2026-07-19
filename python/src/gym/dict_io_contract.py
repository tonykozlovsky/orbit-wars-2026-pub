from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import torch

_LOG = logging.getLogger(__name__)

_ENV_DICT_IO_VALIDATE = "IMPALA_DICT_IO_VALIDATE"


def dict_io_contract_validation_enabled() -> bool:
    return os.environ.get(_ENV_DICT_IO_VALIDATE, "").strip() == "1"


_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def _flags_to_plain(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _flags_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _flags_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_flags_to_plain(x) for x in obj]
    return obj


def _dtype_from_spec(name: str) -> torch.dtype:
    assert isinstance(name, str)
    assert name in _DTYPE_BY_NAME, f"unsupported dtype name {name!r}"
    return _DTYPE_BY_NAME[name]


def _resolve_dim(flags: Any, x: Any) -> int:
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        assert x.startswith("flags."), f"shape dim must be int or flags.* string, got {x!r}"
        v = getattr(flags, x[len("flags.") :])
        return int(v)
    raise TypeError(f"invalid shape dimension spec: {type(x)!r} ({x!r})")


def _resolved_shape(flags: Any, shape_spec: list[Any]) -> tuple[int, ...]:
    assert isinstance(shape_spec, list)
    return tuple(_resolve_dim(flags, d) for d in shape_spec)


def _get_leaf_with_path_error(tree: dict[str, Any], path: str) -> tuple[Any, str | None]:
    parts = [p for p in path.split(".") if p]
    assert len(parts) > 0, "path must be non-empty"
    cur: Any = tree
    walked: list[str] = []
    for p in parts:
        if not isinstance(cur, dict):
            loc = ".".join(walked) if walked else "<root>"
            return None, (
                f"expected dict under {loc!r} to continue path; got type={type(cur).__name__!r} "
                f"while resolving segment {p!r} of path {path!r}"
            )
        if p not in cur:
            keys = sorted(cur.keys())
            loc = ".".join(walked) if walked else "<root>"
            return None, (
                f"missing key {p!r} under {loc!r} (path {path!r}); "
                f"available_keys={keys!r}"
            )
        walked.append(p)
        cur = cur[p]
    return cur, None


def _tensor_actual_line(value: torch.Tensor) -> str:
    assert isinstance(value, torch.Tensor)
    return f"torch.Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device!r}"


def _schema_tensor_line(
    flags: Any,
    shape_spec: list[Any],
    dtype_name: str,
    expected_dtype: torch.dtype,
) -> str:
    exp_shape = _resolved_shape(flags, shape_spec)
    return (
        f"torch.Tensor shape={list(exp_shape)} dtype={expected_dtype} "
        f"(yaml shape={shape_spec!r} dtype_name={dtype_name!r})"
    )


def _fail_contract_validation(
    contract_name: str,
    path: str,
    *,
    expected: str,
    actual: str,
    hint: str,
) -> None:
    block = (
        "dict_io_contract validation failed\n"
        f"  contract: {contract_name}\n"
        f"  path:     {path}\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"  hint:     {hint}"
    )
    _LOG.error("%s", block)
    raise AssertionError(block)


def _normalize_named_leaf_block(block: Any, *, block_name: str) -> list[dict[str, Any]]:
    if isinstance(block, list):
        for entry in block:
            assert isinstance(entry, dict), f"named_leaf_sets[{block_name!r}] list entry must be a dict"
            assert "bool" not in entry, (
                f"named_leaf_sets[{block_name!r}]: list entries must not use ``bool``; tensors only"
            )
        return block
    assert isinstance(block, dict), f"named_leaf_sets[{block_name!r}] must be a list or dict"
    paths = block.get("paths")
    tensors = block.get("tensors")
    assert paths is not None, f"named_leaf_sets[{block_name!r}]: dict block must have paths"
    assert tensors is not None, f"named_leaf_sets[{block_name!r}]: dict block must have tensors"
    assert isinstance(paths, list), f"named_leaf_sets[{block_name!r}].paths must be a list"
    assert isinstance(tensors, dict), f"named_leaf_sets[{block_name!r}].tensors must be a dict"
    assert "bools" not in block, (
        f"named_leaf_sets[{block_name!r}]: ``bools`` is not allowed; declare every leaf under ``tensors`` only"
    )
    paths_set = {x for x in paths if isinstance(x, str)}
    assert len(paths_set) == len(paths), f"named_leaf_sets[{block_name!r}].paths has duplicates or non-str"
    tensor_keys = set(tensors.keys())
    assert tensor_keys <= paths_set, (
        f"named_leaf_sets[{block_name!r}]: tensors keys must be a subset of paths, extra={tensor_keys - paths_set!r}"
    )
    assert paths_set == tensor_keys, (
        f"named_leaf_sets[{block_name!r}]: each path must have a ``tensors`` entry; missing={paths_set - tensor_keys!r}"
    )
    out: list[dict[str, Any]] = []
    for path in paths:
        assert isinstance(path, str)
        spec = tensors[path]
        assert isinstance(spec, dict), f"named_leaf_sets[{block_name!r}].tensors[{path!r}] must be a dict"
        out.append({"path": path, "tensor": spec})
    return out


def _merge_plus_parts(named_leaf_sets: dict[str, Any], part_names: list[Any]) -> list[dict[str, Any]]:
    assert isinstance(part_names, list)
    merged: list[dict[str, Any]] = []
    for p in part_names:
        assert isinstance(p, str), f"plus part must be a str named_leaf_sets key, got {type(p)!r}"
        block = named_leaf_sets[p]
        merged.extend(_normalize_named_leaf_block(block, block_name=p))
    return merged


def _resolve_contract_leaves(cfg: dict[str, Any], contract_name: str) -> list[dict[str, Any]]:
    contracts = cfg["contracts"]
    assert isinstance(contracts, dict), "dict_io_contracts.contracts must be a dict"
    assert contract_name in contracts, f"unknown contract {contract_name!r}"
    cspec = contracts[contract_name]
    assert isinstance(cspec, dict), f"contract {contract_name!r} must be a dict"
    leaves_spec = cspec["leaves"]
    assert isinstance(leaves_spec, dict), f"contract {contract_name!r}.leaves must be a dict"
    plus = leaves_spec["plus"]
    named = cfg["named_leaf_sets"]
    assert isinstance(named, dict), "dict_io_contracts.named_leaf_sets must be a dict"
    merged = _merge_plus_parts(named, plus)
    minus_paths = leaves_spec.get("minus_paths", [])
    if minus_paths:
        assert isinstance(minus_paths, list)
        minus_set = set(minus_paths)
        merged = [e for e in merged if e["path"] not in minus_set]
    paths_seen: set[str] = set()
    for e in merged:
        p = e["path"]
        assert isinstance(p, str)
        assert p not in paths_seen, f"duplicate path in resolved contract {contract_name!r}: {p!r}"
        paths_seen.add(p)
    return merged


def _validate_tensor_leaf(
    flags: Any,
    contract_name: str,
    *,
    path: str,
    value: Any,
    shape_spec: list[Any],
    dtype_name: str,
    tensor_fix_hint: str | None = None,
) -> None:
    exp_dtype = _dtype_from_spec(dtype_name)
    schema = _schema_tensor_line(flags, shape_spec, dtype_name, exp_dtype)
    hint_fix_yaml = tensor_fix_hint or (
        "Fix ``named_leaf_sets`` / ``tensors`` for this ``path`` in "
        "``python/src/configs/dict_io_contract_orbit.yaml`` "
        "(``shape`` list or ``dtype``), or fix env code if the contract is correct."
    )
    if not isinstance(value, torch.Tensor):
        _fail_contract_validation(
            contract_name,
            path,
            expected=schema,
            actual=f"type={type(value).__name__!r} repr={value!r}",
            hint=hint_fix_yaml,
        )
    if value.dtype != exp_dtype:
        _fail_contract_validation(
            contract_name,
            path,
            expected=schema,
            actual=_tensor_actual_line(value),
            hint=(
                f"dtype mismatch: change ``dtype`` in yaml for path {path!r} "
                f"to match runtime (or convert tensor in env). {hint_fix_yaml}"
            ),
        )
    exp_shape = _resolved_shape(flags, shape_spec)
    if tuple(value.shape) != exp_shape:
        _fail_contract_validation(
            contract_name,
            path,
            expected=schema,
            actual=_tensor_actual_line(value),
            hint=(
                f"shape mismatch: adjust ``shape`` in yaml (e.g. ``flags.agents_max_cnt``) "
                f"or env tensor construction. Resolved expected shape is {list(exp_shape)!r}. {hint_fix_yaml}"
            ),
        )


def _assert_tree_dict_tensor_leaves(contract_name: str, tree: dict[str, Any]) -> None:
    """Contract trees must be nested dicts; every leaf must be a ``torch.Tensor`` (no scalars, lists, etc.)."""

    def walk(v: Any, path_so_far: str) -> None:
        if isinstance(v, dict):
            for k, child in v.items():
                seg = f"{path_so_far}.{k}" if path_so_far else k
                walk(child, seg)
            return
        if isinstance(v, torch.Tensor):
            return
        _fail_contract_validation(
            contract_name,
            path_so_far or "<root>",
            expected="nested dict with only torch.Tensor leaves",
            actual=f"type={type(v).__name__!r} repr={v!r}",
            hint="Return tensors only under the validated dict (no Python bool/int/float/list/str leaves).",
        )

    walk(tree, "")


def validate_dict_io_contract(flags: Any, tree: dict[str, Any], contract_name: str) -> None:
    """Raise ``AssertionError`` if ``tree`` violates the resolved leaf contract."""
    raw = flags.dict_io_contracts
    assert raw is not None, "flags.dict_io_contracts is required for dict IO validation"
    cfg = _flags_to_plain(raw)
    assert isinstance(cfg, dict)
    assert dict_io_contract_validation_enabled(), (
        f"set {_ENV_DICT_IO_VALIDATE}=1 to run dict IO contract validation"
    )
    _assert_tree_dict_tensor_leaves(contract_name, tree)
    leaves = _resolve_contract_leaves(cfg, contract_name)
    for entry in leaves:
        path = entry["path"]
        value, err = _get_leaf_with_path_error(tree, path)
        if err is not None:
            _fail_contract_validation(
                contract_name,
                path,
                expected="reachable tensor leaf per contract",
                actual=err,
                hint=(
                    "Path does not exist in the returned dict tree: add the missing nested dict/key in env output, "
                    "or shorten / correct ``path`` in ``dict_io_contract_orbit.yaml``."
                ),
            )
        assert "tensor" in entry, f"{path}: leaf must have tensor spec"
        tspec = entry["tensor"]
        assert isinstance(tspec, dict)
        _validate_tensor_leaf(
            flags,
            contract_name,
            path=path,
            value=value,
            shape_spec=tspec["shape"],
            dtype_name=tspec["dtype"],
        )


def validate_dict_io_contract_step_input(flags: Any, actions: Any, contract_name: str) -> None:
    """Validate ``env.step`` / ``RewardWrapper.step`` action dict against ``contracts_step_inputs``."""
    raw = flags.dict_io_contracts
    assert raw is not None, "flags.dict_io_contracts is required for dict IO validation"
    cfg = _flags_to_plain(raw)
    assert isinstance(cfg, dict)
    assert dict_io_contract_validation_enabled(), (
        f"set {_ENV_DICT_IO_VALIDATE}=1 to run dict IO contract validation"
    )
    cis = cfg.get("contracts_step_inputs")
    assert isinstance(cis, dict), "dict_io_contracts.contracts_step_inputs must be a dict"
    assert contract_name in cis, f"unknown step input contract {contract_name!r}"
    spec = cis[contract_name]
    assert isinstance(spec, dict)
    dict_key = spec["dict_key"]
    len_key = spec["list_length_from_flags"]
    item = spec["item"]
    optional_leaves = spec.get("optional_tensor_leaves")
    optional_dict_leaves = spec.get("optional_dict_tensor_leaves")
    optional_any_leaves = spec.get("optional_any_leaves")
    assert isinstance(dict_key, str) and isinstance(len_key, str)
    assert isinstance(item, dict)
    n = int(getattr(flags, len_key))
    assert n >= 1
    assert isinstance(actions, dict), (
        f"step input contract {contract_name!r}: actions must be dict, got {type(actions).__name__!r}"
    )
    if optional_leaves is None and optional_dict_leaves is None and optional_any_leaves is None:
        assert set(actions.keys()) == {dict_key}, (
            f"step input contract {contract_name!r}: actions keys must be exactly {{{dict_key!r}}}, "
            f"got {sorted(actions.keys())!r}"
        )
    else:
        if optional_leaves is not None:
            assert isinstance(optional_leaves, dict)
            optional_leaf_keys = set(optional_leaves.keys())
        else:
            optional_leaf_keys = set()
        if optional_dict_leaves is not None:
            assert isinstance(optional_dict_leaves, dict)
            optional_dict_leaf_keys = set(optional_dict_leaves.keys())
        else:
            optional_dict_leaf_keys = set()
        if optional_any_leaves is not None:
            assert isinstance(optional_any_leaves, list)
            optional_any_leaf_keys = {x for x in optional_any_leaves if isinstance(x, str)}
            assert len(optional_any_leaf_keys) == len(optional_any_leaves), (
                "optional_any_leaves must be a list of unique str keys"
            )
        else:
            optional_any_leaf_keys = set()
        allowed_keys = {dict_key} | optional_leaf_keys | optional_dict_leaf_keys | optional_any_leaf_keys
        assert set(actions.keys()) <= allowed_keys, (
            f"step input contract {contract_name!r}: actions keys must be subset of {sorted(allowed_keys)!r}, "
            f"got {sorted(actions.keys())!r}"
        )
        assert dict_key in actions
    lst = actions[dict_key]
    assert isinstance(lst, list), (
        f"step input contract {contract_name!r}: {dict_key!r} must be a list, got {type(lst).__name__!r}"
    )
    assert len(lst) == n, (
        f"step input contract {contract_name!r}: len({dict_key}) must be flags.{len_key}={n}, got {len(lst)}"
    )
    shape_spec = item["shape"]
    dtype_name = item["dtype"]
    assert isinstance(shape_spec, list) and isinstance(dtype_name, str)
    cpu_only = bool(item.get("cpu_only", False))
    step_hint = (
        "Fix ``contracts_step_inputs`` / ``item`` for this contract in "
        "``python/src/configs/dict_io_contract_orbit.yaml`` "
        "(``shape``, ``dtype``, ``cpu_only``), or fix the caller."
    )
    for i, t in enumerate(lst):
        path = f"step_input.{dict_key}[{i}]"
        _validate_tensor_leaf(
            flags,
            contract_name,
            path=path,
            value=t,
            shape_spec=shape_spec,
            dtype_name=dtype_name,
            tensor_fix_hint=step_hint,
        )
        if cpu_only:
            assert isinstance(t, torch.Tensor) and not t.is_cuda, (
                f"step input contract {contract_name!r}: {path} must be CPU tensor"
            )

    if optional_leaves is not None:
        for ok, oval in actions.items():
            if ok == dict_key:
                continue
            if optional_any_leaves is not None and ok in optional_any_leaves:
                continue
            if ok not in optional_leaves:
                continue
            leaf_spec = optional_leaves[ok]
            assert isinstance(leaf_spec, dict)
            shape_spec = leaf_spec["shape"]
            dtype_name = leaf_spec["dtype"]
            assert isinstance(shape_spec, list) and isinstance(dtype_name, str)
            cpu_only_opt = bool(leaf_spec.get("cpu_only", False))
            path_o = f"step_input.{ok}"
            _validate_tensor_leaf(
                flags,
                contract_name,
                path=path_o,
                value=oval,
                shape_spec=shape_spec,
                dtype_name=dtype_name,
                tensor_fix_hint=step_hint,
            )
            if cpu_only_opt:
                assert isinstance(oval, torch.Tensor) and not oval.is_cuda, (
                    f"step input contract {contract_name!r}: {path_o} must be CPU tensor"
                )
    if optional_dict_leaves is not None:
        for ok, oval in actions.items():
            if ok == dict_key:
                continue
            if optional_any_leaves is not None and ok in optional_any_leaves:
                continue
            if optional_leaves is not None and ok in optional_leaves:
                continue
            assert ok in optional_dict_leaves, (ok, sorted(optional_dict_leaves.keys()))
            dict_spec = optional_dict_leaves[ok]
            assert isinstance(dict_spec, dict)
            value_spec = dict_spec["value"]
            assert isinstance(value_spec, dict)
            shape_spec = value_spec["shape"]
            dtype_name = value_spec["dtype"]
            assert isinstance(shape_spec, list) and isinstance(dtype_name, str)
            cpu_only_opt = bool(value_spec.get("cpu_only", False))
            path_o = f"step_input.{ok}"
            assert isinstance(oval, dict), (
                f"step input contract {contract_name!r}: {path_o} must be dict[str, Tensor], got {type(oval).__name__!r}"
            )
            for subk, subv in oval.items():
                assert isinstance(subk, str), (
                    f"step input contract {contract_name!r}: {path_o} keys must be str, got {type(subk).__name__!r}"
                )
                path_d = f"{path_o}.{subk}"
                _validate_tensor_leaf(
                    flags,
                    contract_name,
                    path=path_d,
                    value=subv,
                    shape_spec=shape_spec,
                    dtype_name=dtype_name,
                    tensor_fix_hint=step_hint,
                )
                if cpu_only_opt:
                    assert isinstance(subv, torch.Tensor) and not subv.is_cuda, (
                        f"step input contract {contract_name!r}: {path_d} must be CPU tensor"
                    )


def maybe_validate_dict_io_contract_step_input(flags: Any, actions: Any, contract_name: str) -> None:
    if flags is None:
        return
    if not dict_io_contract_validation_enabled():
        return
    raw = flags.dict_io_contracts
    if raw is None:
        return
    cfg = _flags_to_plain(raw)
    if not isinstance(cfg, dict):
        return
    validate_dict_io_contract_step_input(flags, actions, contract_name)


def maybe_validate_dict_io_contract(flags: Any, tree: dict[str, Any], contract_name: str) -> None:
    if not dict_io_contract_validation_enabled():
        return
    raw = flags.dict_io_contracts
    if raw is None:
        return
    cfg = _flags_to_plain(raw)
    if not isinstance(cfg, dict):
        return
    validate_dict_io_contract(flags, tree, contract_name)


def validated_dict_io_contract_output(
    flags: Any | None,
    tree: dict[str, Any],
    contract_name: str,
) -> dict[str, Any]:
    """If ``flags`` is set and ``IMPALA_DICT_IO_VALIDATE=1``, validate ``tree``; always return ``tree``."""
    if flags is not None:
        maybe_validate_dict_io_contract(flags, tree, contract_name)
    return tree
