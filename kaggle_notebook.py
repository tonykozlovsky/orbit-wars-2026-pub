import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/aoti_repro")
TIMEOUT_SECONDS = 90


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _find_submission_root() -> Path:
    unpacked_candidates = tuple(
        p.parents[1]
        for p in sorted(INPUT_ROOT.rglob("kaggle_submission/submission.py"))
        if (p.parent / "checkpoint_policy_logits_aoti_2p.so").is_file()
        and (p.parent / "checkpoint_policy_logits_aoti_4p.so").is_file()
        and tuple(p.parent.glob("aoti_policy_runner_cpp*.so"))
    )
    assert len(unpacked_candidates) == 1, unpacked_candidates
    return unpacked_candidates[0]


def _run_probe(name: str, code: str, args: tuple[str, ...]) -> None:
    print(f"\n===== {name} start =====", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, _ = proc.communicate()
    print(out, end="", flush=True)
    elapsed = time.perf_counter() - t0
    print(
        f"===== {name} done returncode={proc.returncode} timed_out={timed_out} "
        f"elapsed={_fmt(elapsed)} =====",
        flush=True,
    )


if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
WORK_ROOT.mkdir(parents=True)
SUBMISSION_ROOT = _find_submission_root()
print(f"submission_root={SUBMISSION_ROOT}", flush=True)

for path in sorted(SUBMISSION_ROOT.rglob("*")):
    if path.is_file() and (
        path.name.endswith((".pt2", ".so", ".py"))
        or path.name.startswith("checkpoint_policy_logits_aoti")
        or path.name.startswith("aoti_policy_runner_cpp")
    ):
        print(f"file={path.relative_to(SUBMISSION_ROOT)} bytes={path.stat().st_size}", flush=True)


SUBMISSION_GET_MODEL_PROBE = r"""
import importlib
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

print("probe=submission_get_model import_start", flush=True)
import_t0 = time.perf_counter()
submission = importlib.import_module("kaggle_submission.submission")
print(f"probe=submission_get_model import_done elapsed={time.perf_counter() - import_t0:.6f}s", flush=True)
print(f"aoti_2p={submission._submission_model_artifact_paths(2)[3]}", flush=True)
print(f"aoti_4p={submission._submission_model_artifact_paths(4)[3]}", flush=True)

for num_agents in (2, 4):
    load_t0 = time.perf_counter()
    print(f"probe=submission_get_model load_start num_agents={num_agents}", flush=True)
    loaded_model = submission._get_model(num_agents)
    print(
        f"probe=submission_get_model load_done num_agents={num_agents} "
        f"elapsed={time.perf_counter() - load_t0:.6f}s",
        flush=True,
    )
    print(f"num_agents={num_agents} artifact_kind={loaded_model.artifact_kind}", flush=True)
"""


CUSTOM_RUNNER_SO_PROBE = r"""
import importlib.util
import sys
import time
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
pkg_dir = root / "kaggle_submission"
print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

runner_exts = tuple(sorted(pkg_dir.glob("aoti_policy_runner_cpp*.so")))
model_sos = tuple(sorted(pkg_dir.glob("checkpoint_policy_logits_aoti*.so")))
print(f"runner_exts={runner_exts}", flush=True)
print(f"model_sos={model_sos}", flush=True)
assert len(runner_exts) == 1, runner_exts
assert len(model_sos) == 2, model_sos

print(f"runner_ext={runner_exts[0]}", flush=True)

spec = importlib.util.spec_from_file_location("aoti_policy_runner_cpp", str(runner_exts[0]))
assert spec is not None and spec.loader is not None, runner_exts[0]
module = importlib.util.module_from_spec(spec)
print("probe=custom_runner exec_module_start", flush=True)
import_t0 = time.perf_counter()
spec.loader.exec_module(module)
print(f"probe=custom_runner exec_module_done elapsed={time.perf_counter() - import_t0:.6f}s", flush=True)
assert hasattr(module, "AOTIPolicyRunnerCpu"), runner_exts[0]

for model_so in model_sos:
    print(f"model_so={model_so}", flush=True)
    load_t0 = time.perf_counter()
    print("probe=custom_runner runner_init_start", flush=True)
    runner = module.AOTIPolicyRunnerCpu(str(model_so))
    print(
        f"probe=custom_runner runner_init_done elapsed={time.perf_counter() - load_t0:.6f}s",
        flush=True,
    )
    print(f"runner_type={type(runner)}", flush=True)
"""


_run_probe("submission_get_model", SUBMISSION_GET_MODEL_PROBE, (str(SUBMISSION_ROOT),))
_run_probe("custom_runner_so", CUSTOM_RUNNER_SO_PROBE, (str(SUBMISSION_ROOT),))