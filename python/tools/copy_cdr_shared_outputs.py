from __future__ import annotations

import ast
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "python/src/configs/impala_x1_rl_cdr.py"
SHARED_OUTPUTS_PREFIX = "shared_outputs/"
OUTPUTS_PREFIX = "outputs/"


def _shared_output_paths_from_config(config_path: Path) -> list[Path]:
    tree = ast.parse(config_path.read_text())
    paths: list[Path] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue

        raw_path = node.value
        if not raw_path.startswith(SHARED_OUTPUTS_PREFIX):
            continue
        if raw_path in seen:
            continue

        seen.add(raw_path)
        paths.append(Path(raw_path))

    assert paths, f"No {SHARED_OUTPUTS_PREFIX!r} paths found in {config_path}"
    return paths


def main() -> None:
    shared_output_paths = _shared_output_paths_from_config(CONFIG_PATH)

    missing_sources: list[Path] = []
    for shared_output_path in shared_output_paths:
        source_path = PROJECT_ROOT / OUTPUTS_PREFIX / shared_output_path.relative_to("shared_outputs")
        if not source_path.is_file():
            missing_sources.append(source_path)

    if missing_sources:
        missing_text = "\n".join(str(path) for path in missing_sources)
        raise FileNotFoundError(f"Missing source files:\n{missing_text}")

    for shared_output_path in shared_output_paths:
        source_path = PROJECT_ROOT / OUTPUTS_PREFIX / shared_output_path.relative_to("shared_outputs")
        destination_path = PROJECT_ROOT / shared_output_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        print(f"{source_path} -> {destination_path}")

    print(f"Copied {len(shared_output_paths)} files.")


if __name__ == "__main__":
    main()
