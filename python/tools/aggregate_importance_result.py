from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_OUTPUT_NAME = "aggregate_importance_result.txt"
_SCORE_WEIGHTS: dict[str, float] = {
    "loss_self_rl_value_smooth": 4.0,
}


def _importance_files(directory: Path, output_name: str) -> list[Path]:
    root = directory.expanduser().resolve()
    assert root.is_dir(), root
    out = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix == ".txt" and p.name != output_name
    )
    assert len(out) >= 1, root
    return out


def _importance_sort_reverse(path: Path) -> bool:
    stem = path.stem
    if stem.startswith("loss_"):
        return True
    if stem.startswith("accuracy_"):
        return False
    raise AssertionError(f"unknown importance metric direction: {path.name}")


def _score_weight(path: Path) -> float:
    out = float(_SCORE_WEIGHTS.get(path.stem, 1.0))
    assert out > 0.0, (path, out)
    return out


def _parse_importance_file(path: Path) -> dict[str, int]:
    values: dict[str, float] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        assert len(parts) == 3, (path, line_no, stripped)
        feature = parts[0]
        float(parts[1])
        delta = float(parts[2])
        assert feature not in values, (path, feature)
        values[feature] = delta
    assert len(values) >= 1, path

    if _importance_sort_reverse(path):
        ranked_features = sorted(values, key=lambda feature: (-values[feature], feature))
    else:
        ranked_features = sorted(values, key=lambda feature: (values[feature], feature))
    ranks = {feature: rank for rank, feature in enumerate(ranked_features, start=1)}
    return ranks


def _aggregate(paths: list[Path]) -> tuple[list[str], dict[str, list[int]], list[tuple[str, float]]]:
    score_names = [p.stem for p in paths]
    score_weights = [_score_weight(p) for p in paths]
    score_weight_sum = sum(score_weights)
    assert score_weight_sum > 0.0, score_weights
    rank_by_score = [_parse_importance_file(p) for p in paths]
    expected_features = set(rank_by_score[0])
    assert len(expected_features) >= 1, paths[0]
    for path, ranks in zip(paths, rank_by_score, strict=True):
        assert set(ranks) == expected_features, (
            path,
            sorted(expected_features - set(ranks)),
            sorted(set(ranks) - expected_features),
        )

    ranks_by_feature: dict[str, list[int]] = {}
    mean_rank_by_feature: list[tuple[str, float]] = []
    for feature in sorted(expected_features):
        ranks = [score_ranks[feature] for score_ranks in rank_by_score]
        ranks_by_feature[feature] = ranks
        weighted_rank_sum = sum(
            rank * weight for rank, weight in zip(ranks, score_weights, strict=True)
        )
        mean_rank_by_feature.append((feature, weighted_rank_sum / score_weight_sum))

    mean_rank_by_feature.sort(key=lambda x: (x[1], x[0]))
    return score_names, ranks_by_feature, mean_rank_by_feature


def _write_table(
    output_path: Path,
    *,
    score_names: list[str],
    ranks_by_feature: dict[str, list[int]],
    mean_rank_by_feature: list[tuple[str, float]],
) -> None:
    header = ["feature", "mean_rank", *score_names]
    rows = ["\t".join(header)]
    for feature, mean_rank in mean_rank_by_feature:
        ranks = ranks_by_feature[feature]
        rows.append(
            "\t".join(
                [
                    feature,
                    f"{mean_rank:.6f}",
                    *(str(rank) for rank in ranks),
                ]
            )
        )
    output_path.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate feature importance ranks from one directory of .txt files."
    )
    parser.add_argument("importance_dir", help="Directory containing feature importance .txt files")
    parser.add_argument(
        "--output_name",
        default=_DEFAULT_OUTPUT_NAME,
        help=f"Output filename written inside importance_dir; default: {_DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args()

    importance_dir = Path(args.importance_dir).expanduser().resolve()
    output_name = str(args.output_name)
    assert output_name == Path(output_name).name, output_name
    paths = _importance_files(importance_dir, output_name)
    score_names, ranks_by_feature, mean_rank_by_feature = _aggregate(paths)

    output_path = importance_dir / output_name
    _write_table(
        output_path,
        score_names=score_names,
        ranks_by_feature=ranks_by_feature,
        mean_rank_by_feature=mean_rank_by_feature,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
