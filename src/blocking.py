from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


COMMON_SUFFIX_REPLACEMENTS = {
    "pvt": "private",
    "private ltd": "private limited",
    "pvt ltd": "private limited",
    "private limited": "private limited",
    "llp": "llp",
    "ltd": "limited",
    "limited": "limited",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "llc": "llc",
    "co": "company",
    "company": "company",
    "plc": "plc",
}


def normalize_business_name(value: object) -> str:
    """Normalize noisy business names into a stable string for blocking.

    The objective is not perfect semantic normalization, but creating a consistent
    representation that preserves real-name equivalence while removing mostly
    cosmetic noise. This is intentionally separated from the full ML feature
    engineering pipeline.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = text.replace("+", " ")
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for old, new in COMMON_SUFFIX_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    # Remove trailing common legal suffix tokens that create near-duplicate names
    text = re.sub(r"\b(private limited|limited|incorporated|corporation|llc|company|llp|plc)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_source_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "entity_id" not in df.columns:
        raise ValueError(f"Missing entity_id column in {path}")
    return df


@dataclass
class ExactNameBlocker:
    """Exact normalized-name blocking using a dictionary-backed inverted index."""

    def build_name_index(self, source_df: pd.DataFrame) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = defaultdict(list)
        for row in source_df.itertuples(index=False):
            entity_id = str(getattr(row, "entity_id", "")).strip()
            name = normalize_business_name(getattr(row, "business_name", ""))
            if not entity_id or not name:
                continue
            index[name].append(entity_id)
        return {k: sorted(set(v)) for k, v in index.items()}

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> List[Tuple[str, str, str, str]]:
        index = self.build_name_index(pd.concat([s2_df, s3_df], ignore_index=True, sort=False))

        rows: List[Tuple[str, str, str, str]] = []
        for row in s1_df.itertuples(index=False):
            s1_id = str(getattr(row, "entity_id", "")).strip()
            if not s1_id:
                continue

            norm_name = normalize_business_name(getattr(row, "business_name", ""))
            if not norm_name:
                continue

            matching_ids = index.get(norm_name, [])
            for cand_id in matching_ids:
                if cand_id.startswith("S2-"):
                    source = "source2"
                elif cand_id.startswith("S3-"):
                    source = "source3"
                else:
                    continue
                rows.append((s1_id, cand_id, source, "exact_name"))

        deduped: Set[Tuple[str, str, str, str]] = set()
        for row in rows:
            deduped.add(row)
        return sorted(deduped)


def parse_ground_truth(path: str | Path) -> Dict[str, Set[str]]:
    gt = pd.read_csv(path, sep="\t")
    gt = gt[["source1_entity_id", "matched_entity_ids"]].dropna(subset=["source1_entity_id"])
    mapping: Dict[str, Set[str]] = {}
    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        raw = str(row.matched_entity_ids).strip() if not pd.isna(row.matched_entity_ids) else ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        mapping[s1_id] = matches
    return mapping


def evaluate_exact_name_blocker(train_dir: str | Path) -> Dict[str, object]:
    """Evaluate exact-name blocking on the training data using ground truth."""
    train_dir = Path(train_dir)
    s1_df = load_source_file(train_dir / "train_source1.tsv")
    s2_df = load_source_file(train_dir / "train_source2.tsv")
    s3_df = load_source_file(train_dir / "train_source3.tsv")
    gt = parse_ground_truth(train_dir / "train_ground_truth.tsv")

    start = time.perf_counter()
    blocker = ExactNameBlocker()
    candidate_rows = blocker.generate_candidates(s1_df, s2_df, s3_df)
    elapsed = time.perf_counter() - start

    candidate_map: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, cand_id, _, _ in candidate_rows:
        candidate_map[s1_id].add(cand_id)

    total_true_matches = 0
    total_recovered = 0
    missed_by_s1: Dict[str, List[str]] = {}

    for s1_id in sorted(gt):
        true_matches = gt[s1_id]
        total_true_matches += len(true_matches)
        recovered = true_matches & candidate_map.get(s1_id, set())
        total_recovered += len(recovered)
        missed = sorted(true_matches - candidate_map.get(s1_id, set()))
        if missed:
            missed_by_s1[s1_id] = missed

    counts = [len(candidate_map.get(s1_id, set())) for s1_id in s1_df["entity_id"].tolist()]
    total_s1 = len(s1_df)
    total_pairs = sum(len(v) for v in candidate_map.values())

    metrics = {
        "num_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "average_candidates_per_s1": total_pairs / total_s1 if total_s1 else 0.0,
        "median_candidates_per_s1": statistics.median(counts) if counts else 0.0,
        "max_candidates_per_s1": max(counts) if counts else 0.0,
        "candidate_recall": (total_recovered / total_true_matches) if total_true_matches else 1.0,
        "missed_true_matches": total_true_matches - total_recovered,
        "missed_by_s1": missed_by_s1,
        "runtime_seconds": elapsed,
        "candidate_count_distribution": counts,
        "candidate_rows": candidate_rows,
    }
    return metrics


def _print_metrics(metrics: Dict[str, object]) -> None:
    print("Exact Name Blocking Evaluation")
    print("=" * 60)
    print(f"Source 1 entities: {metrics['num_source1_entities']}")
    print(f"Total candidate pairs: {metrics['total_candidate_pairs']}")
    print(f"Average candidates / S1: {metrics['average_candidates_per_s1']:.3f}")
    print(f"Median candidates / S1: {metrics['median_candidates_per_s1']}")
    print(f"Max candidates / S1: {metrics['max_candidates_per_s1']}")
    print(f"Candidate recall: {metrics['candidate_recall']:.4f}")
    print(f"Missed true matches: {metrics['missed_true_matches']}")
    print(f"Runtime: {metrics['runtime_seconds']:.4f}s")

    missed = metrics["missed_by_s1"]
    if missed:
        print("\nMisses by S1 (showing up to 10):")
        for s1_id, ids in list(missed.items())[:10]:
            print(f"  {s1_id}: {', '.join(ids)}")
    else:
        print("\nNo missed true matches.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "student_resource" / "dataset" / "train"
    metrics = evaluate_exact_name_blocker(train_dir)
    _print_metrics(metrics)
