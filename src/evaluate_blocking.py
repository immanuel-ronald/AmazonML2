from __future__ import annotations

from pathlib import Path

from blocking import evaluate_exact_name_blocker


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "student_resource" / "dataset" / "train"
    metrics = evaluate_exact_name_blocker(train_dir)
    print("\n=== Blocking evaluation summary ===")
    print(f"Candidate recall: {metrics['candidate_recall']:.4f}")
    print(f"Avg candidates per S1: {metrics['average_candidates_per_s1']:.3f}")
    print(f"Median candidates per S1: {metrics['median_candidates_per_s1']}")
    print(f"Max candidates per S1: {metrics['max_candidates_per_s1']}")
    print(f"Total candidate pairs: {metrics['total_candidate_pairs']}")
    print(f"Runtime (s): {metrics['runtime_seconds']:.4f}")
    print(f"Missed true matches: {metrics['missed_true_matches']}")
