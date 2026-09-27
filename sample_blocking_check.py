import argparse
from pathlib import Path

import pandas as pd

from src.blocking import EnhancedNameAddressBlocker, parse_ground_truth


def main():
    parser = argparse.ArgumentParser(description='Run a small blocking validation sample.')
    parser.add_argument('--size', type=int, default=100, help='Number of rows to sample from each source.')
    args = parser.parse_args()

    base = Path('student_resource/dataset/train')
    print(f"Loading {args.size} sample rows from datasets...", flush=True)
    s1 = pd.read_csv(base / 'train_source1.tsv', sep='\t').head(args.size)
    s2 = pd.read_csv(base / 'train_source2.tsv', sep='\t').head(args.size)
    s3 = pd.read_csv(base / 'train_source3.tsv', sep='\t').head(args.size)
    print("Loading ground truth dataset...", flush=True)
    gt = parse_ground_truth(base / 'train_ground_truth.tsv')

    print("Generating candidate pairs...", flush=True)
    blocker = EnhancedNameAddressBlocker()
    rows = blocker.generate_candidates(s1, s2, s3)

    candidate_map = {}
    for s1_id, cand_id, _, _ in rows:
        candidate_map.setdefault(s1_id, set()).add(cand_id)

    total_true = 0
    total_recovered = 0
    for s1_id, true_matches in gt.items():
        if s1_id not in s1['entity_id'].values:
            continue
        total_true += len(true_matches)
        total_recovered += len(true_matches & candidate_map.get(s1_id, set()))

    print('sample_s1:', len(s1), flush=True)
    print('sample_s2:', len(s2), flush=True)
    print('sample_s3:', len(s3), flush=True)
    print('candidate_pairs:', len(rows), flush=True)
    print('recall:', round((total_recovered / total_true) if total_true else 0.0, 4), flush=True)
    print('sample_rows:', rows[:10], flush=True)


if __name__ == '__main__':
    main()
