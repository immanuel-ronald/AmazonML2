from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.blocking import (
    EnhancedNameAddressBlocker,
    _business_name_variants,
    _generate_candidates_for_chunk,
    load_source_file,
    normalize_address,
    normalize_business_name,
    normalize_country,
)


def write_candidate_rows(s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path, chunk_size: int = 100_000) -> int:
    blocker = EnhancedNameAddressBlocker()

    name_index: dict[str, list[str]] = defaultdict(list)
    address_index: dict[str, list[str]] = defaultdict(list)
    entity_country: dict[str, str] = {}
    entity_source: dict[str, str] = {}

    print("Building blocking indexes from Source 2 and Source 3...", flush=True)
    for source_path in (s2_path, s3_path):
        print(f"Reading {source_path.name}...", flush=True)
        for chunk in pd.read_csv(source_path, sep="\t", dtype=str, chunksize=chunk_size):
            for row in chunk.itertuples(index=False):
                entity_id = str(getattr(row, "entity_id", "")).strip()
                if not entity_id:
                    continue
                country = normalize_country(getattr(row, "country", ""))
                entity_country[entity_id] = country
                if entity_id.startswith("S2-"):
                    entity_source[entity_id] = "source2"
                elif entity_id.startswith("S3-"):
                    entity_source[entity_id] = "source3"

                raw_name = getattr(row, "business_name", "")
                for variant in _business_name_variants(raw_name):
                    name_index[variant].append(entity_id)

                addr = normalize_address(getattr(row, "business_address", ""))
                if addr:
                    address_index[addr].append(entity_id)

    name_index = {k: sorted(set(v)) for k, v in name_index.items()}
    address_index = {k: sorted(set(v)) for k, v in address_index.items()}
    print(f"Indexes built. Unique name keys: {len(name_index)}, unique address keys: {len(address_index)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    print(f"Generating candidate pairs for {s1_path.name}...", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=chunk_size):
            chunk_rows = _generate_candidates_for_chunk(chunk, name_index, address_index, entity_country, entity_source)
            candidate_map: dict[str, set[str]] = defaultdict(set)
            for s1_id, cand_id, _, _ in chunk_rows:
                candidate_map[s1_id].add(cand_id)

            for s1_id in chunk["entity_id"].astype(str).tolist():
                matches = sorted(candidate_map.get(s1_id, set()))
                writer.writerow([s1_id, ",".join(matches)])
                rows_written += 1
            print(f"Processed {rows_written} Source 1 entities...", flush=True)

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate pairs from the blocking stage.")
    parser.add_argument("--source1", type=Path, default=Path("student_resource/dataset/test/test_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("student_resource/dataset/test/test_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("student_resource/dataset/test/test_source3.tsv"))
    parser.add_argument("--output", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    rows_written = write_candidate_rows(args.source1, args.source2, args.source3, args.output, chunk_size=args.chunk_size)
    print(f"Done! Wrote {rows_written} Source 1 rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
