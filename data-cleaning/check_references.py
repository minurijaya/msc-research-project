"""Check which IDs referenced in train.csv are missing from dataset.csv."""

import csv
from collections import defaultdict
from pathlib import Path

TRAIN = Path(__file__).parent / "train.csv"
DATASET = Path(__file__).parent / "dataset.csv"

# Load all known IDs from dataset.csv
known_ids = set()
with DATASET.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        known_ids.add(row["ID"].strip())

# Scan train.csv for references
missing: dict[str, list[tuple[int, str]]] = defaultdict(list)  # id -> [(line, col)]
total_refs = 0

with TRAIN.open() as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    for lineno, row in enumerate(reader, start=2):
        for col in columns:
            val = row[col].strip()
            total_refs += 1
            if val not in known_ids:
                missing[val].append((lineno, col))

# ── Stats ────────────────────────────────────────────────────────────────────
print(f"Dataset IDs   : {len(known_ids):>6}")
print(f"Train rows    : {lineno - 1:>6}")
print(f"Total refs    : {total_refs:>6}  ({len(columns)} cols × rows)")
print(f"Missing IDs   : {len(missing):>6}  (unique)")
print(f"Missing refs  : {sum(len(v) for v in missing.values()):>6}  (total occurrences)")
print()

print(f"{'Missing ID':<15}  {'Occurrences':>11}  First seen at")
print("-" * 55)
for mid, locs in sorted(missing.items()):
    first_line, first_col = locs[0]
    print(f"{mid:<15}  {len(locs):>11}  line {first_line}, col '{first_col}'")
