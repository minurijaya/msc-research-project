
import csv
import re
import shutil
from pathlib import Path

INPUT = Path(__file__).parent / "train.csv"
OUTPUT = Path(__file__).parent / "train_fixed.csv"

ID_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def fix_id(value: str) -> tuple[str, bool]:
    m = ID_PATTERN.match(value)
    if not m:
        raise ValueError(f"Unrecognisable ID format: {repr(value)}")
    xx = m.group(1).lstrip("0") or "0"
    yyyyy = m.group(2).lstrip("0") or "0"
    fixed = f"{int(xx):02d}-{int(yyyyy):05d}"
    return fixed, fixed != value


def main() -> None:
    fixed_count = 0
    rows_out = []

    with INPUT.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_out.append(header)
        for lineno, row in enumerate(reader, start=2):
            new_row = []
            for val in row:
                fixed, changed = fix_id(val)
                if changed:
                    print(f"  line {lineno}: {repr(val)} -> {repr(fixed)}")
                    fixed_count += 1
                new_row.append(fixed)
            rows_out.append(new_row)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

  

if __name__ == "__main__":
    main()
