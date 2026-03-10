"""
Extract product images from Column C ('Product Image') in all .numbers dataset
files under data/raw/ and save them to data/cleaned/Images_1/.

Image naming convention: {dataset_id:02d}-{no:05d}.jpeg
  e.g. Dataset 1, No=25  -> 01-00025.jpeg
       Dataset 2, No=437 -> 02-00437.jpeg

Dataset index is extracted from the numeric part of the filename:
  'Dataset 1.numbers'      -> 01
  'Dataset 2.xlsx.numbers' -> 02
"""

import os
import re
import shutil
import tempfile
import zipfile

from numbers_parser import Document
from numbers_parser.model import PACKAGE_ID


def get_dataset_index(filename: str) -> int | None:
    """Extract the dataset number from a filename like 'Dataset 1.numbers'."""
    match = re.search(r'Dataset\s+(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_images_from_numbers(filepath: str, dataset_idx: int, output_dir: str) -> int:
    """
    Extract images from Column C of a .numbers file.

    Images are stored as floating ImageArchive objects sorted by Y-position,
    which corresponds to the row order in the table.

    Returns the number of images extracted.
    """
    print(f"  Loading {os.path.basename(filepath)} ...")

    # Extract the zip archive to a temp directory for direct file access
    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            z.extractall(tmpdir)
        data_dir = os.path.join(tmpdir, 'Data')

        doc = Document(filepath)
        sheet = doc.sheets[0]
        table = sheet.tables[0]
        model = table._model
        rows = table.rows()

        num_data_rows = table.num_rows - 1  # exclude header row

        # Verify Column C is 'Product Image'
        col_c_header = rows[0][2].value
        if col_c_header not in ('Product Image', 'product image'):
            print(f"  WARNING: Column C header is '{col_c_header}', expected 'Product Image'")

        # Build identifier -> filename map from package metadata
        pkg = model.objects[PACKAGE_ID]
        id_to_file = {d.identifier: d.file_name for d in pkg.datas if d.file_name}

        # Collect all ImageArchive objects and sort by Y position (= row order)
        image_archives = [
            v for v in model.objects._objects.values()
            if type(v).__name__ == 'ImageArchive'
        ]

        images_sorted = []
        for img in image_archives:
            if not img.HasField('data'):
                continue
            data_id = img.data.identifier
            fname = id_to_file.get(data_id)
            if fname is None:
                continue
            y = img.super.geometry.position.y
            images_sorted.append((y, data_id, fname))

        images_sorted.sort(key=lambda t: t[0])

        extracted = 0
        for i, (y, data_id, src_fname) in enumerate(images_sorted):
            row_idx = i + 1  # 1-indexed, skipping header
            if row_idx > num_data_rows:
                break  # ignore any extra floating images beyond the table rows

            # Read the No value from Column A
            no_val = rows[row_idx][0].value
            if no_val is None:
                print(f"  WARNING: Row {row_idx} has no 'No' value, skipping")
                continue

            no_int = int(no_val)
            ext = os.path.splitext(src_fname)[1]  # .jpeg / .jpg / .png
            out_name = f"{dataset_idx:02d}-{no_int:05d}{ext}"
            out_path = os.path.join(output_dir, out_name)

            src_path = os.path.join(data_dir, src_fname)
            if not os.path.exists(src_path):
                print(f"  WARNING: Source file not found: {src_fname}")
                continue

            shutil.copy2(src_path, out_path)
            extracted += 1

        return extracted

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    raw_dir = os.path.join('data', 'raw')
    output_dir = os.path.join('data', 'cleaned', 'Images_1')
    os.makedirs(output_dir, exist_ok=True)

    # Find all .numbers files and sort by dataset index
    numbers_files = []
    for fname in os.listdir(raw_dir):
        if fname.endswith('.numbers'):
            idx = get_dataset_index(fname)
            if idx is not None:
                numbers_files.append((idx, fname))
            else:
                print(f"Skipping (no dataset number found): {fname}")

    numbers_files.sort()

    if not numbers_files:
        print("No .numbers files found in data/raw/")
        return

    total = 0
    for dataset_idx, fname in numbers_files:
        filepath = os.path.join(raw_dir, fname)
        print(f"Dataset {dataset_idx:02d}: {fname}")
        count = extract_images_from_numbers(filepath, dataset_idx, output_dir)
        print(f"  -> Extracted {count} images")
        total += count

    print(f"\nDone. Total images extracted: {total}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
