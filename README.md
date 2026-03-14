# CLIP-Based Fashion Recommendation Engine Walkthrough

This document outlines how to use the implemented Fashion Recommendation Engine. The system leverages OpenAI's CLIP model with a custom Fusion Module and Triplet Loss optimization.

## 1. Environment Setup

The project uses **Poetry** for dependency management.

```bash
# Install dependencies
poetry install

# Activate virtual environment
source .venv/bin/activate
```

## 2. Data Preparation

### Data Catalog CSV (required for triplet training and inference)

The catalog maps each item ID to its image path and caption. Required columns:

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| `ID`      | Unique item identifier (referenced by triplets)  |
| `Image`   | Relative path to the image (from `--image_dir`)  |
| `Caption` | Text description of the fashion item             |

Example `dataset.csv`:
```csv
ID,Image,Caption
ANC,shirts/shirt_01.jpg,Red cotton shirt classic fit
POS,shirts/shirt_02.jpg,Reddish linen shirt relaxed fit
NEG,dresses/dress_01.jpg,Blue floral summer dress
```

### Triplet CSV (for explicit triplet training)

References item IDs from the catalog. Required columns:

| Column           | Description             |
|------------------|-------------------------|
| `anchor_image`   | ID of the anchor item   |
| `positive_image` | ID of the positive item |
| `negative_image` | ID of the negative item |

Also accepts `anchor_image_id`, `positive_image_id`, `negative_image_id` as aliases.

Example `triplets.csv`:
```csv
anchor_image,positive_image,negative_image
ANC,POS,NEG
ANC,POS,NEG
```

### Siamese Training CSV (alternative, no catalog needed)

For implicit negative mining from the batch. Required columns:
- `image_path`: Relative path to the image (from `--image_dir`)
- `caption`: Text description of the item

Example `train.csv`:
```csv
image_path,caption
dresses/dress_01.jpg,Red summer floral dress with V-neck
shirts/shirt_05.jpg,Blue denim shirt casual fit
```

## 3. Extracting Images from Raw Data

If your source data is in Apple Numbers format, extract product images first:

```bash
python scripts/extract_images.py
```

This reads all `.numbers` files from `data/raw/` and saves images to `data/cleaned/Images_1/` using the naming convention `{dataset_id:02d}-{no:05d}.jpeg`.

## 4. Running Zero-Shot Baseline

Evaluate the pretrained CLIP model without any fine-tuning.

```bash
python scripts/evaluate_zero_shot.py \
    --image_dir path/to/images \
    --metadata_path path/to/val.csv \
    --batch_size 32
```

## 5. Training the Model

### Explicit Triplet Training (recommended)

Fine-tune using your triplet CSV and a data catalog. Both files are required together.

```bash
python scripts/train.py \
    --image_dir path/to/images \
    --triplets_csv path/to/triplets.csv \
    --catalog_csv path/to/dataset.csv \
    --output_dir checkpoints_triplet \
    --epochs 10
```

Default `--catalog_csv` is `data/Cleaned/dataset.csv`.

### Siamese Training (implicit negatives from batch)

Fine-tune using batch hard triplet loss without an explicit triplet file.

```bash
python scripts/train.py \
    --image_dir path/to/images \
    --train_csv path/to/train.csv \
    --output_dir checkpoints \
    --epochs 10 \
    --batch_size 32
```

### Training Options

| Argument           | Default                    | Description                        |
|--------------------|----------------------------|------------------------------------|
| `--image_dir`      | `data/images`              | Root directory for images          |
| `--triplets_csv`   | `None`                     | Path to explicit triplet CSV       |
| `--catalog_csv`    | `data/Cleaned/dataset.csv` | Path to data catalog CSV           |
| `--train_csv`      | `data/train.csv`           | Path to siamese training CSV       |
| `--output_dir`     | `checkpoints`              | Where to save model checkpoints    |
| `--epochs`         | `5`                        | Number of training epochs          |
| `--batch_size`     | `32`                       | Batch size                         |
| `--learning_rate`  | `2e-5`                     | Learning rate                      |
| `--projection_dim` | `256`                      | Output embedding dimension         |
| `--margin`         | `0.5`                      | Triplet loss margin                |
| `--freeze_clip`    | `False`                    | Freeze CLIP backbone weights       |
| `--dry-run`        | `False`                    | Disable WandB logging              |

## 6. PDP Recommendation Inference

Run inference to get recommendations for a specific product (image + title). The catalog CSV is used to build the search index.

```bash
python scripts/infer_pdp.py \
    --image_dir path/to/images \
    --catalog_csv path/to/dataset.csv \
    --model_path checkpoints/checkpoint-epoch-9 \
    --query_image path/to/query.jpg \
    --query_text "Red floral dress" \
    --top_k 5
```

The catalog CSV must have `image_path` and `caption` columns when used for inference indexing.

## 7. Verification

Verify the pipeline using built-in verification mode (generates dummy data automatically):

```bash
# Verify Data Loading
python scripts/verify_data.py

# Verify Training Loop - Siamese mode (dry run)
python scripts/train.py --dry-run --verify-mode --epochs 1 --batch_size 2

# Verify Training Loop - Triplet mode (dry run)
python scripts/train.py --dry-run --verify-mode --triplets_csv dummy --epochs 1 --batch_size 2

# Verify PDP Inference
python scripts/verify_pdp.py
```

## Architecture Overview

- **Model**: `FashionCLIPModel` ([src/model/modeling.py](src/model/modeling.py))
    - Base: `openai/clip-vit-base-patch32`
    - Fusion: Concatenates Image and Text features
    - Projection: MLP reducing to 256 dimensions
- **Loss**: `BatchHardTripletLoss` ([src/model/loss.py](src/model/loss.py))
    - Optimizes the embedding space by pushing dissimilar items apart.
- **Data**:
    - `FashionCLIPDataset` ([src/data/dataset.py](src/data/dataset.py)) — for siamese/zero-shot training
    - `TripletFashionDataset` ([src/data/triplet_dataset.py](src/data/triplet_dataset.py)) — for explicit triplet training; requires both a triplet CSV and a data catalog CSV
