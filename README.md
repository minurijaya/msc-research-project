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

> [!NOTE]
> `faiss-cpu` is currently excluded due to Python 3.13 compatibility issues. The implemented code uses brute-force similarity search for evaluation (Recall@K), which is sufficient for development.

## 2. Data Preparation

The system expects data in the following structure:
- **Images**: A directory containing all product images.
- **Metadata**: CSV files (`train.csv`, `val.csv`) with at least two columns:
    - `image_path`: Relative path to the image file (from the image root directory).
    - `caption`: Text description of the fashion item.

Example `train.csv`:
```csv
image_path,caption
dresses/dress_01.jpg,Red summer floral dress with V-neck
shirts/shirt_05.jpg,Blue denim shirt casual fit
```

## 3. Running Zero-Shot Baseline

Evaluate the pretrained CLIP model performance without any fine-tuning.

```bash
python scripts/evaluate_zero_shot.py \
    --image_dir path/to/images \
    --metadata_path path/to/val.csv \
    --batch_size 32
```

**Output**:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- WandB logs (if configured)

## 4. Training the Model

Fine-tune the model using Triplet Margin Loss with Hard Negative Mining.

```bash
python scripts/train.py \
    --image_dir path/to/images \
    --train_csv path/to/train.csv \
    --output_dir checkpoints \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --margin 0.5
```

**Key Arguments**:
- `--projection_dim`: Dimension of the output embedding (default: 256).
- `--freeze_clip`: Freeze the base CLIP model and train only the projection head (useful for small datasets).
- `--dry-run`: Run without logging to WandB.

## 5. Verification

You can verify the entire pipeline using the built-in verification mode which generates dummy data:

```bash
# Verify Data Loading
python scripts/verify_data.py

# Verify Training Loop (Dry Run)
python scripts/train.py --dry-run --verify-mode --epochs 1 --batch_size 2
```

## Architecture Overview

- **Model**: `FashionCLIPModel` (`src/model/modeling.py`)
    - Base: `openai/clip-vit-base-patch32`
    - Fusion: Concatenates Image and Text features
    - Projection: MLP reducing to 256 dimensions
- **Loss**: `BatchHardTripletLoss` (`src/model/loss.py`)
    - Optimizes the embedding space by pushing dissimilar items apart.
- **Data**: `FashionCLIPDataset` (`src/data/dataset.py`)
    - Returns pairs of augmented images for Siamese training.
