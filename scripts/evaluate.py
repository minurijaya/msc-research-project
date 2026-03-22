"""
Evaluation script for FashionCLIP.

Computes Triplet Accuracy, Recall@K and MRR for:
  - Fine-tuned FashionCLIP (from checkpoint)
  - Zero-shot CLIP baseline (pretrained ViT-B/32, image-only)

Results are logged to Weights & Biases as a summary table.

Usage (mirrors the training command):
  python scripts/evaluate.py \
      --image_dir /content/drive/MyDrive/FashionCLIP/ \
      --catalog_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/dataset.csv \
      --triplets_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/train.csv \
      --model_path checkpoints/checkpoint-epoch-40 \
      --batch_size 64
"""

import sys
import os
import argparse
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.modeling import FashionCLIPModel
from src.data.transforms import get_transforms

import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Catalog Dataset (loads every item from the ID/Image/Caption catalog CSV)
# ─────────────────────────────────────────────────────────────────────────────

class CatalogDataset(Dataset):
    """Loads all items from the catalog CSV for building the embedding index."""

    def __init__(self, catalog_path: str, image_root: str, tokenizer, transform):
        df = pd.read_csv(catalog_path)
        # Support ID/Image/Caption or image_path/caption column conventions
        if "ID" in df.columns:
            df = df.rename(columns={"Image": "image_path", "Caption": "caption"})
        self.data = df.reset_index(drop=True)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        pixel_values = self.transform(image)

        enc = self.tokenizer(
            str(row["caption"]),
            padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> str:
    """Return the highest-epoch checkpoint directory inside output_dir."""
    pattern = os.path.join(output_dir, "checkpoint-epoch-*")
    dirs = sorted(glob.glob(pattern),
                  key=lambda p: int(p.rsplit("-", 1)[-1]))
    if not dirs:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    return dirs[-1]


def load_fashionclip(checkpoint_dir: str, device: torch.device,
                     projection_dim: int = 256) -> FashionCLIPModel:
    model = FashionCLIPModel(
        clip_model_name="openai/clip-vit-base-patch32",
        projection_dim=projection_dim
    )
    fusion_path = os.path.join(checkpoint_dir, "fusion.pt")
    proj_path   = os.path.join(checkpoint_dir, "projection.pt")
    if os.path.exists(fusion_path):
        model.fusion.load_state_dict(
            torch.load(fusion_path, map_location=device))
    else:
        print(f"  [warn] fusion.pt not found in {checkpoint_dir}")
    if os.path.exists(proj_path):
        model.projection.load_state_dict(
            torch.load(proj_path, map_location=device))
    else:
        print(f"  [warn] projection.pt not found in {checkpoint_dir}")
    model.to(device).eval()
    return model


@torch.no_grad()
def build_fashionclip_embeddings(model, catalog_dataset, batch_size, device):
    """Encode every catalog item with the fine-tuned FashionCLIP model."""
    loader = DataLoader(catalog_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_embs = []
    for batch in tqdm(loader, desc="  FashionCLIP catalog embeddings"):
        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        emb = model(pv, ids, msk)          # already L2-normalised
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)      # (N, 256)


@torch.no_grad()
def build_zeroshot_embeddings(clip_model, catalog_dataset, batch_size, device):
    """Encode every catalog item with zero-shot CLIP (image features only)."""
    loader = DataLoader(catalog_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_embs = []
    for batch in tqdm(loader, desc="  Zero-shot CLIP catalog embeddings"):
        pv = batch["pixel_values"].to(device)
        emb = clip_model.get_image_features(pixel_values=pv)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)      # (N, 512)


def compute_triplet_metrics(embeddings: torch.Tensor,
                             triplets_df: pd.DataFrame,
                             id_to_idx: dict,
                             k_values=(1, 5, 10)) -> dict:
    """
    Given catalog embeddings (N, D) and triplet definitions, compute:
      - Triplet Accuracy  : sim(anchor,pos) > sim(anchor,neg)
      - R@K               : positive in top-K from full catalog
      - MRR               : 1 / rank(positive) averaged over triplets
    """
    embeddings = embeddings.float()  # ensure fp32

    triplet_correct = 0
    reciprocal_ranks = []
    recall_hits = {k: 0 for k in k_values}
    valid = 0

    N = len(embeddings)

    for _, row in tqdm(triplets_df.iterrows(),
                       total=len(triplets_df),
                       desc="  evaluating triplets", leave=False):
        anc_id = row["anchor_image"]
        pos_id = row["positive_image"]
        neg_id = row["negative_image"]

        if anc_id not in id_to_idx or pos_id not in id_to_idx or neg_id not in id_to_idx:
            continue

        a_idx = id_to_idx[anc_id]
        p_idx = id_to_idx[pos_id]
        n_idx = id_to_idx[neg_id]

        a_emb = embeddings[a_idx]          # (D,)
        p_emb = embeddings[p_idx]
        n_emb = embeddings[n_idx]

        sim_pos = torch.dot(a_emb, p_emb).item()
        sim_neg = torch.dot(a_emb, n_emb).item()

        # Triplet accuracy
        if sim_pos > sim_neg:
            triplet_correct += 1

        # Full-catalog retrieval: similarities to all items
        sims = (embeddings @ a_emb).numpy()   # (N,)
        sims[a_idx] = -1.0                     # exclude self

        sorted_idx = np.argsort(sims)[::-1]   # descending
        rank = np.where(sorted_idx == p_idx)[0]
        if len(rank) == 0:
            continue
        rank = int(rank[0]) + 1               # 1-indexed

        reciprocal_ranks.append(1.0 / rank)

        for k in k_values:
            if rank <= k:
                recall_hits[k] += 1

        valid += 1

    if valid == 0:
        return {}

    results = {
        "Triplet Accuracy": triplet_correct / valid,
        "MRR": float(np.mean(reciprocal_ranks)),
    }
    for k in k_values:
        results[f"R@{k}"] = recall_hits[k] / valid

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if args.model_path:
        ckpt_dir = args.model_path
    else:
        ckpt_dir = find_latest_checkpoint(args.output_dir)
    print(f"Evaluating checkpoint: {ckpt_dir}")

    # ── Tokenizer & transforms ────────────────────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _, val_transform = get_transforms(img_size=224)

    # ── Catalog dataset ───────────────────────────────────────────────────────
    print("\nLoading catalog...")
    catalog_ds = CatalogDataset(
        catalog_path=args.catalog_csv,
        image_root=args.image_dir,
        tokenizer=tokenizer,
        transform=val_transform,
    )
    print(f"  Catalog size: {len(catalog_ds)} items")

    # Build ID → catalog index mapping
    catalog_df = pd.read_csv(args.catalog_csv)
    if "ID" in catalog_df.columns:
        id_col = "ID"
    else:
        id_col = catalog_df.columns[0]
    id_to_idx = {str(row[id_col]): i for i, row in catalog_df.iterrows()}

    # ── Load triplets ─────────────────────────────────────────────────────────
    print("Loading triplets...")
    triplets_df = pd.read_csv(args.triplets_csv)
    # Normalise column names
    rename = {}
    for col in ["anchor_image", "positive_image", "negative_image"]:
        if col not in triplets_df.columns and f"{col}_id" in triplets_df.columns:
            rename[f"{col}_id"] = col
    if rename:
        triplets_df = triplets_df.rename(columns=rename)
    # Ensure IDs are strings
    for col in ["anchor_image", "positive_image", "negative_image"]:
        triplets_df[col] = triplets_df[col].astype(str)
    print(f"  Triplets: {len(triplets_df)}")

    # ── WandB init ────────────────────────────────────────────────────────────
    epoch_num = ckpt_dir.rsplit("-", 1)[-1] if "epoch" in ckpt_dir else "?"
    if not args.dry_run:
        wandb.init(
            project="fashion-clip-recommender",
            name=f"eval-epoch-{epoch_num}",
            config={
                "checkpoint": ckpt_dir,
                "catalog_size": len(catalog_ds),
                "num_triplets": len(triplets_df),
                "projection_dim": args.projection_dim,
                "batch_size": args.batch_size,
            }
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Fine-tuned FashionCLIP evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Fine-tuned FashionCLIP ──────────────────────────────────────")
    fc_model = load_fashionclip(ckpt_dir, device, args.projection_dim)
    fc_embeddings = build_fashionclip_embeddings(
        fc_model, catalog_ds, args.batch_size, device)
    fc_results = compute_triplet_metrics(fc_embeddings, triplets_df, id_to_idx)

    print("  Results:")
    for k, v in fc_results.items():
        print(f"    {k}: {v:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Zero-shot CLIP baseline (image-only)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Zero-Shot CLIP Baseline (image-only) ────────────────────────")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    zs_embeddings = build_zeroshot_embeddings(
        clip_model, catalog_ds, args.batch_size, device)
    zs_results = compute_triplet_metrics(zs_embeddings, triplets_df, id_to_idx)

    print("  Results:")
    for k, v in zs_results.items():
        print(f"    {k}: {v:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Summary & WandB logging
    # ══════════════════════════════════════════════════════════════════════════
    print("\n═══════════════════════════════════════════════════════════════")
    print(f"{'Metric':<22} {'Zero-Shot CLIP':>16} {'FashionCLIP (ft)':>18} {'Δ':>8}")
    print("─" * 68)
    for metric in ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]:
        zs  = zs_results.get(metric, float("nan"))
        fc  = fc_results.get(metric, float("nan"))
        delta = fc - zs if not (np.isnan(zs) or np.isnan(fc)) else float("nan")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {metric:<20} {zs:>16.4f} {fc:>18.4f} {arrow}{abs(delta):>6.4f}")
    print("═══════════════════════════════════════════════════════════════\n")

    if not args.dry_run:
        # Log fine-tuned metrics with fc_ prefix
        wandb.log({f"fc/{k}": v for k, v in fc_results.items()})
        # Log zero-shot metrics with zs_ prefix
        wandb.log({f"zs/{k}": v for k, v in zs_results.items()})
        # Log deltas
        for metric in fc_results:
            if metric in zs_results:
                wandb.log({f"delta/{metric}": fc_results[metric] - zs_results[metric]})

        # WandB comparison table
        table = wandb.Table(columns=["Metric", "Zero-Shot CLIP", "FashionCLIP (fine-tuned)", "Delta"])
        for metric in ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]:
            zs  = zs_results.get(metric, float("nan"))
            fc  = fc_results.get(metric, float("nan"))
            delta = fc - zs if not (np.isnan(zs) or np.isnan(fc)) else float("nan")
            table.add_data(metric, round(zs, 4), round(fc, 4), round(delta, 4))
        wandb.log({"evaluation_summary": table})

        # Bar chart: R@K comparison
        bar_data = wandb.Table(columns=["K", "Zero-Shot CLIP", "FashionCLIP"])
        for k in [1, 5, 10]:
            bar_data.add_data(k,
                              zs_results.get(f"R@{k}", 0),
                              fc_results.get(f"R@{k}", 0))
        wandb.log({
            "recall_comparison": wandb.plot.bar(
                bar_data, "K",
                ["Zero-Shot CLIP", "FashionCLIP"],
                title="Recall@K: Zero-Shot vs Fine-tuned")
        })

        wandb.finish()
        print("Results logged to Weights & Biases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate FashionCLIP checkpoint against zero-shot CLIP baseline"
    )
    parser.add_argument("--image_dir",      type=str, required=True,
                        help="Root directory for product images")
    parser.add_argument("--catalog_csv",    type=str, required=True,
                        help="Product catalog CSV (ID, Image, Caption)")
    parser.add_argument("--triplets_csv",   type=str, required=True,
                        help="Triplets CSV (anchor_image, positive_image, negative_image)")
    parser.add_argument("--model_path",     type=str, default=None,
                        help="Path to a specific checkpoint dir. "
                             "If omitted, the latest in --output_dir is used.")
    parser.add_argument("--output_dir",     type=str, default="checkpoints",
                        help="Checkpoint root (used to find latest if --model_path is not set)")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Disable WandB logging (print results only)")

    args = parser.parse_args()
    main(args)
