"""
Post-hoc per-epoch evaluation for overfitting analysis.

Sweeps every checkpoint-epoch-* directory and evaluates Triplet Accuracy,
Recall@K and MRR on both the TRAINING triplet set and a held-out TEST set.
Logging both curves to WandB makes overfitting immediately visible:
  - train/* metrics keep rising → model still learning on training data
  - test/*  metrics plateau or fall → generalisation has peaked

Usage:
    python scripts/evaluate_epochs.py \
        --image_dir /content/drive/MyDrive/FashionCLIP/ \
        --catalog_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/dataset.csv \
        --triplets_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/train.csv \
        --test_csv     /content/drive/MyDrive/FashionCLIP/data/Cleaned/test.csv \
        --output_dir checkpoints \
        --batch_size 64

To merge eval curves into the original training run:
        --wandb_run_id <run_id>   (from the WandB run page URL)
"""

import sys, os, glob, argparse, json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.modeling import FashionCLIPModel
from src.data.transforms import get_transforms

import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CatalogDataset(Dataset):
    def __init__(self, catalog_path, image_root, tokenizer, transform):
        df = pd.read_csv(catalog_path)
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

def sorted_checkpoints(output_dir):
    """Return checkpoint directories sorted by epoch number."""
    pattern = os.path.join(output_dir, "checkpoint-epoch-*")
    dirs = glob.glob(pattern)
    dirs = sorted(dirs, key=lambda p: int(p.rsplit("-", 1)[-1]))
    return dirs


def load_fashionclip_weights(model: FashionCLIPModel, ckpt_dir: str,
                              device: torch.device) -> FashionCLIPModel:
    """
    Load fusion.pt and projection.pt from ckpt_dir into an existing model.
    Also tries to reload the CLIP backbone if saved there.
    Returns the model in eval mode.
    """
    fusion_path = os.path.join(ckpt_dir, "fusion.pt")
    proj_path   = os.path.join(ckpt_dir, "projection.pt")

    if os.path.exists(fusion_path):
        model.fusion.load_state_dict(torch.load(fusion_path, map_location=device))
    if os.path.exists(proj_path):
        model.projection.load_state_dict(torch.load(proj_path, map_location=device))

    # Try loading fine-tuned CLIP backbone (may not exist if not saved)
    clip_config = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(clip_config):
        try:
            finetuned_clip = CLIPModel.from_pretrained(ckpt_dir)
            model.clip = finetuned_clip.to(device)
        except Exception:
            pass  # Fall back to the backbone already on the model

    model.to(device).eval()
    return model


@torch.no_grad()
def encode_catalog(model, loader, device):
    """Encode all catalog items; returns (N, D) L2-normalised tensor."""
    all_embs = []
    for batch in loader:
        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        emb = model(pv, ids, msk)   # already L2-normalised by FashionCLIPModel
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def triplet_metrics(embeddings: torch.Tensor,
                    triplets_df: pd.DataFrame,
                    id_to_idx: dict,
                    k_values=(1, 5, 10)):
    embeddings = embeddings.float()
    triplet_correct, valid = 0, 0
    reciprocal_ranks = []
    recall_hits = {k: 0 for k in k_values}

    for _, row in triplets_df.iterrows():
        a_id = str(row["anchor_image"])
        p_id = str(row["positive_image"])
        n_id = str(row["negative_image"])
        if a_id not in id_to_idx or p_id not in id_to_idx or n_id not in id_to_idx:
            continue

        a = embeddings[id_to_idx[a_id]]
        p = embeddings[id_to_idx[p_id]]
        n = embeddings[id_to_idx[n_id]]

        if torch.dot(a, p).item() > torch.dot(a, n).item():
            triplet_correct += 1

        sims = (embeddings @ a).numpy()
        sims[id_to_idx[a_id]] = -1.0
        sorted_idx = np.argsort(sims)[::-1]
        rank_arr = np.where(sorted_idx == id_to_idx[p_id])[0]
        if len(rank_arr) == 0:
            continue
        rank = int(rank_arr[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in k_values:
            if rank <= k:
                recall_hits[k] += 1
        valid += 1

    if valid == 0:
        return {}
    result = {
        "Triplet Accuracy": triplet_correct / valid,
        "MRR":              float(np.mean(reciprocal_ranks)),
    }
    for k in k_values:
        result[f"R@{k}"] = recall_hits[k] / valid
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (local chart as backup / supplement to WandB)
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(train_records: list, test_records: list, out_path: str):
    """
    Plot train vs test curves for each metric on the same axes.
    If test_records is empty, only train curves are drawn.
    A vertical dashed line marks the test-best epoch (overfitting point).
    """
    epochs       = [r["epoch"] for r in train_records]
    metric_keys  = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]
    has_test     = len(test_records) > 0
    test_by_ep   = {r["epoch"]: r for r in test_records}

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4.8 * len(metric_keys), 4.8))

    colours = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD", "#C0392B"]

    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        c  = colours[i]

        train_vals = [r.get(metric, float("nan")) for r in train_records]
        ax.plot(epochs, train_vals, "o-", color=c, linewidth=2,
                label="Train", alpha=0.9)

        if has_test:
            test_epochs = sorted(test_by_ep.keys())
            test_vals   = [test_by_ep[e].get(metric, float("nan"))
                           for e in test_epochs]
            ax.plot(test_epochs, test_vals, "s--", color=c, linewidth=2,
                    alpha=0.55, label="Test")

            # Mark test-best (overfitting onset)
            finite = [(e, v) for e, v in zip(test_epochs, test_vals)
                      if not np.isnan(v)]
            if finite:
                best_e, best_v = max(finite, key=lambda x: x[1])
                ax.axvline(best_e, color=c, linestyle=":", linewidth=1.4, alpha=0.7)
                ax.annotate(f"best\ne={best_e}\n{best_v:.3f}",
                            xy=(best_e, best_v),
                            xytext=(best_e + 0.4, best_v - 0.10),
                            fontsize=7, color=c,
                            arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        if has_test:
            ax.legend(fontsize=8)

    fig.suptitle("FashionCLIP – Train vs Test Evaluation Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Find checkpoints ──────────────────────────────────────────────────────
    ckpt_dirs = sorted_checkpoints(args.output_dir)
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found in {args.output_dir}")
    print(f"Found {len(ckpt_dirs)} checkpoints: "
          f"epoch {ckpt_dirs[0].rsplit('-',1)[-1]} → {ckpt_dirs[-1].rsplit('-',1)[-1]}")

    # ── Tokenizer & transforms ────────────────────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _, val_transform = get_transforms(img_size=224)

    # ── Catalog ───────────────────────────────────────────────────────────────
    print("Loading catalog...")
    catalog_ds = CatalogDataset(
        args.catalog_csv, args.image_dir, tokenizer, val_transform)
    catalog_loader = DataLoader(
        catalog_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  {len(catalog_ds)} items")

    # ID → index mapping
    raw_df = pd.read_csv(args.catalog_csv)
    id_col = "ID" if "ID" in raw_df.columns else raw_df.columns[0]
    id_to_idx = {str(row[id_col]): i for i, row in raw_df.iterrows()}

    # ── Triplets ──────────────────────────────────────────────────────────────
    def load_triplets(path, label):
        df = pd.read_csv(path)
        rename = {}
        for col in ["anchor_image", "positive_image", "negative_image"]:
            if col not in df.columns and f"{col}_id" in df.columns:
                rename[f"{col}_id"] = col
        if rename:
            df = df.rename(columns=rename)
        for col in ["anchor_image", "positive_image", "negative_image"]:
            df[col] = df[col].astype(str)
        print(f"  {label}: {len(df)} triplets")
        return df

    print("Loading triplets...")
    triplets_df = load_triplets(args.triplets_csv, "train")
    test_df     = load_triplets(args.test_csv, "test") if args.test_csv else None
    if not args.test_csv:
        print("  No --test_csv provided; only train triplets will be evaluated.")

    # ── WandB init ────────────────────────────────────────────────────────────
    if not args.dry_run:
        if args.wandb_run_id:
            # Resume the original training run so eval metrics appear in the same charts
            run = wandb.init(
                project="fashion-clip-recommender",
                id=args.wandb_run_id,
                resume="must",
            )
            print(f"Resumed WandB run: {args.wandb_run_id}")
        else:
            run = wandb.init(
                project="fashion-clip-recommender",
                name="eval-all-epochs",
                config={
                    "catalog_size": len(catalog_ds),
                    "train_triplets": len(triplets_df),
                    "test_triplets": len(test_df) if test_df is not None else 0,
                    "projection_dim": args.projection_dim,
                    "num_checkpoints": len(ckpt_dirs),
                }
            )
            print(f"New WandB run: {run.id}")
            print("  Tip: pass --wandb_run_id to merge with your training run.")

    # ── Pre-compute zero-shot baseline once (on test set if available) ─────────
    print("\nComputing zero-shot CLIP baseline (once)...")
    zs_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    zs_clip.eval()
    zs_embs = []
    with torch.no_grad():
        for batch in tqdm(catalog_loader, desc="  zero-shot embeddings"):
            pv = batch["pixel_values"].to(device)
            e  = F.normalize(zs_clip.get_image_features(pixel_values=pv), dim=-1)
            zs_embs.append(e.cpu())
    zs_embeddings  = torch.cat(zs_embs, dim=0)
    ref_df         = test_df if test_df is not None else triplets_df
    zs_metrics     = triplet_metrics(zs_embeddings, ref_df, id_to_idx)
    print("  Zero-shot (on {}):".format("test" if test_df is not None else "train"),
          {k: f"{v:.4f}" for k, v in zs_metrics.items()})
    del zs_clip

    # ── Initialise model (reuse shell across checkpoints) ─────────────────────
    model = FashionCLIPModel(
        clip_model_name="openai/clip-vit-base-patch32",
        projection_dim=args.projection_dim
    ).to(device)

    # ── Sweep checkpoints ─────────────────────────────────────────────────────
    train_records, test_records = [], []
    print(f"\nEvaluating {len(ckpt_dirs)} checkpoints...")

    for ckpt_dir in ckpt_dirs:
        epoch = int(ckpt_dir.rsplit("-", 1)[-1])
        print(f"\n  Epoch {epoch:>3}  ({ckpt_dir})")

        model = load_fashionclip_weights(model, ckpt_dir, device)
        with torch.no_grad():
            fc_embeddings = encode_catalog(model, catalog_loader, device)

        # Train metrics
        tr_m = triplet_metrics(fc_embeddings, triplets_df, id_to_idx)
        tr_m["epoch"] = epoch
        train_records.append(tr_m)

        # Test metrics (if test set provided)
        te_m = {}
        if test_df is not None:
            te_m = triplet_metrics(fc_embeddings, test_df, id_to_idx)
            te_m["epoch"] = epoch
            test_records.append(te_m)

        # Print
        tr_str = "  train: " + "  ".join(f"{k}={v:.4f}" for k, v in tr_m.items() if k != "epoch")
        print(tr_str)
        if te_m:
            te_str = "  test:  " + "  ".join(f"{k}={v:.4f}" for k, v in te_m.items() if k != "epoch")
            print(te_str)

        if not args.dry_run:
            log = {f"train/{k}": v for k, v in tr_m.items() if k != "epoch"}
            if te_m:
                log.update({f"test/{k}": v for k, v in te_m.items() if k != "epoch"})
            wandb.log(log, step=epoch)

    # ── Log zero-shot baseline as reference ───────────────────────────────────
    if not args.dry_run:
        split = "test" if test_df is not None else "train"
        wandb.log({f"zeroshot/{split}/{k}": v for k, v in zs_metrics.items()}, step=0)

    # ── Summary table in WandB ────────────────────────────────────────────────
    if not args.dry_run:
        metric_keys = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]
        src_records = test_records if test_records else train_records
        src_label   = "Test" if test_records else "Train"

        best = {}
        for m in metric_keys:
            vals = [(r["epoch"], r.get(m, float("nan"))) for r in src_records]
            vals = [(e, v) for e, v in vals if not np.isnan(v)]
            if vals:
                best[m] = max(vals, key=lambda x: x[1])

        table = wandb.Table(columns=["Metric", "Zero-Shot",
                                     f"Best FashionCLIP ({src_label})",
                                     "Best Epoch", "Δ vs Zero-Shot"])
        for m in metric_keys:
            zs_v = zs_metrics.get(m, float("nan"))
            if m in best:
                best_e, best_v = best[m]
                table.add_data(m, round(zs_v, 4), round(best_v, 4),
                               best_e, round(best_v - zs_v, 4))
        wandb.log({"best_eval_summary": table})

    # ── Local matplotlib plot ─────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "eval_curves.png")
    plot_curves(train_records, test_records, plot_path)

    if not args.dry_run:
        wandb.log({"eval_curves": wandb.Image(plot_path)})
        wandb.finish()

    # ── Print overfitting summary ─────────────────────────────────────────────
    metric_keys = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]

    def summarise(records, label):
        print(f"\n{'═'*65}")
        print(f"  {label.upper()} — OVERFITTING ANALYSIS")
        print(f"{'═'*65}")
        for m in metric_keys:
            vals = [(r["epoch"], r.get(m, float("nan"))) for r in records]
            vals = [(e, v) for e, v in vals if not np.isnan(v)]
            if not vals:
                continue
            best_e, best_v = max(vals, key=lambda x: x[1])
            last_e, last_v = vals[-1]
            drop   = last_v - best_v
            status = ("✓ stable"   if abs(drop) < 0.01
                      else "⚠ degraded" if drop < 0
                      else "↑ improving")
            print(f"  {m:<22}  peak={best_v:.4f}@ep{best_e:<3}  "
                  f"last={last_v:.4f}  Δ={drop:+.4f}  {status}")

    summarise(train_records, "Train set")
    if test_records:
        summarise(test_records, "Test set")
    print(f"\n  Full curves: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",      type=str, required=True)
    parser.add_argument("--catalog_csv",    type=str, required=True)
    parser.add_argument("--triplets_csv",   type=str, required=True)
    parser.add_argument("--test_csv",       type=str, default=None,
                        help="Held-out test triplets CSV for overfitting analysis "
                             "(anchor_image, positive_image, negative_image)")
    parser.add_argument("--output_dir",     type=str, default="checkpoints",
                        help="Directory containing checkpoint-epoch-* folders")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--wandb_run_id",   type=str, default=None,
                        help="Resume an existing WandB run ID to merge eval "
                             "curves with training loss (find it in the run URL)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Disable WandB logging; only print and save local plot")
    args = parser.parse_args()
    main(args)
