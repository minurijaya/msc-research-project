"""
Post-hoc per-epoch evaluation for overfitting analysis.

Sweeps every checkpoint-epoch-* directory, evaluates Triplet Accuracy,
Recall@K and MRR on the triplet set, then logs the results to WandB
at the matching epoch step.

Optionally resumes the original training run (--wandb_run_id) so that
eval curves appear in the same chart panel as the training loss.

Usage (mirrors the training command):
    python scripts/evaluate_epochs.py \
        --image_dir /content/drive/MyDrive/FashionCLIP/ \
        --catalog_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/dataset.csv \
        --triplets_csv /content/drive/MyDrive/FashionCLIP/data/Cleaned/train.csv \
        --output_dir checkpoints \
        --batch_size 64

To log into the SAME WandB run as training, add:
        --wandb_run_id <run_id_from_training>
(find it on the WandB run page URL: .../runs/<run_id>)
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

def plot_curves(records: list, out_path: str):
    """
    records: list of dicts with keys 'epoch', 'train_loss' (optional),
             'Triplet Accuracy', 'R@1', 'R@5', 'R@10', 'MRR'
    """
    epochs = [r["epoch"] for r in records]
    metrics = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]
    has_loss = any("train_loss" in r for r in records)

    n_panels = len(metrics) + (1 if has_loss else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    if has_loss:
        loss_vals = [r.get("train_loss", float("nan")) for r in records]
        axes[panel].plot(epochs, loss_vals, "o-", color="#E74C3C", linewidth=1.8)
        axes[panel].set_title("Training Loss", fontsize=11)
        axes[panel].set_xlabel("Epoch")
        axes[panel].set_ylabel("Loss")
        axes[panel].grid(True, alpha=0.3)
        panel += 1

    colours = ["#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#E67E22"]
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, float("nan")) for r in records]
        axes[panel].plot(epochs, vals, "o-", color=colours[i], linewidth=1.8, label=metric)
        axes[panel].set_title(metric, fontsize=11)
        axes[panel].set_xlabel("Epoch")
        axes[panel].set_ylabel("Score")
        axes[panel].set_ylim(0, 1.05)
        axes[panel].grid(True, alpha=0.3)

        # Mark peak
        finite = [(e, v) for e, v in zip(epochs, vals) if not np.isnan(v)]
        if finite:
            best_e, best_v = max(finite, key=lambda x: x[1])
            axes[panel].axvline(best_e, color=colours[i], linestyle="--", alpha=0.5)
            axes[panel].annotate(f"peak\ne={best_e}\n{best_v:.3f}",
                                  xy=(best_e, best_v),
                                  xytext=(best_e + 0.5, best_v - 0.08),
                                  fontsize=7, color=colours[i])
        panel += 1

    fig.suptitle("FashionCLIP – Training & Evaluation Curves", fontsize=13, fontweight="bold")
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
    triplets_df = pd.read_csv(args.triplets_csv)
    rename = {}
    for col in ["anchor_image", "positive_image", "negative_image"]:
        if col not in triplets_df.columns and f"{col}_id" in triplets_df.columns:
            rename[f"{col}_id"] = col
    if rename:
        triplets_df = triplets_df.rename(columns=rename)
    for col in ["anchor_image", "positive_image", "negative_image"]:
        triplets_df[col] = triplets_df[col].astype(str)
    print(f"  {len(triplets_df)} triplets")

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
                    "num_triplets": len(triplets_df),
                    "projection_dim": args.projection_dim,
                    "num_checkpoints": len(ckpt_dirs),
                }
            )
            print(f"New WandB run: {run.id}")
            print("  Tip: pass --wandb_run_id to merge with your training run.")

    # ── Pre-compute zero-shot baseline once ───────────────────────────────────
    print("\nComputing zero-shot CLIP baseline (once)...")
    zs_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    zs_clip.eval()
    zs_embs = []
    with torch.no_grad():
        for batch in tqdm(catalog_loader, desc="  zero-shot embeddings"):
            pv = batch["pixel_values"].to(device)
            e  = F.normalize(zs_clip.get_image_features(pixel_values=pv), dim=-1)
            zs_embs.append(e.cpu())
    zs_embeddings = torch.cat(zs_embs, dim=0)
    zs_metrics = triplet_metrics(zs_embeddings, triplets_df, id_to_idx)
    print("  Zero-shot:", {k: f"{v:.4f}" for k, v in zs_metrics.items()})
    del zs_clip  # free VRAM

    # ── Initialise model (reuse shell across checkpoints) ─────────────────────
    model = FashionCLIPModel(
        clip_model_name="openai/clip-vit-base-patch32",
        projection_dim=args.projection_dim
    ).to(device)

    # ── Sweep checkpoints ─────────────────────────────────────────────────────
    records = []
    print(f"\nEvaluating {len(ckpt_dirs)} checkpoints...")

    for ckpt_dir in ckpt_dirs:
        epoch = int(ckpt_dir.rsplit("-", 1)[-1])
        print(f"\n  Epoch {epoch:>3}  ({ckpt_dir})")

        model = load_fashionclip_weights(model, ckpt_dir, device)

        with torch.no_grad():
            fc_embeddings = encode_catalog(model, catalog_loader, device)

        metrics = triplet_metrics(fc_embeddings, triplets_df, id_to_idx)
        metrics["epoch"] = epoch
        records.append(metrics)

        summary = "  " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch")
        print(summary)

        if not args.dry_run:
            log_dict = {f"eval/{k}": v for k, v in metrics.items() if k != "epoch"}
            wandb.log(log_dict, step=epoch)

    # ── Log zero-shot as horizontal reference lines ───────────────────────────
    if not args.dry_run:
        wandb.log({f"eval/zeroshot_{k}": v for k, v in zs_metrics.items()}, step=0)

    # ── Summary table in WandB ────────────────────────────────────────────────
    if not args.dry_run:
        metric_keys = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]
        best_records = {}
        for m in metric_keys:
            vals = [(r["epoch"], r.get(m, float("nan"))) for r in records]
            vals = [(e, v) for e, v in vals if not np.isnan(v)]
            if vals:
                best_e, best_v = max(vals, key=lambda x: x[1])
                best_records[m] = (best_e, best_v)

        table = wandb.Table(columns=["Metric", "Zero-Shot", "Best Fine-Tuned", "Best Epoch", "Delta"])
        for m in metric_keys:
            zs_v = zs_metrics.get(m, float("nan"))
            if m in best_records:
                best_e, best_v = best_records[m]
                delta = best_v - zs_v
                table.add_data(m, round(zs_v, 4), round(best_v, 4), best_e, round(delta, 4))
        wandb.log({"best_eval_summary": table})

    # ── Local matplotlib plot ─────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "eval_curves.png")
    plot_curves(records, plot_path)

    if not args.dry_run:
        wandb.log({"eval_curves": wandb.Image(plot_path)})
        wandb.finish()

    # ── Print overfitting summary ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("═" * 60)
    metric_keys = ["Triplet Accuracy", "R@1", "R@5", "R@10", "MRR"]
    for m in metric_keys:
        vals = [(r["epoch"], r.get(m, float("nan"))) for r in records]
        vals = [(e, v) for e, v in vals if not np.isnan(v)]
        if not vals:
            continue
        best_e, best_v = max(vals, key=lambda x: x[1])
        last_e, last_v = vals[-1]
        drop = last_v - best_v
        status = "✓ stable" if abs(drop) < 0.01 else ("⚠ degraded" if drop < 0 else "↑ improving")
        print(f"  {m:<22}  peak={best_v:.4f} @ epoch {best_e:>2}  "
              f"last={last_v:.4f}  Δ={drop:+.4f}  {status}")
    print("═" * 60)
    print(f"\nFull curves saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",      type=str, required=True)
    parser.add_argument("--catalog_csv",    type=str, required=True)
    parser.add_argument("--triplets_csv",   type=str, required=True)
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
