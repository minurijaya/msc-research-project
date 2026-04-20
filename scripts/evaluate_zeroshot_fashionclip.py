
import sys
import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.transforms import get_transforms

import wandb


# Dataset 

class CatalogDataset(Dataset):
    def __init__(self, catalog_path, image_root, transform):
        df = pd.read_csv(catalog_path)
        if "ID" in df.columns:
            df = df.rename(columns={"Image": "image_path", "Caption": "caption"})
        self.data = df.reset_index(drop=True)
        self.image_root = image_root
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
        return {"pixel_values": self.transform(image)}


# Triplet metrics 

def triplet_metrics(embeddings, triplets_df, id_to_idx, k_values=(1, 5, 10)):
    embeddings = embeddings.float()
    triplet_correct, valid = 0, 0
    reciprocal_ranks = []
    recall_hits = {k: 0 for k in k_values}

    for _, row in tqdm(triplets_df.iterrows(), total=len(triplets_df),
                       desc="  evaluating triplets", leave=False):
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


#  Main 

def main(args):
    MODEL_NAME = "patrickjohncyh/fashion-clip"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")

    #  Transforms & catalog 
    _, val_transform = get_transforms(img_size=224)

    catalog_ds = CatalogDataset(args.catalog_csv, args.image_dir, val_transform)
    catalog_loader = DataLoader(
        catalog_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Catalogue: {len(catalog_ds)} items")

    raw_df = pd.read_csv(args.catalog_csv)
    id_col = "ID" if "ID" in raw_df.columns else raw_df.columns[0]
    id_to_idx = {str(row[id_col]): i for i, row in raw_df.iterrows()}

    #  Test triplets 
    test_df = pd.read_csv(args.test_csv)
    for col in ["anchor_image", "positive_image", "negative_image"]:
        if col not in test_df.columns and f"{col}_id" in test_df.columns:
            test_df = test_df.rename(columns={f"{col}_id": col})
        test_df[col] = test_df[col].astype(str)
    print(f"Test triplets: {len(test_df)}")

    #  Load model 
    print(f"\nLoading {MODEL_NAME} ...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    #  Encode catalogue (image features only) 
    all_embs = []
    with torch.no_grad():
        for batch in tqdm(catalog_loader, desc="Encoding catalogue"):
            pv  = batch["pixel_values"].to(device)
            out = model.get_image_features(pixel_values=pv)
            out = out.pooler_output if not isinstance(out, torch.Tensor) else out
            all_embs.append(F.normalize(out, dim=-1).cpu())
    embeddings = torch.cat(all_embs, dim=0)   # (N, 512)
    print(f"Embeddings shape: {embeddings.shape}")

    # Evaluating on test triplets
    print("\nEvaluating on test triplets")
    metrics = triplet_metrics(embeddings, test_df, id_to_idx)

   
    for k, v in metrics.items():
        print(f"  {k:<22} {v:.4f}")

    # logging to WandB
    if not args.dry_run:
        wandb.init(
            project="fashion-clip-recommender",
            name="zeroshot-fashion-clip",
            config={
                "model":          MODEL_NAME,
                "catalog_size":   len(catalog_ds),
                "test_triplets":  len(test_df),
                "modality":       "image-only",
            }
        )
        wandb.log({f"zeroshot/{k}": v for k, v in metrics.items()})
        wandb.finish()
       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation for patrickjohncyh/fashion-clip"
    )
    parser.add_argument("--image_dir",   type=str, required=True,
                        help="Root directory for product images")
    parser.add_argument("--catalog_csv", type=str, required=True,
                        help="Product catalogue CSV (ID, Image, Caption)")
    parser.add_argument("--test_csv",    type=str, required=True,
                        help="Held-out test triplets CSV")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Disable WandB logging")
    args = parser.parse_args()
    main(args)
