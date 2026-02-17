import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("Faiss not found. Using simple PyTorch/Numpy fallback.")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import FashionCLIPDataset
from src.data.transforms import get_transforms
from src.model.modeling import FashionCLIPModel

class SimpleIndex:
    def __init__(self, dim):
        self.dim = dim
        self.db = None
        
    def add(self, embeddings):
        if self.db is None:
            self.db = embeddings
        else:
            self.db = np.concatenate([self.db, embeddings], axis=0)
            
    def search(self, query, k):
        # query: (1, dim), db: (N, dim)
        # Cosine sim = dot product if normalized
        scores = np.dot(self.db, query.T).squeeze(1) # (N,)
        
        # Top-k
        indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[indices]
        
        return top_scores.reshape(1, -1), indices.reshape(1, -1)

def load_model(model_path, device, projection_dim=256):
    model_name = "openai/clip-vit-base-patch32"
    model = FashionCLIPModel(
        clip_model_name=model_name,
        projection_dim=projection_dim
    )
    
    if os.path.isdir(model_path):
        fusion_path = os.path.join(model_path, "fusion.pt")
        proj_path = os.path.join(model_path, "projection.pt")
        
        if os.path.exists(fusion_path):
            model.fusion.load_state_dict(torch.load(fusion_path, map_location=device))
        if os.path.exists(proj_path):
            model.projection.load_state_dict(torch.load(proj_path, map_location=device))
            
        try:
            model.clip.from_pretrained(model_path)
        except:
            print("Loaded fresh CLIP backbone.")
    
    model.to(device)
    model.eval()
    return model

def build_index(args, model, device, tokenizer, transform):
    print("Building Index from Catalog...")
    dataset = FashionCLIPDataset(
        image_root_dir=args.image_dir,
        metadata_path=args.catalog_csv,
        tokenizer=tokenizer,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            emb = model(pixel_values, input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Normalize
    if faiss:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        index = SimpleIndex(embeddings.shape[1])
    
    index.add(embeddings)
    return index, dataset.data

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    _, val_transform = get_transforms(img_size=224)
    
    model = load_model(args.model_path, device, projection_dim=args.projection_dim)
    
    index, catalog_df = build_index(args, model, device, tokenizer, val_transform)
    
    print(f"Processing Query: {args.query_image} | {args.query_text}")
    try:
        image = Image.open(args.query_image).convert("RGB")
        pixel_values = val_transform(image).unsqueeze(0).to(device)
    except:
        print("Error loading query image.")
        return

    text_inputs = tokenizer(
        args.query_text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    with torch.no_grad():
        query_emb = model(pixel_values, input_ids, attention_mask)
        query_emb = query_emb.cpu().numpy()
        
    D, I = index.search(query_emb, k=args.top_k)
    
    print("\n--- Top Recommendations ---")
    for rank, idx in enumerate(I[0]):
        row = catalog_df.iloc[idx]
        score = D[0][rank]
        print(f"{rank+1}. [Score: {score:.4f}] {row['image_path']} - {row['caption']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Root dir for images")
    parser.add_argument("--catalog_csv", type=str, required=True, help="Catalog metadata")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--query_image", type=str, required=True)
    parser.add_argument("--query_text", type=str, required=True)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=5)
    
    args = parser.parse_args()
    infer(args)
