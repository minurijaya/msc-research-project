import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import FashionCLIPDataset
from src.data.transforms import get_transforms
from src.utils.metrics import compute_recall_at_k, compute_mrr
from transformers import CLIPModel, CLIPProcessor

def evaluate(args):
    # Initialize WandB
    if not args.dry_run:
        wandb.init(project="fashion-clip-recommender", name="zero-shot-baseline")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Load Data
    _, val_transform = get_transforms(img_size=224)
    
    # Use dummy data if paths not provided or verification mode
    if args.verify_mode:
        # Create dummy data on fly (similar to verify_data.py)
        import pandas as pd
        from PIL import Image
        os.makedirs("data/dummy_eval", exist_ok=True)
        img_path = "data/dummy_eval/test.jpg"
        Image.new('RGB', (224, 224), color='blue').save(img_path)
        pd.DataFrame({"image_path": ["dummy_eval/test.jpg"]*10, "caption": ["A blue dress"]*10}).to_csv("data/dummy_eval.csv", index=False)
        args.image_dir = "data"
        args.metadata_path = "data/dummy_eval.csv"
    
    dataset = FashionCLIPDataset(
        image_root_dir=args.image_dir,
        metadata_path=args.metadata_path,
        tokenizer=processor.tokenizer,
        transform=val_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    
    # Inference Loop
    image_embeddings = []
    text_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Embeddings"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get features
            img_emb = model.get_image_features(pixel_values=pixel_values)
            txt_emb = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            
            # Ensure we have tensors (some transformer versions return objects or require extraction)
            if hasattr(img_emb, "image_embeds"):
                img_emb = img_emb.image_embeds
            elif hasattr(img_emb, "pooler_output"):
                img_emb = img_emb.pooler_output
                
            if hasattr(txt_emb, "text_embeds"):
                txt_emb = txt_emb.text_embeds
            elif hasattr(txt_emb, "pooler_output"):
                txt_emb = txt_emb.pooler_output
            
            image_embeddings.append(img_emb.detach().cpu())
            text_embeddings.append(txt_emb.detach().cpu())
            
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    # Compute Metrics
    # Text-to-Image Retrieval
    t2i_recall = compute_recall_at_k(text_embeddings, image_embeddings, k=[1, 5, 10])
    t2i_mrr = compute_mrr(text_embeddings, image_embeddings)
    
    print("--- Results ---")
    print(f"R@1: {t2i_recall['R@1']:.4f}")
    print(f"R@5: {t2i_recall['R@5']:.4f}")
    print(f"R@10: {t2i_recall['R@10']:.4f}")
    print(f"MRR: {t2i_mrr:.4f}")
    
    if not args.dry_run:
        wandb.log({
            "R@1": t2i_recall['R@1'],
            "R@5": t2i_recall['R@5'],
            "R@10": t2i_recall['R@10'],
            "MRR": t2i_mrr
        })
        wandb.finish()
        
    # Cleanup verification data
    if args.verify_mode:
        import shutil
        shutil.rmtree("data/dummy_eval")
        os.remove("data/dummy_eval.csv")
        # Removing data dir only if empty/created by us
        try:
           os.rmdir("data") 
        except:
           pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--metadata_path", type=str, default="data/val.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Disable wandb logging")
    parser.add_argument("--verify-mode", action="store_true", help="Run with generated dummy data")
    
    args = parser.parse_args()
    evaluate(args)
