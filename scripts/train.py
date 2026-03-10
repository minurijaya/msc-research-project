import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import FashionCLIPDataset
from src.data.transforms import get_transforms
from src.model.modeling import FashionCLIPModel
from src.model.loss import BatchHardTripletLoss
from src.engine.trainer import FashionTrainer

from src.data.triplet_dataset import TripletFashionDataset

def main(args):
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize Components
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    train_transform, _ = get_transforms(img_size=224)
    
    # 2. Dataset
    # Use dummy data if verify mode
    if args.verify_mode:
        import pandas as pd
        from PIL import Image
        if args.triplets_csv:
            # Create dummy catalog + triplet data (new ID-based format)
            os.makedirs("data/dummy_triplets", exist_ok=True)
            Image.new('RGB', (224, 224), color='red').save("data/dummy_triplets/anc.jpg")
            Image.new('RGB', (224, 224), color='red').save("data/dummy_triplets/pos.jpg")
            Image.new('RGB', (224, 224), color='blue').save("data/dummy_triplets/neg.jpg")

            # Catalog: ID -> image path & caption
            pd.DataFrame({
                "ID":      ["ANC", "POS", "NEG"],
                "Image":   ["dummy_triplets/anc.jpg", "dummy_triplets/pos.jpg", "dummy_triplets/neg.jpg"],
                "Caption": ["Red shirt", "Reddish shirt", "Blue shirt"],
            }).to_csv("data/dummy_catalog.csv", index=False)

            # Triplets: only IDs
            pd.DataFrame({
                "anchor_image":   ["ANC"] * 10,
                "positive_image": ["POS"] * 10,
                "negative_image": ["NEG"] * 10,
            }).to_csv("data/triplets.csv", index=False)

            args.image_dir = "data"
            args.triplets_csv = "data/triplets.csv"
            args.catalog_csv  = "data/dummy_catalog.csv"
        else:
            os.makedirs("data/dummy_train", exist_ok=True)
            img_path = "data/dummy_train/train.jpg"
            Image.new('RGB', (224, 224), color='green').save(img_path)
            pd.DataFrame({"image_path": ["dummy_train/train.jpg"]*10, "caption": ["A green shirt"]*10}).to_csv("data/train.csv", index=False)
            args.image_dir = "data"
            args.train_csv = "data/train.csv"
        
    if args.triplets_csv:
        print(f"Using Explicit Triplet Dataset from {args.triplets_csv}")
        dataset = TripletFashionDataset(
            image_root_dir=args.image_dir,
            data_catalog_path=args.catalog_csv,
            metadata_path=args.triplets_csv,
            tokenizer=tokenizer,
            transform=train_transform
        )
    else:
        print(f"Using Siamese Dataset from {args.train_csv}")
        dataset = FashionCLIPDataset(
            image_root_dir=args.image_dir,
            metadata_path=args.train_csv,
            tokenizer=tokenizer,
            transform=train_transform,
            return_pairs=True # Critical for triplet loss
        )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    # 3. Model
    model = FashionCLIPModel(
        clip_model_name=model_name,
        projection_dim=args.projection_dim,
        freeze_clip=args.freeze_clip
    ).to(device)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = BatchHardTripletLoss(margin=args.margin)
    
    # 5. Trainer
    trainer = FashionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None, # Todo
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config={
            "use_wandb": not args.dry_run,
            "output_dir": args.output_dir
        }
    )
    
    # 6. Train
    if not args.dry_run:
        import wandb
        wandb.init(project="fashion-clip-recommender", name="triplet-loss-run")
        
    trainer.train(num_epochs=args.epochs)
    
    if args.verify_mode:
        import shutil
        if os.path.exists("data/dummy_triplets"):
            shutil.rmtree("data/dummy_triplets")
        for f in ["data/triplets.csv", "data/dummy_catalog.csv", "data/train.csv"]:
            if os.path.exists(f):
                os.remove(f)
        try: os.rmdir("data")
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--triplets_csv", type=str, default=None, help="Path to explicit triplets CSV (anchor_image, positive_image, negative_image IDs)")
    parser.add_argument("--catalog_csv", type=str, default="data/Cleaned/dataset.csv", help="Path to data catalog CSV (ID, Image, Caption)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--freeze_clip", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-mode", action="store_true")
    
    args = parser.parse_args()
    main(args)
