import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import numpy as np
from typing import Optional

class FashionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        device: str,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        
        self.global_step = 0

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Check if batch contains explicit triplets
            if "anchor_pixel_values" in batch:
                # Explicit Triplet Mode
                anc_img = batch["anchor_pixel_values"].to(self.device)
                anc_ids = batch["anchor_input_ids"].to(self.device)
                anc_mask = batch["anchor_attention_mask"].to(self.device)
                
                pos_img = batch["positive_pixel_values"].to(self.device)
                pos_ids = batch["positive_input_ids"].to(self.device)
                pos_mask = batch["positive_attention_mask"].to(self.device)
                
                neg_img = batch["negative_pixel_values"].to(self.device)
                neg_ids = batch["negative_input_ids"].to(self.device)
                neg_mask = batch["negative_attention_mask"].to(self.device)
                
                # Forward Pass
                anchor_emb = self.model(pixel_values=anc_img, input_ids=anc_ids, attention_mask=anc_mask)
                positive_emb = self.model(pixel_values=pos_img, input_ids=pos_ids, attention_mask=pos_mask)
                negative_emb = self.model(pixel_values=neg_img, input_ids=neg_ids, attention_mask=neg_mask)
                
                # Standard Triplet Loss
                loss = torch.nn.functional.triplet_margin_loss(
                    anchor_emb, positive_emb, negative_emb, 
                    margin=self.config.get("margin", 0.5)
                )
                
            else:
                # Batch Hard Mode (Siamese)
                pixel_values_1 = batch["pixel_values"].to(self.device)
                input_ids_1 = batch["input_ids"].to(self.device)
                attention_mask_1 = batch["attention_mask"].to(self.device)
                
                pixel_values_2 = batch["pixel_values_2"].to(self.device)
                input_ids_2 = batch["input_ids_2"].to(self.device)
                attention_mask_2 = batch["attention_mask_2"].to(self.device)
                
                batch_size = pixel_values_1.size(0)
                
                # Forward Pass View 1
                emb1 = self.model(
                    pixel_values=pixel_values_1,
                    input_ids=input_ids_1,
                    attention_mask=attention_mask_1
                )
                
                # Forward Pass View 2
                emb2 = self.model(
                    pixel_values=pixel_values_2,
                    input_ids=input_ids_2, # Same text usually
                    attention_mask=attention_mask_2
                )
                
                # Combine for Triplet Loss
                embeddings = torch.cat([emb1, emb2], dim=0)
                labels = torch.cat([
                    torch.arange(batch_size, device=self.device),
                    torch.arange(batch_size, device=self.device)
                ], dim=0)
                
                loss = self.loss_fn(embeddings, labels)
            
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({"loss": loss.item()})
            
            if self.config.get("use_wandb", False):
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step
                })
                
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        # Full evaluation loop is complex (R@K), keeping it simple or reusing separate script logic.
        # For now, just return 0 or implement simple validation loss if data has pairs.
        # If val_loader is standard (not pairs), we can't compute triplet loss easily without mining.
        # So we skip val loss in this loop and rely on external R@K eval script.
        return {}

    def save_checkpoint(self, path, epoch=None):
        # Save model wrapper
        if hasattr(self.model, "module"):
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        print(f"Saved model to {path}")

        if self.config.get("use_wandb", False) and epoch is not None:
            artifact = wandb.Artifact(
                name="fashion-clip-checkpoint",
                type="model",
                metadata={"epoch": epoch}
            )
            artifact.add_dir(path)
            wandb.log_artifact(artifact)

    def train(self, num_epochs):
        if self.config.get("use_wandb", False):
            wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Loss: {train_loss:.4f}")
            
            # Checkpoint (every epoch for now)
            save_path = os.path.join(self.config["output_dir"], f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)
            self.save_checkpoint(save_path, epoch=epoch)
