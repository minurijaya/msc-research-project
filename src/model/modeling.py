import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig

class FashionCLIPModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_clip: bool = False
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
                
        self.config = self.clip.config
        embed_dim = self.config.projection_dim  # Usually 512 for ViT-B/32
        
        # Fusion Module: Concatenate Image and Text embeddings -> MLP
        # Input dim = embed_dim * 2 (if both present)
        # We handle missing modality by zero-padding, so input is always 2*embed_dim? 
        # Or better: We project each to common space then average?
        # Let's stick to concatenation + MLP as requested for "fusion".
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Projection Head: Compress to lower dimension
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
    def forward(
        self, 
        pixel_values: torch.Tensor = None, 
        input_ids: torch.Tensor = None, 
        attention_mask: torch.Tensor = None
    ):

        batch_size = 0
        device = self.clip.device
        
        # Extract Image Features
        if pixel_values is not None:
            image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
            if hasattr(image_embeds, "pooler_output"):
                image_embeds = image_embeds.pooler_output
            batch_size = image_embeds.shape[0]
        else:
            image_embeds = None

        # Extract Text Features
        if input_ids is not None:
            text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(text_embeds, "pooler_output"):
                text_embeds = text_embeds.pooler_output
            batch_size = text_embeds.shape[0] if batch_size == 0 else batch_size
        else:
            text_embeds = None
            
        # Handle Missing Modalities
        embed_dim = self.config.projection_dim
        
        if image_embeds is None:
            image_embeds = torch.zeros((batch_size, embed_dim), device=device)
        if text_embeds is None:
            text_embeds = torch.zeros((batch_size, embed_dim), device=device)
            
        # Fusion
        # Concatenate along dim 1
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        fused = self.fusion(combined)
        # Projection
        projected = self.projection(fused)
        # Normalize
        embeddings = projected / projected.norm(dim=-1, keepdim=True)
        
        return embeddings

    def save_pretrained(self, save_directory):
        self.clip.save_pretrained(save_directory)
        torch.save(self.fusion.state_dict(), f"{save_directory}/fusion.pt")
        torch.save(self.projection.state_dict(), f"{save_directory}/projection.pt")
