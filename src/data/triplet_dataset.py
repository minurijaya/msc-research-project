import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from typing import List, Optional, Callable, Dict, Any

class TripletFashionDataset(Dataset):
    """
    Dataset for Explicit Triplet Training.
    Expects metadata format:
    - anchor_image, anchor_caption
    - positive_image, positive_caption
    - negative_image, negative_caption
    """
    def __init__(
        self,
        image_root_dir: str,
        data_catalog_path: str,
        metadata_path: str,
        tokenizer: Any,
        transform: Optional[Callable] = None,
        max_length: int = 77,
    ):
        """
        Args:
            image_root_dir (str): Path to root image directory.
            data_catalog_path (str): Path to CSV containing image paths & captions.
            metadata_path (str): Path to CSV containing triplets.
            tokenizer: HuggingFace CLIPTokenizer.
            transform: Image transforms.
        """
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data catalog: ID -> {image_path, caption}
        catalog_df = pd.read_csv(data_catalog_path)
        required_catalog_cols = ["ID", "Image", "Caption"]
        for col in required_catalog_cols:
            if col not in catalog_df.columns:
                raise ValueError(f"Missing required column '{col}' in data catalog.")
        self.catalog = catalog_df.set_index("ID")[["Image", "Caption"]].to_dict(orient="index")

        # Load triplet metadata (only IDs needed)
        if metadata_path.endswith('.csv'):
            self.data = pd.read_csv(metadata_path)
        elif metadata_path.endswith('.parquet'):
            self.data = pd.read_parquet(metadata_path)
        else:
            raise ValueError("Metadata must be CSV or Parquet.")

        # Support both 'anchor_image' and 'anchor_image_id' column naming conventions
        col_rename = {}
        for col in ["anchor_image", "positive_image", "negative_image"]:
            if col not in self.data.columns and f"{col}_id" in self.data.columns:
                col_rename[f"{col}_id"] = col
        if col_rename:
            self.data = self.data.rename(columns=col_rename)

        required_cols = ["anchor_image", "positive_image", "negative_image"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column '{col}' in triplet metadata.")

    def __len__(self):
        return len(self.data)

    def _resolve(self, item_id):
        """Look up image path and caption from catalog by ID."""
        if item_id not in self.catalog:
            raise KeyError(f"ID '{item_id}' not found in data catalog.")
        entry = self.catalog[item_id]
        return entry["Image"], entry["Caption"]

    def _process_item(self, item_id):
        # Resolve image path and caption from catalog
        image_path, caption = self._resolve(item_id)

        # Load Image
        full_path = os.path.join(self.image_root_dir, image_path)
        try:
            image_pil = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            image_pil = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            pixel_values = self.transform(image_pil)
        else:
            pixel_values = image_pil

        # Tokenize Text
        text_inputs = self.tokenizer(
            str(caption),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return pixel_values, text_inputs.input_ids.squeeze(0), text_inputs.attention_mask.squeeze(0)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Process Anchor
        anc_img, anc_ids, anc_mask = self._process_item(row["anchor_image"])
        
        # Process Positive
        pos_img, pos_ids, pos_mask = self._process_item(row["positive_image"])
        
        # Process Negative
        neg_img, neg_ids, neg_mask = self._process_item(row["negative_image"])
        
        return {
            "anchor_pixel_values": anc_img,
            "anchor_input_ids": anc_ids,
            "anchor_attention_mask": anc_mask,
            
            "positive_pixel_values": pos_img,
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_mask,
            
            "negative_pixel_values": neg_img,
            "negative_input_ids": neg_ids,
            "negative_attention_mask": neg_mask
        }
