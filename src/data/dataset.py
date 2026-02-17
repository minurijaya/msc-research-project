import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from typing import List, Optional, Callable, Dict, Any

class FashionCLIPDataset(Dataset):
    """
    Dataset for Fashion Recommendation using CLIP.
    Expected data format:
    - Image directory: Root folder containing images.
    - Metadata: DataFrame or CSV with 'image_path' and 'caption' columns.
    """
    def __init__(
        self,
        image_root_dir: str,
        metadata_path: str,
        tokenizer: Any,
        transform: Optional[Callable] = None,
        image_column: str = "image_path",
        text_column: str = "caption",
        max_length: int = 77,
        return_pairs: bool = False,
    ):
        """
        Args:
            image_root_dir (str): Path to the directory with images.
            metadata_path (str): Path to CSV/Parquet file with metadata.
            tokenizer: HuggingFace CLIPTokenizer.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_column (str): Column name for image filenames/paths.
            text_column (str): Column name for text descriptions.
            max_length (int): Maximum sequence length for tokenization.
            return_pairs (bool): If True, returns two augmented views of the same image/text for triplet loss.
        """
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.text_column = text_column
        self.max_length = max_length
        self.return_pairs = return_pairs

        # Load metadata
        if metadata_path.endswith('.csv'):
            self.data = pd.read_csv(metadata_path)
        elif metadata_path.endswith('.parquet'):
            self.data = pd.read_parquet(metadata_path)
        else:
            raise ValueError("Metadata must be CSV or Parquet.")

        # Ensure columns exist
        if self.image_column not in self.data.columns:
            raise ValueError(f"Image column '{self.image_column}' not found in metadata.")
        if self.text_column not in self.data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in metadata.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load Image
        image_path = os.path.join(self.image_root_dir, row[self.image_column])
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image or handle error appropriately
            image_pil = Image.new("RGB", (224, 224), (0, 0, 0))

        # Helper to process image and text
        def process_sample(img, txt):
            if self.transform:
                pixel_values = self.transform(img)
            else:
                pixel_values = img # Should ideally be tensor already
                
            text_inputs = self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return pixel_values, text_inputs.input_ids.squeeze(0), text_inputs.attention_mask.squeeze(0)

        text = str(row[self.text_column])
        
        pixel_values_1, input_ids_1, attention_mask_1 = process_sample(image_pil, text)
        
        item = {
            "pixel_values": pixel_values_1,
            "input_ids": input_ids_1,
            "attention_mask": attention_mask_1,
            "caption": text
        }
        
        if self.return_pairs:
            # Second view (different random transform on same image)
            pixel_values_2, input_ids_2, attention_mask_2 = process_sample(image_pil, text)
            item["pixel_values_2"] = pixel_values_2
            item["input_ids_2"] = input_ids_2 # Same as 1 if deterministic, but fine
            item["attention_mask_2"] = attention_mask_2

        return item
