import sys
import os
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import FashionCLIPDataset
from src.data.transforms import get_transforms

def verify_data_loading():
    print("Verifying Data Loading...")
    
    # 1. Create Dummy Data
    os.makedirs("data/dummy_images", exist_ok=True)
    dummy_image_path = "data/dummy_images/test_image.jpg"
    Image.new('RGB', (500, 500), color='red').save(dummy_image_path)
    
    dummy_metadata = pd.DataFrame({
        "image_path": ["dummy_images/test_image.jpg"],
        "caption": ["A red fashion item."]
    })
    metadata_path = "data/dummy_metadata.csv"
    dummy_metadata.to_csv(metadata_path, index=False)
    
    # 2. Initialize Components
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    train_transform, _ = get_transforms(img_size=224)
    
    # 3. Initialize Dataset
    dataset = FashionCLIPDataset(
        image_root_dir="data",
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        transform=train_transform
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # 4. Fetch Sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["pixel_values"].shape)
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention Mask shape:", sample["attention_mask"].shape)
    print("Caption:", sample["caption"])
    
    # 5. Cleanup
    os.remove(dummy_image_path)
    os.remove(metadata_path)
    os.rmdir("data/dummy_images")
    os.rmdir("data")
    print("Verification Successful!")

if __name__ == "__main__":
    verify_data_loading()
