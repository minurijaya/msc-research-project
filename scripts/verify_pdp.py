import sys
import os
import pandas as pd
from PIL import Image
import torch
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.infer_pdp import infer

# Mock args class
class Args:
    pass

def verify_pdp():
    print("Verifying PDP Inference...")
    
    # 1. Setup Dummy Data
    os.makedirs("data/dummy_pdp", exist_ok=True)
    
    # Catalog Images
    img1 = "data/dummy_pdp/item1.jpg"
    img2 = "data/dummy_pdp/item2.jpg"
    Image.new('RGB', (224, 224), color='red').save(img1)
    Image.new('RGB', (224, 224), color='blue').save(img2)
    
    # Catalog CSV
    pd.DataFrame({
        "image_path": ["dummy_pdp/item1.jpg", "dummy_pdp/item2.jpg"],
        "caption": ["Red Item", "Blue Item"]
    }).to_csv("data/pdp_catalog.csv", index=False)
    
    # Query Image
    query_img = "data/dummy_pdp/query.jpg"
    Image.new('RGB', (224, 224), color='darkred').save(query_img)
    
    # 2. Mock Arguments
    args = Args()
    args.image_dir = "data"
    args.catalog_csv = "data/pdp_catalog.csv"
    args.model_path = "openai/clip-vit-base-patch32" # Use base clip for dry run
    args.query_image = query_img
    args.query_text = "Dark red item"
    args.projection_dim = 256
    args.top_k = 2
    
    # 3. Run Inference
    try:
        infer(args)
        print("\nVerification Successful!")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # 4. Cleanup
    shutil.rmtree("data/dummy_pdp")
    os.remove("data/pdp_catalog.csv")
    try: os.rmdir("data") 
    except: pass

if __name__ == "__main__":
    verify_pdp()
