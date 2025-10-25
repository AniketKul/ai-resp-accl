# download_models.py
from huggingface_hub import snapshot_download
import os

def download_models():
    """Download Qwen2.5 draft and target models"""
    
    models = {
        "draft": "Qwen/Qwen2.5-7B-Instruct",
        "target": "Qwen/Qwen2.5-72B-Instruct-AWQ"
    }
    
    for name, model_id in models.items():
        print(f"\n{'='*70}")
        print(f"📦 Downloading {name.upper()} model: {model_id}")
        print(f"{'='*70}")
        
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=f"./models/{name}",
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.md", "*.txt", "*.pdf"]  # Skip docs
            )
            print(f"✅ {name.upper()} model downloaded successfully!")
            
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            print("Make sure you have access to Qwen models on HuggingFace")
    
    print(f"\n{'='*70}")
    print("✅ All models downloaded!")
    print(f"{'='*70}")
    print(f"\nModels saved in: ./models/")
    print(f"  - Draft:  ./models/draft/  (~14 GB)")
    print(f"  - Target: ./models/target/ (~40 GB - AWQ quantized)")

if __name__ == "__main__":
    download_models()
