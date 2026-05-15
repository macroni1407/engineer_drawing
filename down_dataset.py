import os
from huggingface_hub import snapshot_download

os.makedirs("bom_set", exist_ok=True)

snapshot_download(
    repo_id="macroni2002/bom_dataset",
    repo_type="dataset",
    local_dir="bom_set",
    # 1. Bỏ qua file README và các file hệ thống của Hugging Face
    ignore_patterns=["README.md", ".gitattributes", ".jsonl"], 
)