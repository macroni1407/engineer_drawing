import os
# import gdown
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


os.makedirs("model_weight", exist_ok=True)

# FILES = {
#     "tiny_10k.pth": "1vFx7BAcw2G1kmcTEZc2gsQO2EshBhB5y",
#     "tiny_20k.pth": "1q_gNxoHOT-l6Di3kqiADBc9snP_CxkI5",
#     "small_20k.pth": "1Iw2Xg_WHNsowgcSHb9bxbyKmZL0wponh",
# }


# for filename, file_id in FILES.items():
#     output = f"model_weight/{filename}"

#     if not os.path.exists(output):
#         url = f"https://drive.google.com/uc?id={file_id}"
#         print(f"Downloading {filename}...")
#         gdown.download(url, output, quiet=False)

snapshot_download(
    repo_id="macroni2002/mask2former_base_weights",
    repo_type="model",
    local_dir="model_weight",
    # 1. Bỏ qua file README và các file hệ thống của Hugging Face
    ignore_patterns=["README.md", ".gitattributes", ".jsonl", ".DS_Store"], 
)