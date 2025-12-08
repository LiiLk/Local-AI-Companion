
import sys
# Add venv site-packages
sys.path.append("./venv/lib/python3.12/site-packages")
from huggingface_hub import list_repo_files

repos = [
    "unsloth/Qwen3-VL-8B-Instruct-GGUF",
]

for repo in repos:
    print(f"\n📂 Checking {repo}...")
    try:
        files = list_repo_files(repo)
        for f in files:
            if "Q4_K_M" in f or "mmproj" in f or "q4_k_m" in f.lower():
                print(f"  - {f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
