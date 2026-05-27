"""Download a model snapshot from ModelScope to /root/autodl-tmp/LLM-Research/<short-name>.

Skips redundant artifacts (.pth duplicates, .gguf, original/, .bin partials when safetensors are present).
Resumes partial downloads.
"""
import argparse
import sys
from pathlib import Path

from modelscope import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="ModelScope repo, e.g. LLM-Research/Meta-Llama-3.1-8B")
    p.add_argument("--cache-dir", default="/root/autodl-tmp", help="Cache root")
    args = p.parse_args()

    print(f"=== downloading {args.repo} → {args.cache_dir} ===")
    try:
        path = snapshot_download(
            args.repo,
            cache_dir=args.cache_dir,
            ignore_file_pattern=[
                r".*\.pth$",       # PyTorch .pth duplicates (we use safetensors)
                r"original/.*",     # Meta's original pickle files
                r".*\.gguf$",       # GGUF quantizations
                r".*-of-.*\.bin$",  # Sharded pickle .bin (use safetensors instead)
                r"consolidated\..*", # Meta's consolidated.* files
            ],
        )
        print(f"=== done. local path: {path} ===")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
