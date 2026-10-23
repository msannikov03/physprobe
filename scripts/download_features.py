"""Download the published features and per-clip signals into this clone (CPU-only reproduction).

Usage: python scripts/download_features.py [--revision v1.1]
Sets HF_HUB_DISABLE_XET=1 before importing huggingface_hub: the anonymous Hugging Face API quota
(500 calls per 5 min) is exhausted by the xet transfer path, which costs one call per file (6,700 files).
Safe to rerun: finished files are skipped.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="v1.1")
    ap.add_argument("--repo", default="businesslion/physprobe-features")
    ap.add_argument("--tries", type=int, default=4)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    for k in range(1, a.tries + 1):
        try:
            p = snapshot_download(a.repo, repo_type="dataset", revision=a.revision, local_dir=ROOT,
                                  allow_patterns=["data/*", "features/*"])
            print("downloaded to", p)
            return 0
        except Exception as e:  # transient 429 / SSL timeouts: wait and resume
            print(f"attempt {k}/{a.tries} failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            if k == a.tries:
                return 1
            time.sleep(60 * k)


if __name__ == "__main__":
    sys.exit(main())
