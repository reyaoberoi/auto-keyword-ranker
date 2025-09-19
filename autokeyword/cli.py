"""Simple CLI wrapper for autokeyword."""


import argparse
from .core import rank_keywords
import sys



def main(argv=None):
    p = argparse.ArgumentParser(description="Auto Keyword Ranker CLI")
    p.add_argument("--text", help="Text (or path to .txt file) to extract keywords from", required=True)
    p.add_argument("--top", type=int, default=10, help="Number of top keywords to show")
    p.add_argument("--use-emb", action="store_true", help="Use sentence-transformer re-ranking (optional)")
    args = p.parse_args(argv)


    text = args.text
    # if looks like a path, try to read file
    try:
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        # treat as raw text
        pass


    kws = rank_keywords(text, top_n=args.top, use_embeddings=args.use_emb)
    for w, s in kws:
        print(f"{w}\t{s:.3f}")

if __name__ == "__main__":
    main()