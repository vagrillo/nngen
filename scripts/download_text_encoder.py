"""Pre-download the frozen MiniLM text encoder into the HuggingFace cache.

Usage:
    python scripts/download_text_encoder.py
"""

from sentence_transformers import SentenceTransformer  # noqa: F401 (warms the ST cache)
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    print(f"Downloading {MODEL} ...")
    AutoTokenizer.from_pretrained(MODEL)
    AutoModel.from_pretrained(MODEL)
    SentenceTransformer(MODEL)
    print("Done.")


if __name__ == "__main__":
    main()
