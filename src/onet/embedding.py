"""Sentence embeddings for O*NET documents and survey job titles."""

import hashlib
import logging
import sys
import time

import numpy as np
import pandas as pd

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 128
# BGE retrieval convention: queries carry an instruction, documents do not.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

log = logging.getLogger("onet.embedding")


def setup_logging(log_dir):
    """Log to the console and to a timestamped file under log_dir."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"embedding_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log.setLevel(logging.INFO)
    log.handlers = []
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(file_handler)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.info(f"Log file: {log_file}")


def log_gpu_status(prefix=""):
    import torch

    if not torch.cuda.is_available():
        log.info(f"{prefix}GPU: not available, using CPU")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log.info(f"{prefix}GPU {i} {props.name}: "
                 f"{torch.cuda.memory_allocated(i) / 1024**3:.2f}/"
                 f"{props.total_memory / 1024**3:.2f} GB allocated")


def load_model(cache_dir):
    """Load the model, downloading it into cache_dir/models on first use."""
    import torch
    from sentence_transformers import SentenceTransformer

    model_dir = cache_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {MODEL_NAME} on {device} (cache: {model_dir})")
    start = time.time()
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(model_dir), device=device)
    log.info(f"Model loaded in {time.time() - start:.1f}s, "
             f"dimension {model.get_sentence_embedding_dimension()}")
    log_gpu_status("  ")
    return model


def embed_texts(model, texts, query=False):
    """L2-normalized embeddings, one row per text."""
    texts = ["[empty]" if pd.isna(t) else str(t) for t in texts]
    if query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    return model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)


def fingerprint(texts):
    """SHA-256 of the texts in order."""
    return hashlib.sha256("\x1f".join(map(str, texts)).encode()).hexdigest()


def save_embeddings(path, keys, texts, embeddings):
    """Store embeddings with their keys and a fingerprint of the texts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.asarray(list(keys), dtype=str), embeddings=embeddings,
             texts_sha256=np.asarray(fingerprint(texts)), model=np.asarray(MODEL_NAME))


def load_embeddings(path, keys, texts):
    """Embeddings from `path`, checked against the keys and texts now on disk."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/02_embed.py first.")
    with np.load(path) as data:
        stale = (str(data["model"]) != MODEL_NAME
                 or data["keys"].tolist() != list(keys)
                 or str(data["texts_sha256"]) != fingerprint(texts))
        if stale:
            raise SystemExit(f"{path} does not match the current inputs. "
                             "Run scripts/02_embed.py again.")
        return data["embeddings"]
