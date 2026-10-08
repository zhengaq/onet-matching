#!/usr/bin/env python3
"""Stage 02: embed the O*NET documents and the cleaned job titles.

Writes onet_embeddings.npz and <survey>_job_embeddings.npz to the cache. Each
file stores the texts it was computed from, so stage 03 refuses a cache that
no longer matches the stage 01 outputs. Downloads the model on first use.
"""

import argparse
import time

import pandas as pd

from onet.config import load_settings, require
from onet.embedding import (embed_texts, load_model, log, log_gpu_status,
                            save_embeddings, setup_logging)
from onet.survey import read_clean


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    settings = load_settings()
    require(settings.onet_docs, settings.survey_clean)
    setup_logging(settings.log_dir)
    model = load_model(settings.cache_dir)

    onet = pd.read_csv(settings.onet_docs, dtype=str, keep_default_na=False)
    people = read_clean(settings.survey_clean)
    for name, keys, texts, query, path in [
        ("O*NET documents", onet.onet_code, onet.doc_text, False, settings.onet_embeddings),
        ("job titles", people.participant_id, people.job_title_clean, True, settings.survey_embeddings),
    ]:
        start = time.time()
        embeddings = embed_texts(model, texts.tolist(), query=query)
        save_embeddings(path, keys, texts, embeddings)
        log.info(f"Embedded {len(texts)} {name} in {time.time() - start:.1f}s -> {path}")
    log_gpu_status()


if __name__ == "__main__":
    main()
