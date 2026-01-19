import json
import logging
import os
from typing import List

import lmdb
import numpy as np
import requests

from invdiff.serve.global_vars import CLIP_CACHE_FILE, CLIP_URL
from invdiff.serve.utils_general import get_from_cache, save_to_cache

if not os.path.exists(CLIP_CACHE_FILE):
    os.makedirs(CLIP_CACHE_FILE)

clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))


def get_embeddings(inputs: List[str], model: str, modality: str) -> np.ndarray:
    input_to_embeddings = {}
    for inp in inputs:
        key = json.dumps([inp, model])
        cached_value = get_from_cache(key, clip_cache)
        if cached_value is not None:
            logging.debug(f"CLIP Cache Hit")
            input_to_embeddings[inp] = json.loads(cached_value)

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]

    BATCH_SIZE = 1024  # tune: 256/512/1024 depending on server
    TIMEOUT = 120

    if len(uncached_inputs) > 0:
        try:
            for start in range(0, len(uncached_inputs), BATCH_SIZE):
                batch = uncached_inputs[start : start + BATCH_SIZE]

                response = requests.post(
                    CLIP_URL,
                    data={modality: json.dumps(batch), "model": model},  # include model (important)
                    timeout=TIMEOUT,
                ).json()

                embs = response.get("embeddings", None)
                if embs is None or len(embs) != len(batch):
                    raise RuntimeError(
                        f"Bad CLIP response: got {None if embs is None else len(embs)} embeddings for batch size {len(batch)}"
                    )

                for inp, embedding in zip(batch, embs):
                    input_to_embeddings[inp] = embedding
                    key = json.dumps([inp, model])
                    save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            logging.error(f"CLIP Error: {e}")
            for inp in uncached_inputs:
                input_to_embeddings[inp] = None

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


if __name__ == "__main__":
    embeddings = get_embeddings(
        ["data/teaser.png"],
        "ViT-bigG-14",
        "image",
    )
    print(embeddings)

    embeddings = get_embeddings(["haha", "hello world"], "ViT-bigG-14", "text")
    print(embeddings)
