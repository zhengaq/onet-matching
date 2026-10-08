import numpy as np
import pytest

from onet.embedding import load_embeddings, save_embeddings


def test_cache_must_match_inputs(tmp_path):
    path = tmp_path / "e.npz"
    vectors = np.eye(2, dtype=np.float32)
    save_embeddings(path, ["001", "002"], ["nurse", "welder"], vectors)
    assert np.array_equal(load_embeddings(path, ["001", "002"], ["nurse", "welder"]), vectors)
    with pytest.raises(SystemExit):
        load_embeddings(path, ["001", "002"], ["nurse", "electrician"])
    with pytest.raises(SystemExit):
        load_embeddings(path, ["002", "001"], ["nurse", "welder"])
    with pytest.raises(SystemExit):
        load_embeddings(tmp_path / "missing.npz", [], [])
