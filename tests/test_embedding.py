"""Pinned embedders for the matcher (engineering spec §3.7).

Covers the mechanism that a real embedder plugs into: `dense_adapter` turning a
dense encoder into the matcher's `Embedder`, `set_embedder` swapping the global
pin (and restoring it), the `EmbeddingMatcher` then computing cosine over the
dense vectors, and the friendly error when the optional model dep is absent.

The real sentence-transformers model itself is not run here (heavy dep + download,
same as the real provider / zenoh broker) — a fake dense encoder stands in.
"""

from collections.abc import Sequence

import plumbline.core.matcher as matcher_module
import pytest
from plumbline.core.matcher import EmbeddingMatcher, set_embedder
from plumbline.core.trace import Payload
from plumbline.embedding import dense_adapter, sentence_transformer_embedder

# A fake dense "model": three unit-ish vectors — warm≈hot, both ⟂ cold.
_VECTORS: dict[str, Sequence[float]] = {
    "warm": [1.0, 0.0, 0.0],
    "hot": [0.9, 0.1, 0.0],
    "cold": [0.0, 0.0, 1.0],
}


def _fake_encode(text: str) -> Sequence[float]:
    return _VECTORS[text]


def test_dense_adapter_exposes_a_vector_as_an_index_keyed_mapping() -> None:
    embed = dense_adapter(_fake_encode)
    assert embed("warm") == {"0": 1.0, "1": 0.0, "2": 0.0}


def test_matcher_uses_the_installed_dense_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Install the dense embedder via the module hook (auto-restored by monkeypatch).
    monkeypatch.setattr(matcher_module, "_embed", dense_adapter(_fake_encode))
    matcher = EmbeddingMatcher(threshold=0.2)

    warm = Payload(inline={"caption": "warm"})
    assert matcher.matches(warm, Payload(inline={"caption": "hot"})).is_match is True
    assert matcher.matches(warm, Payload(inline={"caption": "cold"})).is_match is False


def test_set_embedder_pins_globally_and_can_be_restored() -> None:
    original = matcher_module._embed
    try:
        embedder = dense_adapter(_fake_encode)
        set_embedder(embedder)
        assert matcher_module._embed is embedder
    finally:
        set_embedder(original)
    assert matcher_module._embed is original


def test_sentence_transformer_embedder_reports_the_missing_extra() -> None:
    # sentence-transformers is not installed in this environment.
    with pytest.raises(ModuleNotFoundError, match=r"plumbline\[embeddings\]"):
        sentence_transformer_embedder()
