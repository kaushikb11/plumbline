"""Pinned embedders for the EmbeddingMatcher (engineering spec §3.7).

The counterfactual matcher and the regression gate compare free text (captions,
prompts) by cosine distance between embeddings. The dependency-free default is a
bag-of-tokens vectorizer (`plumbline.core.matcher`); this module provides real
*semantic* embedders and the plumbing to pin one:

    from plumbline.embedding import sentence_transformer_embedder
    from plumbline.core.matcher import set_embedder
    set_embedder(sentence_transformer_embedder("sentence-transformers/all-MiniLM-L6-v2"))

Like the httpx and zenoh integrations, this imports the heavy dependency lazily —
nothing here is imported by the core, so `pip install plumbline` stays light.
Install the real model support with:  pip install "plumbline[embeddings]".

`set_embedder` pins the embedder *globally* so an episode and its replay use the
same one; record the model id (below) alongside the episode for reproducibility.
"""

import importlib
from collections.abc import Callable, Mapping, Sequence

from plumbline.core.matcher import Embedder, set_embedder

__all__ = ["dense_adapter", "install_sentence_transformer", "sentence_transformer_embedder"]

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def dense_adapter(encode: Callable[[str], Sequence[float]]) -> Embedder:
    """Adapt any dense encoder (`text -> vector`) to the matcher's `Embedder`.

    The dense vector is exposed as an index-keyed mapping so it flows through the
    matcher's cosine distance unchanged (the matcher normalizes, so the raw vector
    need not be unit-length).
    """

    def embed(text: str) -> Mapping[str, float]:
        return {str(index): float(value) for index, value in enumerate(encode(text))}

    return embed


def sentence_transformer_embedder(model_id: str = DEFAULT_MODEL_ID) -> Embedder:
    """A pinned sentence-transformers embedder (real semantic similarity, §3.7).

    Lazily imports sentence-transformers so the core stays dependency-free.
    """
    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the extra
        raise ModuleNotFoundError(
            "sentence_transformer_embedder needs the optional dependency: "
            "pip install 'plumbline[embeddings]'"
        ) from exc

    model = module.SentenceTransformer(model_id)

    def encode(text: str) -> Sequence[float]:
        return list(model.encode(text))

    return dense_adapter(encode)


def install_sentence_transformer(model_id: str = DEFAULT_MODEL_ID) -> str:
    """Load and pin a sentence-transformers model for the matcher; return its id."""
    set_embedder(sentence_transformer_embedder(model_id))
    return model_id
