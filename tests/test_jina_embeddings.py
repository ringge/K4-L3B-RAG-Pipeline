"""Exercise the Jina embedding option without making network requests."""

import sys
from types import SimpleNamespace

import pytest

from src import task4_chunking_indexing as indexing


def test_jina_embeddings_batch_and_preserve_input_order(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        data = [
            {"index": index, "embedding": [float(int(value.split("-")[1]))]}
            for index, value in enumerate(json["input"])
        ]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": data[::-1]})

    monkeypatch.setattr("requests.post", fake_post)
    texts = [f"text-{index}" for index in range(indexing.EMBEDDING_BATCH_SIZE + 1)]

    assert indexing.embed_texts(texts) == [[float(index)] for index in range(len(texts))]
    assert [len(call[2]["input"]) for call in calls] == [indexing.EMBEDDING_BATCH_SIZE, 1]
    for url, headers, payload, timeout in calls:
        assert url == "https://api.jina.ai/v1/embeddings"
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "jina-embeddings-v3"
        assert payload["task"] == "text-matching"
        assert timeout == 60


def test_jina_requires_api_key(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("JINA_API_KEY", "")
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        indexing.embed_texts(["text"])


def test_jina_uses_its_own_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setattr(indexing, "CHROMA_DIR", tmp_path)
    calls = []
    collection = object()

    class FakeClient:
        def __init__(self, *, path):
            assert path == str(tmp_path)

        def get_or_create_collection(self, **kwargs):
            calls.append(kwargs)
            return collection

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    assert indexing.get_collection() is collection
    assert calls == [{"name": "rag_documents_jina_v3", "metadata": {"hnsw:space": "cosine"}}]


def test_sentence_transformers_option_still_uses_configured_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    calls = []

    def fake_model(name):
        calls.append(name)
        return SimpleNamespace(encode=lambda texts, batch_size: SimpleNamespace(tolist=lambda: [[1.0]]))

    monkeypatch.setattr(indexing, "_get_embedding_model", fake_model)
    assert indexing.embed_texts(["text"]) == [[1.0]]
    assert calls == ["test-model"]
