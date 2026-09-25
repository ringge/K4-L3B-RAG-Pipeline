"""Verify reruns avoid embedding unchanged chunks."""

from src import task4_chunking_indexing as indexing


def test_pipeline_skips_unchanged_chunks_and_reembeds_changes(monkeypatch):
    stored = {}
    embedded_inputs = []
    upserted_ids = []
    contents = ["first", "second"]

    class FakeCollection:
        def get(self, *, ids, include):
            assert include == ["documents", "metadatas"]
            found = [item_id for item_id in ids if item_id in stored]
            return {
                "ids": found,
                "documents": [stored[item_id][0] for item_id in found],
                "metadatas": [stored[item_id][1] for item_id in found],
            }

        def upsert(self, *, ids, documents, embeddings, metadatas):
            upserted_ids.extend(ids)
            for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
                stored[item_id] = (document, metadata)

    collection = FakeCollection()
    monkeypatch.setattr(indexing, "_embedding_provider", lambda: "jina")
    monkeypatch.setattr(indexing, "get_collection", lambda: collection)
    monkeypatch.setattr(indexing, "load_documents", lambda: [{"id": "doc"}])
    monkeypatch.setattr(indexing, "chunk_documents", lambda docs: [
        {"id": f"doc::chunk-{index}", "content": content, "metadata": {"source": "doc.md"}}
        for index, content in enumerate(contents)
    ])

    def fake_embed(texts):
        embedded_inputs.extend(texts)
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(indexing, "embed_texts", fake_embed)

    indexing.run_pipeline()
    assert embedded_inputs == ["first", "second"]
    assert upserted_ids == ["doc::chunk-0", "doc::chunk-1"]

    indexing.run_pipeline()
    assert embedded_inputs == ["first", "second"]
    assert upserted_ids == ["doc::chunk-0", "doc::chunk-1"]

    contents[1] = "revised"
    indexing.run_pipeline()
    assert embedded_inputs == ["first", "second", "revised"]
    assert upserted_ids == ["doc::chunk-0", "doc::chunk-1", "doc::chunk-1"]
