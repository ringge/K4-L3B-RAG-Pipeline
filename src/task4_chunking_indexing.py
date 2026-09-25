"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from functools import lru_cache
from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
JINA_EMBEDDING_MODEL = "jina-embeddings-v3"
JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"

COLLECTION_NAME = "rag_documents"
JINA_COLLECTION_NAME = "rag_documents_jina_v3"
EMBEDDING_BATCH_SIZE = 32
INDEX_BATCH_SIZE = 256


@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embedding_provider() -> str:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    return os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")


def _embedding_signature() -> str:
    provider = _embedding_provider()
    if provider == "jina":
        return f"jina:{JINA_EMBEDDING_MODEL}:text-matching"
    if provider == "sentence_transformers":
        return f"sentence_transformers:{os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL)}"
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _embed_with_jina(texts: list[str]) -> list[list[float]]:
    import requests

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY is required when EMBEDDING_PROVIDER=jina")

    embeddings = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = requests.post(
            JINA_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": JINA_EMBEDDING_MODEL, "task": "text-matching", "input": batch},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(batch):
            raise ValueError("Jina embedding count does not match input count")
        embeddings.extend(item["embedding"] for item in sorted(data, key=lambda item: item["index"]))
    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the provider shared by indexing and semantic search."""
    if not texts:
        return []

    provider = _embedding_provider()
    if provider == "jina":
        return _embed_with_jina(texts)
    if provider != "sentence_transformers":
        raise ValueError(f"Unsupported embedding provider: {provider}")

    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    model = _get_embedding_model(model_name)
    return model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE).tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=JINA_COLLECTION_NAME if _embedding_provider() == "jina" else COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative_path = path.relative_to(STANDARDIZED_DIR)
        if relative_path.parts[0] not in {"legal", "news"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        headings = re.findall(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
        title = next((heading for heading in headings if heading != "Unknown"), path.stem)
        source_match = re.search(r"^\*\*Source:\*\*\s*(\S+)", content, re.MULTILINE)
        documents.append({
            "id": relative_path.as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": relative_path.parts[0],
                "url": source_match.group(1) if source_match else None,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        texts = (text for text in splitter.split_text(document["content"]) if text.strip())
        for index, text in enumerate(texts):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    embedded = []
    total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        print(f"Embedding batch {start // EMBEDDING_BATCH_SIZE + 1}/{total_batches}...", flush=True)
        vectors = embed_texts([chunk["content"] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match chunk count")
        embedded.extend({**chunk, "embedding": vector} for chunk, vector in zip(batch, vectors))
        print(f"Embedded {len(embedded)}/{len(chunks)} chunks", flush=True)
    return embedded


def _chroma_metadata(metadata: dict) -> dict:
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def _changed_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks whose text, metadata, or embedding configuration differs."""
    if not chunks:
        return []

    collection = get_collection()
    existing = {}
    for start in range(0, len(chunks), INDEX_BATCH_SIZE):
        ids = [chunk["id"] for chunk in chunks[start:start + INDEX_BATCH_SIZE]]
        records = collection.get(ids=ids, include=["documents", "metadatas"])
        existing.update(zip(records["ids"], zip(records["documents"], records["metadatas"])))

    return [
        chunk for chunk in chunks
        if existing.get(chunk["id"]) != (chunk["content"], _chroma_metadata(chunk["metadata"]))
    ]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    for start in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[start:start + INDEX_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )
        print(f"Indexed {min(start + INDEX_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks", flush=True)


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print(f"Loading documents from {STANDARDIZED_DIR}...", flush=True)
    documents = load_documents()
    print(f"Documents loaded: {len(documents)}; chunking...", flush=True)
    chunks = chunk_documents(documents)
    signature = _embedding_signature()
    chunks = [
        {**chunk, "metadata": {**chunk["metadata"], "embedding_signature": signature}}
        for chunk in chunks
    ]
    print(f"Chunks created: {len(chunks)}; checking {_embedding_provider()} index...", flush=True)
    changed_chunks = _changed_chunks(chunks)
    print(f"Unchanged: {len(chunks) - len(changed_chunks)}; embedding: {len(changed_chunks)}", flush=True)
    embedded_chunks = embed_chunks(changed_chunks)
    if embedded_chunks:
        print("Writing embeddings to ChromaDB...", flush=True)
        index_to_vectorstore(embedded_chunks)
    else:
        print("No changes to write to ChromaDB", flush=True)
    print(f"Done. Updated chunks: {len(embedded_chunks)}; total current chunks: {len(chunks)}", flush=True)


if __name__ == "__main__":
    run_pipeline()
