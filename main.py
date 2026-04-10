from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

FALLBACK_SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]

PREFERRED_DATA_DIRS = [
    Path("data/data/data"),
    Path("data"),
]

DEFAULT_DEMO_CHUNK_SIZE = 400

FILE_METADATA = {
    "khach_hang": {"category": "customer_support", "audience": "customer"},
    "tai_xe": {"category": "driver_policy", "audience": "driver"},
    "donhang": {"category": "delivery_process", "audience": "driver"},
    "nhahang": {"category": "merchant_policy", "audience": "merchant"},
    "chính sách bảo vệ dữ liệu cá nhân": {"category": "privacy_policy", "audience": "all"},
    "điều khoản chung": {"category": "general_terms", "audience": "all"},
    "python_intro": {"category": "reference_material", "audience": "student"},
    "vector_store_notes": {"category": "reference_material", "audience": "student"},
    "rag_system_design": {"category": "reference_material", "audience": "student"},
    "customer_support_playbook": {"category": "reference_material", "audience": "support"},
    "chunking_experiment_report": {"category": "experiment_report", "audience": "student"},
    "vi_retrieval_notes": {"category": "reference_material", "audience": "student"},
}


def _configure_console_encoding() -> None:
    """Avoid UnicodeEncodeError when sample files contain Vietnamese text on Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _resolve_default_sample_files() -> list[str]:
    """Prefer the newest nested dataset when it exists, then fall back to the original sample set."""
    allowed_extensions = {".md", ".txt"}

    for data_dir in PREFERRED_DATA_DIRS:
        if not data_dir.exists() or not data_dir.is_dir():
            continue

        candidates = []
        for path in sorted(data_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in allowed_extensions:
                continue
            if path.name == ".gitkeep":
                continue

            # Skip very large policy dumps in the default demo so search previews stay readable.
            try:
                if path.stat().st_size > 120_000:
                    continue
            except OSError:
                pass

            candidates.append(str(path))

        if candidates:
            return candidates

    return list(FALLBACK_SAMPLE_FILES)


def _infer_metadata(path: Path) -> dict[str, str]:
    metadata = dict(FILE_METADATA.get(path.stem.casefold(), {}))
    metadata.setdefault("source", str(path))
    metadata.setdefault("extension", path.suffix.lower())
    return metadata


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load and chunk files for the manual RAG demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    chunker = RecursiveChunker(chunk_size=DEFAULT_DEMO_CHUNK_SIZE)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        base_metadata = _infer_metadata(path)
        chunks = chunker.chunk(content)
        chunk_count = len(chunks)

        for index, chunk_content in enumerate(chunks):
            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_index"] = index
            chunk_metadata["chunk_count"] = chunk_count
            documents.append(
                Document(
                    id=path.stem,
                    content=chunk_content,
                    metadata=chunk_metadata,
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or _resolve_default_sample_files()
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    sources: dict[str, int] = {}
    for doc in docs:
        source = doc.metadata["source"]
        sources[source] = sources.get(source, 0) + 1

    print(f"\nLoaded {len(docs)} chunks from {len(sources)} documents")
    for source, chunk_count in sorted(sources.items()):
        print(f"  - {Path(source).stem}: {source} ({chunk_count} chunks)")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        chunk_label = ""
        if "chunk_index" in result["metadata"] and "chunk_count" in result["metadata"]:
            chunk_label = (
                f" chunk={result['metadata']['chunk_index'] + 1}/"
                f"{result['metadata']['chunk_count']}"
            )
        print(
            f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}"
            f"{chunk_label}"
        )
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    _configure_console_encoding()
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
