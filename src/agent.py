from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved_chunks = self.store.search(question, top_k=top_k)

        if retrieved_chunks:
            context_blocks = []
            for index, chunk in enumerate(retrieved_chunks, start=1):
                source = chunk.get("metadata", {}).get("source") or chunk.get("metadata", {}).get("doc_id") or chunk.get("id")
                context_blocks.append(f"[{index}] Source: {source}\nContent: {chunk['content']}")
            context = "\n\n".join(context_blocks)
        else:
            context = "No relevant context found."

        prompt = (
            "You are a knowledge base assistant. Answer the question using only the provided context. "
            "If the context is insufficient, say so clearly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
