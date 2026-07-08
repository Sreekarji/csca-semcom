"""
Local Knowledge Base (LKB) for CSCA Left Brain.
Implements Section III.B.1 of Sun et al. 2026.

Stores intent templates in IG1253-like format, supports RAG retrieval
with cosine similarity + cross-encoder reranking.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder


# Default intent templates in IG1253-like format
DEFAULT_TEMPLATES = [
    # Low-latency intents
    {"text": "send it within 1 second", "intent": {"dest": "receiver", "time": 1.0, "size": 200, "quality": 0.8}},
    {"text": "deliver urgently, no delay", "intent": {"dest": "receiver", "time": 0.5, "size": 100, "quality": 0.7}},
    {"text": "real-time transmission needed", "intent": {"dest": "receiver", "time": 0.1, "size": 50, "quality": 0.6}},
    {"text": "send it as fast as possible", "intent": {"dest": "receiver", "time": 0.3, "size": 150, "quality": 0.7}},
    {"text": "low latency delivery required", "intent": {"dest": "receiver", "time": 0.5, "size": 100, "quality": 0.8}},
    {"text": "stream this in real time", "intent": {"dest": "receiver", "time": 0.05, "size": 500, "quality": 0.6}},
    {"text": "time-critical: send immediately", "intent": {"dest": "receiver", "time": 0.2, "size": 80, "quality": 0.7}},
    # High-quality intents
    {"text": "send it with high resolution", "intent": {"dest": "receiver", "time": 5.0, "size": 2000, "quality": 0.95}},
    {"text": "lossless transmission required", "intent": {"dest": "receiver", "time": 10.0, "size": 5000, "quality": 1.0}},
    {"text": "need accurate data, no corruption", "intent": {"dest": "receiver", "time": 8.0, "size": 1000, "quality": 0.98}},
    {"text": "high fidelity audio needed", "intent": {"dest": "receiver", "time": 3.0, "size": 800, "quality": 0.95}},
    {"text": "send the photo in full quality", "intent": {"dest": "receiver", "time": 5.0, "size": 3000, "quality": 0.99}},
    {"text": "ensure maximum image clarity", "intent": {"dest": "receiver", "time": 6.0, "size": 4000, "quality": 0.97}},
    # Balanced intents
    {"text": "send it reliably, moderate speed", "intent": {"dest": "receiver", "time": 3.0, "size": 500, "quality": 0.85}},
    {"text": "best effort delivery is fine", "intent": {"dest": "receiver", "time": 5.0, "size": 300, "quality": 0.7}},
    {"text": "normal priority, standard quality", "intent": {"dest": "receiver", "time": 4.0, "size": 400, "quality": 0.75}},
    {"text": "balanced latency and quality", "intent": {"dest": "receiver", "time": 2.0, "size": 600, "quality": 0.85}},
    {"text": "send reliably even if slow", "intent": {"dest": "receiver", "time": 10.0, "size": 500, "quality": 0.9}},
    {"text": "standard communication, no rush", "intent": {"dest": "receiver", "time": 8.0, "size": 300, "quality": 0.8}},
    {"text": "send to device b with moderate speed", "intent": {"dest": "b", "time": 3.0, "size": 400, "quality": 0.8}},
]


class LocalKnowledgeBase:
    """
    Local Knowledge Base (LKB) for CSCA.
    Stores intent templates, supports RAG retrieval with:
    1. SentenceTransformer embedding + cosine similarity (initial retrieval)
    2. Cross-encoder reranking (CohereRerank substitute)
    3. Knowledge chunking for LAM context window
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        templates: list = None,
        chunk_size: int = 512,
    ):
        self.chunk_size = chunk_size

        # Embedding model for cosine similarity retrieval
        print(f"[LKB] Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Cross-encoder reranker (CohereRerank substitute)
        print(f"[LKB] Loading reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        # Load templates
        self.templates = templates or DEFAULT_TEMPLATES
        self.texts = [t["text"] for t in self.templates]

        # Pre-encode all template texts
        print(f"[LKB] Encoding {len(self.templates)} intent templates...")
        self.embeddings = self.embedder.encode(self.texts, convert_to_tensor=True)
        print(f"[LKB] Ready. {len(self.templates)} templates loaded.")

    def retrieve(self, query_text: str, k: int = 3) -> list:
        """
        RAG retrieval: cosine similarity + cross-encoder reranking.
        Returns top-k most relevant templates with scores.
        """
        # Step 1: Cosine similarity retrieval (top 2*k candidates)
        query_emb = self.embedder.encode(query_text, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), self.embeddings
        )
        top_indices = torch.topk(cos_scores, k=min(2 * k, len(self.templates))).indices.tolist()

        # Step 2: Cross-encoder reranking (CohereRerank substitute)
        candidates = [(query_text, self.texts[i]) for i in top_indices]
        rerank_scores = self.reranker.predict(candidates)

        # Sort by rerank score
        scored = sorted(zip(top_indices, rerank_scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:k]:
            results.append({
                "text": self.texts[idx],
                "intent": self.templates[idx]["intent"],
                "cos_score": float(cos_scores[idx]),
                "rerank_score": float(score),
            })

        return results

    def chunk_knowledge(self, max_chunk_chars: int = None) -> list:
        """
        Split stored knowledge into chunks that fit LAM context window.
        Each chunk is a formatted string of intent examples.
        """
        if max_chunk_chars is None:
            max_chunk_chars = self.chunk_size

        chunks = []
        current_chunk = ""
        for t in self.templates:
            entry = f"Example: \"{t['text']}\" -> {t['intent']}\n"
            if len(current_chunk) + len(entry) > max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = entry
            else:
                current_chunk += entry
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def add_template(self, text: str, intent: dict):
        """Add a new intent template to the LKB."""
        self.templates.append({"text": text, "intent": intent})
        self.texts.append(text)
        new_emb = self.embedder.encode(text, convert_to_tensor=True)
        self.embeddings = torch.cat([self.embeddings, new_emb.unsqueeze(0)], dim=0)

    def format_retrieved(self, retrieved: list) -> str:
        """Format retrieved templates as context string for LAM prompt."""
        lines = []
        for i, r in enumerate(retrieved):
            lines.append(
                f"  {i+1}. \"{r['text']}\" -> intent: {r['intent']} "
                f"(similarity: {r['cos_score']:.3f}, rerank: {r['rerank_score']:.3f})"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    lkb = LocalKnowledgeBase()

    # Test retrieval
    query = "send photo quickly with high quality"
    print(f"\nQuery: {query}")
    results = lkb.retrieve(query, k=3)
    print("Top-3 retrieved:")
    for r in results:
        print(f"  \"{r['text']}\" -> {r['intent']}")
        print(f"    cos={r['cos_score']:.3f}, rerank={r['rerank_score']:.3f}")

    # Test chunking
    chunks = lkb.chunk_knowledge()
    print(f"\nKnowledge chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(c)} chars, {c.count('Example')} examples")
