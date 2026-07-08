"""
RAG-Enhanced Intent Parser for CSCA Left Brain.
Implements Section III.B.1 of Sun et al. 2026.

Extends IntentParser with:
- RAG retrieval from LKB (Local Knowledge Base)
- CohereRerank via cross-encoder
- Self-RAG reflection tokens (ISREL, ISSUP)
- Confidence scoring
"""

import json
import re
import numpy as np
from llama_cpp import Llama
from lkb import LocalKnowledgeBase

INTENT_PROMPT_WITH_RAG = """You are a communication intent parser. Given the user's request and similar examples from the knowledge base, convert the intent into a structured format.

Similar examples from knowledge base:
{retrieved_context}

User intent: {intent_text}

Based on the examples above, extract the communication intent as a JSON object with fields:
- "time": delay requirement in seconds (float)
- "quality": data similarity quality 0.0-1.0 (float)
- "size": estimated data size in kB (float, use 200 if not specified)
- "dest": destination (string, use "receiver" if not specified)

Reply with ONLY a JSON object like {{"time": 1.0, "quality": 0.9, "size": 200, "dest": "receiver"}}"""

ISREL_PROMPT = """You retrieved the following knowledge base entries for the user's intent:

{retrieved_context}

User intent: {intent_text}

Are the retrieved entries relevant to the user's intent? Consider whether the examples help understand what the user wants.
Answer with exactly one token: ISREL_YES or ISREL_NO"""

ISSUP_PROMPT = """User intent: {intent_text}

Your generated output: {generated_output}

Is your output supported by and consistent with the user's intent? Does it accurately reflect what the user asked for?
Answer with exactly one token: ISSUP_YES or ISSUP_NO"""

DELAY_MAX = 10.0
QUALITY_MAX = 1.0


class RAGIntentParser:
    """
    RAG-enhanced intent parser implementing the paper's left brain pipeline:
    1. Retrieve relevant examples from LKB using cosine similarity + reranking
    2. Check ISREL (are retrieved examples relevant?)
    3. Generate intent using LAM with augmented prompt
    4. Check ISSUP (is generated output supported?)
    5. If ISREL_NO or ISSUP_NO: retry with different k
    """

    def __init__(
        self,
        model_path: str = r"D:\MP2\models\Qwen.Qwen2-VL-7B.Q4_K_M.gguf",
        n_gpu_layers: int = 28,
        n_ctx: int = 1024,
        verbose: bool = False,
        lkb: LocalKnowledgeBase = None,
    ):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self.lkb = lkb or LocalKnowledgeBase()
        print(f"[RAGIntentParser] Model loaded: {model_path}")

    def _llm_generate(self, prompt: str, max_tokens: int = 128, stop=None) -> str:
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            stop=stop or ["\n\n", "Intent:", "Note:"],
        )
        return response["choices"][0]["text"].strip()

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from LLM output."""
        text = text.strip()
        # Try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try extracting JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # Try finding JSON-like pattern
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _check_token(self, prompt: str, expected_yes: str, expected_no: str) -> bool:
        """Run ISREL or ISSUP check. Returns True if YES, False if NO."""
        output = self._llm_generate(prompt, max_tokens=10, stop=["\n", "."])
        output_upper = output.upper().strip()
        if expected_yes in output_upper:
            return True
        elif expected_no in output_upper:
            return False
        # Default to YES if unclear
        print(f"[RAGIntentParser] WARNING: Unclear reflection token: {output!r}")
        return True

    def _normalize(self, delay_s: float, quality: float):
        delay_norm = float(np.clip(1.0 - (delay_s / DELAY_MAX), 0.0, 1.0))
        quality_norm = float(np.clip(quality, 0.0, 1.0))
        return [delay_norm, quality_norm]

    def parse(self, intent_text: str, max_retries: int = 2) -> dict:
        """
        Full RAG pipeline for intent parsing.
        1. Retrieve from LKB
        2. ISREL check
        3. Generate intent with LAM
        4. ISSUP check
        5. Retry if checks fail
        """
        for attempt in range(max_retries + 1):
            k = 3 + attempt  # Increase k on retry

            # Step 1: RAG retrieval from LKB
            retrieved = self.lkb.retrieve(intent_text, k=k)
            context_str = self.lkb.format_retrieved(retrieved)

            # Step 2: ISREL check (are retrieved examples relevant?)
            isrel_prompt = ISREL_PROMPT.format(
                retrieved_context=context_str,
                intent_text=intent_text,
            )
            is_relevant = self._check_token(isrel_prompt, "ISREL_YES", "ISREL_NO")

            # Step 3: Generate intent with augmented prompt
            aug_prompt = INTENT_PROMPT_WITH_RAG.format(
                retrieved_context=context_str,
                intent_text=intent_text,
            )
            raw_output = self._llm_generate(aug_prompt, max_tokens=128)

            # Step 4: ISSUP check (is output supported by intent?)
            issup_prompt = ISSUP_PROMPT.format(
                intent_text=intent_text,
                generated_output=raw_output,
            )
            is_supported = self._check_token(issup_prompt, "ISSUP_YES", "ISSUP_NO")

            # Parse the output
            parsed = self._extract_json(raw_output)

            if parsed and is_relevant and is_supported:
                # Success
                delay_s = float(parsed.get("time", 1.0))
                quality = float(parsed.get("quality", 0.8))
                delay_norm, quality_norm = self._normalize(delay_s, quality)

                return {
                    "intent_text": intent_text,
                    "raw_output": raw_output,
                    "parsed_intent": parsed,
                    "delay_seconds": delay_s,
                    "quality_similarity": quality,
                    "intent_vector": [delay_norm, quality_norm],
                    "delay_intent": delay_norm,
                    "quality_intent": quality_norm,
                    "retrieved_examples": [r["text"] for r in retrieved],
                    "isrel": is_relevant,
                    "issup": is_supported,
                    "confidence": self._compute_confidence(retrieved, is_relevant, is_supported),
                    "attempt": attempt + 1,
                }

            # If checks failed, retry with more examples
            if not is_relevant:
                print(f"[RAGIntentParser] ISREL_NO on attempt {attempt+1}, retrying with k={k+2}")
            if not is_supported:
                print(f"[RAGIntentParser] ISSUP_NO on attempt {attempt+1}, retrying with k={k+2}")
            if not parsed:
                print(f"[RAGIntentParser] Parse failed on attempt {attempt+1}, retrying")

        # Fallback: return best effort
        delay_s = 1.0
        quality = 0.8
        delay_norm, quality_norm = self._normalize(delay_s, quality)
        return {
            "intent_text": intent_text,
            "raw_output": raw_output if 'raw_output' in dir() else "",
            "parsed_intent": {"time": delay_s, "quality": quality},
            "delay_seconds": delay_s,
            "quality_similarity": quality,
            "intent_vector": [delay_norm, quality_norm],
            "delay_intent": delay_norm,
            "quality_intent": quality_norm,
            "retrieved_examples": [],
            "isrel": False,
            "issup": False,
            "confidence": 0.0,
            "attempt": max_retries + 1,
        }

    def _compute_confidence(self, retrieved: list, is_relevant: bool, is_supported: bool) -> float:
        """Compute confidence score based on retrieval quality and reflection tokens."""
        if not retrieved:
            return 0.0
        avg_cos = np.mean([r["cos_score"] for r in retrieved])
        avg_rerank = np.mean([r["rerank_score"] for r in retrieved])
        # Normalize rerank score (cross-encoder scores are typically 0-10)
        rerank_norm = min(avg_rerank / 10.0, 1.0)
        base_score = 0.4 * avg_cos + 0.3 * rerank_norm
        if is_relevant:
            base_score += 0.15
        if is_supported:
            base_score += 0.15
        return float(np.clip(base_score, 0.0, 1.0))


if __name__ == "__main__":
    parser = RAGIntentParser()

    test_intents = [
        "Send it within 1 second",
        "Send it with high resolution",
        "Send the data reliably even if slow",
        "I need this delivered urgently to device b",
        "Standard quality, no rush at all",
    ]

    print("=" * 60)
    print("RAG INTENT PARSER TEST")
    print("=" * 60)

    for intent in test_intents:
        print(f"\n{'='*60}")
        print(f"Intent: {intent}")
        result = parser.parse(intent)

        print(f"  Retrieved examples:")
        for ex in result["retrieved_examples"]:
            print(f"    - {ex}")
        print(f"  ISREL: {'YES' if result['isrel'] else 'NO'}")
        print(f"  ISSUP: {'YES' if result['issup'] else 'NO'}")
        print(f"  Raw output: {result['raw_output']}")
        print(f"  Intent vector: [{result['delay_intent']:.3f}, {result['quality_intent']:.3f}]")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Attempt: {result['attempt']}")
