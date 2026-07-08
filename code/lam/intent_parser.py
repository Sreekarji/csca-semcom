import json
import re
import numpy as np
from llama_cpp import Llama

INTENT_PROMPT = (
    "This sentence may contain the user's intent regarding delay and quality. "
    "Please convert it into an array, where the first number indicates the time "
    "(in seconds) within which the user expects the task to be completed, and the "
    "second number represents the desired communication quality, expressed as data "
    "similarity. Reply with ONLY a JSON array like [1.0, 0.9]. "
    "Intent: {intent}"
)

DELAY_MAX = 10.0
QUALITY_MAX = 1.0

class IntentParser:
    def __init__(
        self,
        model_path: str = r"D:\MP2\models\Qwen.Qwen2-VL-7B.Q4_K_M.gguf",
        n_gpu_layers: int = 28,
        n_ctx: int = 512,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        print(f"[IntentParser] Model loaded: {model_path}")

    def _extract_array(self, text: str):
        # Try direct JSON parse first
        text = text.strip()
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and len(arr) >= 2:
                return arr[:2]
        except Exception:
            pass
        # Regex fallback
        match = re.search(r"\[([0-9.]+)[,\s]+([0-9.]+)\]", text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        # Single number fallback
        nums = re.findall(r"[0-9]+\.?[0-9]*", text)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
        return None

    def _normalize(self, delay_s: float, quality: float):
        delay_norm = float(np.clip(1.0 - (delay_s / DELAY_MAX), 0.0, 1.0))
        quality_norm = float(np.clip(quality, 0.0, 1.0))
        return [delay_norm, quality_norm]

    def parse(self, intent_text: str) -> dict:
        prompt = INTENT_PROMPT.format(intent=intent_text)
        response = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.1,
            stop=["\n", "Intent:", "Note:"],
        )
        raw = response["choices"][0]["text"].strip()
        arr = self._extract_array(raw)

        if arr is None:
            print(f"[IntentParser] WARNING: Could not parse output: {raw!r}")
            arr = [1.0, 0.8]

        delay_s, quality = arr[0], arr[1]
        delay_norm, quality_norm = self._normalize(delay_s, quality)

        return {
            "intent_text": intent_text,
            "raw_output": raw,
            "delay_seconds": delay_s,
            "quality_similarity": quality,
            "intent_vector": [delay_norm, quality_norm],
            "delay_intent": delay_norm,
            "quality_intent": quality_norm,
        }

    def parse_batch(self, intents: list) -> list:
        return [self.parse(i) for i in intents]


if __name__ == "__main__":
    parser = IntentParser()
    test_intents = [
        "Send it within 1 second",
        "Send it with high resolution",
        "Send the data reliably even if slow",
    ]
    for intent in test_intents:
        result = parser.parse(intent)
        print(f"\nIntent: {result['intent_text']}")
        print(f"Raw output: {result['raw_output']}")
        print(f"Delay intent: {result['delay_intent']:.3f}")
        print(f"Quality intent: {result['quality_intent']:.3f}")
        print(f"Intent vector: {result['intent_vector']}")
