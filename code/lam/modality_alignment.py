import os
import numpy as np
import torch
import whisper
from intent_parser import IntentParser

class ModalityAlignment:
    def __init__(
        self,
        whisper_model_dir: str = r"D:\MP2\models\whisper",
        whisper_model_size: str = "base",
        intent_parser: IntentParser = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ModalityAlignment] Loading Whisper {whisper_model_size} on {self.device}")
        self.whisper_model = whisper.load_model(
            whisper_model_size,
            device=self.device,
            download_root=whisper_model_dir,
        )
        self.intent_parser = intent_parser or IntentParser()
        print("[ModalityAlignment] Ready")

    def _text_to_description(self, text: str) -> str:
        return text.strip()

    def _image_to_description(self, image_path: str) -> str:
        # Use Qwen2-VL via intent_parser's llm for image captioning
        try:
            from llama_cpp import Llama
            prompt = (
                f"Describe this image in one sentence suitable for communication intent: "
                f"[Image: {os.path.basename(image_path)}]. "
                "Output only the description."
            )
            response = self.intent_parser.llm(
                prompt, max_tokens=64, temperature=0.1
            )
            desc = response["choices"][0]["text"].strip()
            return desc if desc else f"image from {os.path.basename(image_path)}"
        except Exception as e:
            print(f"[ModalityAlignment] Image description fallback: {e}")
            return f"high quality image content from {os.path.basename(image_path)}"

    def _audio_to_text(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    def align(self, input_data, modality: str = "text") -> dict:
        modality = modality.lower()

        if modality == "text":
            description = self._text_to_description(input_data)
        elif modality == "image":
            description = self._image_to_description(input_data)
        elif modality == "audio":
            description = self._audio_to_text(input_data)
        else:
            raise ValueError(f"Unknown modality: {modality}. Use text/image/audio.")

        intent_result = self.intent_parser.parse(description)

        return {
            "original_input": input_data,
            "modality": modality,
            "unified_text": description,
            "intent_vector": intent_result["intent_vector"],
            "delay_intent": intent_result["delay_intent"],
            "quality_intent": intent_result["quality_intent"],
        }


if __name__ == "__main__":
    aligner = ModalityAlignment()

    # Test text
    result = aligner.align("Send it within 1 second", modality="text")
    print(f"\nText modality:")
    print(f"  Unified text: {result['unified_text']}")
    print(f"  Intent vector: {result['intent_vector']}")

    # Test image (placeholder)
    result = aligner.align("D:/MP2/test_image.jpg", modality="image")
    print(f"\nImage modality:")
    print(f"  Unified text: {result['unified_text']}")
    print(f"  Intent vector: {result['intent_vector']}")
