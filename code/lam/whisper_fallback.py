"""
Whisper Fallback — ffmpeg-free transcription using HuggingFace transformers.
Use when openai-whisper fails due to missing ffmpeg.
"""
import torch
import numpy as np


class WhisperFallback:
    """ffmpeg-free Whisper transcription using HuggingFace transformers."""

    def __init__(self, model_name: str = "openai/whisper-base", device: str = None):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[WhisperFallback] Loading {model_name} on {self.device}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("[WhisperFallback] Ready")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file. Supports .wav natively without ffmpeg."""
        import soundfile as sf
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            return {"text": f"[transcription failed: {e}]", "error": str(e)}

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Stereo to mono

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": transcription}


if __name__ == "__main__":
    import glob
    w = WhisperFallback()
    audio_files = glob.glob(r"D:\MP2\data\raw\audio\*.wav")[:1]
    if audio_files:
        result = w.transcribe(audio_files[0])
        print(f"File: {audio_files[0]}")
        print(f"Transcription: {result['text'][:100]}")
    else:
        print("No .wav files found")
