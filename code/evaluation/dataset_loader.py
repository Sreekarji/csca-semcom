import os
import json
import torch
import numpy as np
from pathlib import Path


class DatasetLoader:
    """
    Dataset loader for CSCA multimodal evaluation.
    Loads text (SST), audio (LibriSpeech), and images (CIFAR-10/synthetic).
    """

    def __init__(
        self,
        text_path: str = r"D:\MP2\data\raw\sst_sentences.json",
        audio_dir: str = r"D:\MP2\data\raw\audio",
        image_dir: str = r"D:\MP2\data\raw\images_google",
    ):
        self.text_path = text_path
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self._text_data = None
        self._audio_files = None
        self._image_files = None

    def load_text(self, n: int = None) -> list:
        if self._text_data is None:
            if not os.path.exists(self.text_path):
                raise FileNotFoundError(f"Text data not found: {self.text_path}")
            with open(self.text_path, "r") as f:
                self._text_data = json.load(f)
        data = self._text_data
        if n is not None:
            data = data[:n]
        return [item["text"] if isinstance(item, dict) else item for item in data]

    def load_audio(self, n: int = None) -> list:
        if self._audio_files is None:
            if not os.path.exists(self.audio_dir):
                raise FileNotFoundError(f"Audio dir not found: {self.audio_dir}")
            exts = {".wav", ".mp3", ".flac", ".ogg"}
            self._audio_files = sorted([
                str(p) for p in Path(self.audio_dir).rglob("*")
                if p.suffix.lower() in exts
            ])
        files = self._audio_files
        if n is not None:
            files = files[:n]
        return files

    def load_images(self, n: int = None) -> list:
        if self._image_files is None:
            if not os.path.exists(self.image_dir):
                raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
            exts = {".jpg", ".jpeg", ".png"}
            self._image_files = sorted([
                str(p) for p in Path(self.image_dir).rglob("*")
                if p.suffix.lower() in exts
            ])
        files = self._image_files
        if n is not None:
            files = files[:n]
        return files

    def get_batch(self, modality: str, n: int = 10) -> list:
        modality = modality.lower()
        if modality == "text":
            data = self.load_text()
        elif modality == "audio":
            data = self.load_audio()
        elif modality == "image":
            data = self.load_images()
        else:
            raise ValueError(f"Unknown modality: {modality}")
        indices = np.random.choice(len(data), min(n, len(data)), replace=False)
        return [data[i] for i in indices]

    def report(self):
        print(f"Text samples: {len(self.load_text())}")
        print(f"Audio files: {len(self.load_audio())}")
        print(f"Image files: {len(self.load_images())}")


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.report()
    print("\nSample text:", loader.get_batch("text", 3))
    print("Sample audio:", loader.get_batch("audio", 3))
    print("Sample images:", loader.get_batch("image", 3))
