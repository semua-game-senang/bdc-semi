"""Frozen image/text encoders. No training and no evaluation labels."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from .common import save_matrix, unit_rows


def _as_tensor(result):
    return result if hasattr(result, "detach") else result.pooler_output


class ClipBackend:
    def __init__(self, model_id: str, revision: str | None, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)
        if not self.revision:
            raise ValueError("model revision could not be resolved; provide a pinned revision")

    def encode_images(self, rows: list[dict], batch_size: int = 32) -> np.ndarray:
        chunks = []
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(row["path"]) as opened:
                    images.append(opened.convert("RGB"))
            batch = self.processor(images=images, return_tensors="pt")
            with self.torch.inference_mode():
                result = self.model.get_image_features(pixel_values=batch["pixel_values"].to(self.device))
            chunks.append(_as_tensor(result).float().cpu().numpy())
            for image in images:
                image.close()
        if not chunks:
            raise ValueError("no images to encode")
        return unit_rows(np.vstack(chunks))

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        if not texts:
            raise ValueError("no texts to encode")
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self.processor(text=texts[start:start + batch_size], padding=True,
                                   truncation=True, return_tensors="pt")
            with self.torch.inference_mode():
                result = self.model.get_text_features(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device))
            chunks.append(_as_tensor(result).float().cpu().numpy())
        return unit_rows(np.vstack(chunks))

    def encode_pil_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        if not images:
            raise ValueError("no in-memory images to encode")
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = self.processor(images=images[start:start + batch_size], return_tensors="pt")
            with self.torch.inference_mode():
                result = self.model.get_image_features(
                    pixel_values=batch["pixel_values"].to(self.device))
            chunks.append(_as_tensor(result).float().cpu().numpy())
        return unit_rows(np.vstack(chunks))

    def metadata(self) -> dict:
        return {"model": self.model_id, "revision": self.revision, "device": str(self.device),
                "transformers": importlib.metadata.version("transformers"),
                "torch": importlib.metadata.version("torch")}


def extract_clip(manifest: dict, output: Path, *, model_id: str, revision: str | None,
                 device: str = "cpu", batch_size: int = 32) -> dict:
    from .common import representatives
    rows = representatives(manifest)
    backend = ClipBackend(model_id, revision, device)
    ids = [row["id"] for row in rows]
    matrix = backend.encode_images(rows, batch_size)
    metadata = backend.metadata() | {"manifest_sha256": manifest["manifest_sha256"], "batch_size": batch_size}
    save_matrix(output, matrix, ids, metadata)
    return metadata
