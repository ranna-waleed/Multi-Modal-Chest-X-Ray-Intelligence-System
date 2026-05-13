"""
CLIP Model Wrapper

Wraps OpenAI CLIP (and BiomedCLIP) for:
  1. Image-text similarity retrieval (alternative to ColPali in RAG)
  2. Report generation baseline via zero-shot image captioning prompts
  3. Embedding chest X-rays for comparison with MedGemma

Models supported:
  - openai/clip-vit-large-patch14          (general CLIP)
  - microsoft/BiomedCLIP-PubMedBERT_256    (biomedical fine-tuned)
  - openai/clip-vit-base-patch16

Paper: "Learning Transferable Visual Models From Natural Language" (Radford 2021)
HF:    https://huggingface.co/openai/clip-vit-large-patch14
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class CLIPModel:
    """
    CLIP vision-language model wrapper.

    Capabilities:
    - Embed images → fixed-size float vectors (for cosine similarity retrieval)
    - Embed text → fixed-size float vectors
    - Zero-shot image-text similarity scoring
    - Report-style text ranking against a CXR image

    In the RAG pipeline, CLIP serves as an alternative retriever to ColPali:
    - Single-vector per image/text (simpler than ColPali's multi-vector MaxSim)
    - Fast cosine similarity search via FAISS
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: str = "auto",
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None
        self._loaded = False

        logger.info(f"CLIPModel initialized (model_id={model_id})")

    #  Loading  

    def load(self) -> "CLIPModel":
        """Load CLIP model and processor."""
        if self._loaded:
            return self

        logger.info(f"Loading CLIP: {self.model_id}")

        self.processor = CLIPProcessor.from_pretrained(
            self.model_id,
            token=self.hf_token,
        )
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            token=self.hf_token,
        ).to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"CLIP loaded on device: {self.device}")
        return self

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call model.load() first.")

    #  Image Embedding 

    def embed_images(
        self,
        images: list[Union[Image.Image, str, Path]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed images into CLIP visual feature vectors.

        Args:
            images: List of PIL Images or paths
            normalize: L2-normalize embeddings (default True for cosine sim)

        Returns:
            np.ndarray of shape (N, embed_dim)
        """
        self._ensure_loaded()

        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, (str, Path)) else img.convert("RGB")
            for img in images
        ]

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)

        embeddings = image_features.cpu().float().numpy()

        if normalize:
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    #  Text Embedding 

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed texts into CLIP text feature vectors.

        Args:
            texts: List of text strings
            normalize: L2-normalize embeddings

        Returns:
            np.ndarray of shape (N, embed_dim)
        """
        self._ensure_loaded()

        # CLIP text has max token limit (~77 tokens); truncate long texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)

        embeddings = text_features.cpu().float().numpy()

        if normalize:
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    #  Zero-Shot Classification / Similarity 

    def image_text_similarity(
        self,
        image: Union[Image.Image, str, Path],
        candidate_texts: list[str],
    ) -> list[dict]:
        """
        Compute cosine similarity between an image and candidate texts.

        Useful for zero-shot report assessment:
        - Give a CXR image and a list of finding descriptions
        - Returns which descriptions are most similar to the image

        Args:
            image: CXR image
            candidate_texts: List of text descriptions to rank

        Returns:
            List of dicts sorted by similarity score (descending)
        """
        self._ensure_loaded()

        img_emb = self.embed_images([image])               # (1, D)
        txt_emb = self.embed_texts(candidate_texts)        # (N, D)

        # Cosine similarity (embeddings already normalized)
        scores = (img_emb @ txt_emb.T)[0]                 # (N,)

        results = sorted(
            [
                {"text": t, "score": float(s)}
                for t, s in zip(candidate_texts, scores)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )
        return results

    #  Report Generation (CLIP-guided) 

    def generate_report_clip(
        self,
        image: Union[Image.Image, str, Path],
        finding_templates: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate a pseudo-report by ranking radiology finding templates
        against the input CXR image using CLIP similarity.

        This is CLIP's "report generation" — not autoregressive but a
        retrieval-based summary. Useful for comparison against MedGemma.

        Args:
            image: CXR image
            finding_templates: Optional custom finding descriptions to rank

        Returns:
            dict with top findings and similarity scores
        """
        if finding_templates is None:
            finding_templates = DEFAULT_FINDING_TEMPLATES

        results = self.image_text_similarity(image, finding_templates)
        top_findings = [r for r in results if r["score"] > 0.20][:5]

        # Build simple pseudo-report
        if top_findings:
            findings_text = "\n".join(
                f"- {r['text']} (similarity: {r['score']:.3f})"
                for r in top_findings
            )
            impression = top_findings[0]["text"]
        else:
            findings_text = "No significant findings detected above similarity threshold."
            impression = "No significant findings."

        return {
            "model": f"CLIP ({self.model_id})",
            "raw_text": findings_text,
            "findings": findings_text,
            "impression": impression,
            "recommendations": "Clinical correlation recommended.",
            "similarity_scores": results[:10],
        }

    def get_model_info(self) -> dict:
        return {
            "name": "CLIP",
            "model_id": self.model_id,
            "task": "Vision-Language Similarity (Retrieval & Zero-shot Classification)",
            "embedding_dim": 768 if "base" in self.model_id else 1024,
            "loaded": self._loaded,
        }


#  Default Radiology Finding Templates for CLIP 

DEFAULT_FINDING_TEMPLATES = [
    # Normal
    "No acute cardiopulmonary findings.",
    "Normal chest X-ray with clear lung fields.",
    # Lungs
    "Bilateral lower lobe atelectasis is present.",
    "Right lower lobe consolidation consistent with pneumonia.",
    "Left lower lobe pneumonia with air space opacity.",
    "Bilateral pulmonary infiltrates suggesting pneumonia.",
    "Lung opacity consistent with consolidation or atelectasis.",
    "Small right pleural effusion is present.",
    "Bilateral pleural effusions are present.",
    "Pulmonary edema with bilateral alveolar opacities.",
    "Interstitial edema with Kerley B lines.",
    "Pneumothorax on the left side.",
    "Right-sided pneumothorax.",
    "Lung nodule or mass in the right upper lobe.",
    "Hyperinflation consistent with emphysema or COPD.",
    # Heart
    "Cardiomegaly with enlarged cardiac silhouette.",
    "Normal cardiac size and contour.",
    "Enlarged cardiac silhouette suggesting cardiomegaly.",
    # Mediastinum
    "Mediastinal widening.",
    "Normal mediastinal contours.",
    "Enlarged mediastinum.",
    # Support Devices
    "Endotracheal tube in appropriate position.",
    "Central venous catheter with tip in superior vena cava.",
    "Nasogastric tube coursing to the stomach.",
    "Pacemaker leads in right ventricle.",
    # Bones
    "Rib fractures present.",
    "No acute osseous abnormality.",
    # Diaphragm
    "Normal diaphragm position and contour.",
    "Elevated right hemidiaphragm.",
]
