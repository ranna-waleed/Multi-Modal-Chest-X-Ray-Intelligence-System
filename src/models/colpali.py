"""
ColPali Model Wrapper

Wraps the ColPali (Contextualized Late Interaction over PaliGemma) model,
a vision-language retrieval model that embeds document pages (including
images) into multi-vector representations for efficient retrieval.

Used in this system as the RETRIEVER in the RAG pipeline:
  - Embeds CXR images into query vectors
  - Embeds radiology reports (as document pages) into document vectors
  - Retrieves top-K most relevant reports for a given CXR query

Paper: "ColPali: Efficient Document Retrieval with Vision Language Models"
       https://arxiv.org/abs/2407.01449
HF:    https://huggingface.co/vidore/colpali-v1.2
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ColPaliModel:
    """
    ColPali late-interaction retrieval model.

    In the RAG pipeline:
    - Documents (radiology reports rendered as images or text) are indexed
      offline by calling `embed_documents()`.
    - At query time, a CXR image is embedded with `embed_query()` and the
      top-K documents are retrieved with `retrieve()`.

    ColPali uses the MaxSim operator (like ColBERT) over multi-vector
    patch-level representations.
    """

    def __init__(
        self,
        model_id: str = "vidore/colpali-v1.2",
        device: str = "auto",
        batch_size: int = 4,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        self.model = None
        self.processor = None
        self._loaded = False

        logger.info(f"ColPaliModel initialized (model_id={model_id})")

    #  Loading  

    def load(self) -> "ColPaliModel":
        """Load ColPali model and processor."""
        if self._loaded:
            return self

        logger.info(f"Loading ColPali: {self.model_id}")

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor

            if self.device == "auto":
                _device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                _device = self.device

            self.model = ColPali.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map=_device,
                token=self.hf_token,
            ).eval()

            self.processor = ColPaliProcessor.from_pretrained(
                self.model_id,
                token=self.hf_token,
            )

            self._loaded = True
            self._device = _device
            logger.info(f"ColPali loaded on device: {_device}")

        except ImportError:
            logger.error(
                "colpali_engine not installed. Run: pip install colpali-engine"
            )
            raise

        return self

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call model.load() first.")

    #  Embedding 

    def embed_images(
        self,
        images: list[Union[Image.Image, str, Path]],
        is_query: bool = True,
    ) -> torch.Tensor:
        """
        Embed a list of images into ColPali multi-vector representations.

        Args:
            images: List of PIL Images or paths
            is_query: True for query embeddings, False for document embeddings

        Returns:
            Tensor of shape (N, seq_len, dim)
        """
        self._ensure_loaded()

        # Load images if paths
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        all_embeddings = []

        for i in range(0, len(pil_images), self.batch_size):
            batch = pil_images[i : i + self.batch_size]

            if is_query:
                # Query: image only
                batch_inputs = self.processor.process_images(batch)
            else:
                # Document: image only (ColPali treats doc pages as images)
                batch_inputs = self.processor.process_images(batch)

            batch_inputs = {k: v.to(self._device) for k, v in batch_inputs.items()}

            with torch.inference_mode():
                embeddings = self.model(**batch_inputs)  # (B, seq_len, dim)

            all_embeddings.append(embeddings.cpu().float())

        return torch.cat(all_embeddings, dim=0)

    def embed_text_queries(self, queries: list[str]) -> torch.Tensor:
        """
        Embed text queries (for text-based QA retrieval).

        Args:
            queries: List of query strings

        Returns:
            Tensor of shape (N, seq_len, dim)
        """
        self._ensure_loaded()

        all_embeddings = []
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i : i + self.batch_size]
            batch_inputs = self.processor.process_queries(batch)
            batch_inputs = {k: v.to(self._device) for k, v in batch_inputs.items()}

            with torch.inference_mode():
                embeddings = self.model(**batch_inputs)

            all_embeddings.append(embeddings.cpu().float())

        return torch.cat(all_embeddings, dim=0)

    #  Scoring (MaxSim) 

    @staticmethod
    def maxsim_score(
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MaxSim similarity between query and document embeddings.

        MaxSim: For each query token, find the max cosine similarity with
        any document token, then sum across query tokens.

        Args:
            query_embs: (Q, Lq, dim)
            doc_embs:   (D, Ld, dim)

        Returns:
            scores: (Q, D) similarity matrix
        """
        # Normalize
        q = torch.nn.functional.normalize(query_embs, dim=-1)  # (Q, Lq, dim)
        d = torch.nn.functional.normalize(doc_embs, dim=-1)    # (D, Ld, dim)

        # (Q, Lq, dim) x (D, dim, Ld) -> (Q, D, Lq, Ld)
        # Use einsum for efficiency
        scores = torch.einsum("qid,djd->qij", q, d.transpose(1, 2))  # not quite right
        # Correct: (Q, Lq, dim) @ (D, dim, Ld) -> need broadcasting
        # (Q, 1, Lq, dim) * (1, D, Ld, dim)
        q = q.unsqueeze(1)   # (Q, 1, Lq, dim)
        d = d.unsqueeze(0)   # (1, D, Ld, dim)

        # Similarity: (Q, D, Lq, Ld)
        sim = torch.matmul(q, d.transpose(-1, -2))  # (Q, D, Lq, Ld)

        # MaxSim: max over doc tokens for each query token, then sum
        max_sim = sim.max(dim=-1).values  # (Q, D, Lq)
        scores = max_sim.sum(dim=-1)       # (Q, D)

        return scores

    #  Retrieval 

    def retrieve(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        doc_metadata: list[dict],
        top_k: int = 5,
    ) -> list[list[dict]]:
        """
        Retrieve top-K documents for each query.

        Args:
            query_embs: Query embeddings (Q, Lq, dim)
            doc_embs:   Document embeddings (D, Ld, dim)
            doc_metadata: List of D metadata dicts (e.g., {"text": ..., "id": ...})
            top_k: Number of documents to retrieve

        Returns:
            List of Q lists, each containing top_k result dicts
        """
        scores = self.maxsim_score(query_embs, doc_embs)  # (Q, D)

        results = []
        for q_idx in range(scores.shape[0]):
            q_scores = scores[q_idx]  # (D,)
            topk_indices = q_scores.topk(min(top_k, len(doc_metadata))).indices.tolist()
            topk_scores = q_scores.topk(min(top_k, len(doc_metadata))).values.tolist()

            q_results = []
            for rank, (doc_idx, score) in enumerate(zip(topk_indices, topk_scores)):
                result = {
                    "rank": rank + 1,
                    "score": float(score),
                    "doc_idx": doc_idx,
                    **doc_metadata[doc_idx],
                }
                q_results.append(result)

            results.append(q_results)

        return results

    def get_model_info(self) -> dict:
        return {
            "name": "ColPali",
            "model_id": self.model_id,
            "task": "Multi-Vector Document Retrieval (RAG)",
            "similarity": "MaxSim (Late Interaction)",
            "loaded": self._loaded,
        }
