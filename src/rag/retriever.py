"""
RAG Retriever

Orchestrates the retrieval pipeline for the QA mode:

  1. Load knowledge base (radiology reports)
  2. At query time: embed query image/text
  3. Retrieve top-K relevant reports
  4. Format context for the generator (MedGemma)

Supports two retrieval backends:
  - CLIP  → single-vector FAISS (fast, approximate)
  - ColPali → multi-vector MaxSim (richer, exact, slower at scale)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever.

    Usage:
        retriever = RAGRetriever(backend="colpali")
        retriever.load_knowledge_base("data/sample_reports")
        results = retriever.retrieve(image=img, query="Is there pneumonia?", top_k=3)
        context = retriever.format_context(results)
    """

    def __init__(
        self,
        backend: str = "colpali",   # "colpali" | "clip"
        clip_model=None,
        colpali_model=None,
        vector_store=None,
        top_k: int = 5,
    ):
        assert backend in ("colpali", "clip"), f"Unknown backend: {backend}"
        self.backend = backend
        self.clip_model = clip_model
        self.colpali_model = colpali_model
        self.vector_store = vector_store
        self.top_k = top_k

        self._knowledge_base: list[dict] = []
        self._indexed = False

    #  Knowledge Base Loading 

    def load_knowledge_base(
        self,
        reports_path: str,
        rebuild_index: bool = False,
        index_cache_path: Optional[str] = None,
    ) -> "RAGRetriever":
        """
        Load radiology reports from disk and build (or load cached) index.

        Expects either:
          - A directory of .txt files (one report per file)
          - A single JSON file with list of {"id": ..., "text": ...} dicts

        Args:
            reports_path: Path to reports directory or JSON file
            rebuild_index: Force rebuilding the index even if cache exists
            index_cache_path: Path to cache the built index

        Returns:
            self
        """
        reports_path = Path(reports_path)

        if not reports_path.exists():
            logger.warning(f"Reports path does not exist: {reports_path}. Using demo data.")
            self._knowledge_base = self._get_demo_reports()
        elif reports_path.is_file() and reports_path.suffix == ".json":
            with open(reports_path) as f:
                self._knowledge_base = json.load(f)
        elif reports_path.is_dir():
            self._knowledge_base = []
            for txt_file in sorted(reports_path.glob("*.txt")):
                self._knowledge_base.append({
                    "id": txt_file.stem,
                    "text": txt_file.read_text(encoding="utf-8").strip(),
                    "source": str(txt_file),
                })
            if not self._knowledge_base:
                logger.warning("No .txt files found in reports_path. Using demo data.")
                self._knowledge_base = self._get_demo_reports()
        else:
            self._knowledge_base = self._get_demo_reports()

        logger.info(f"Loaded {len(self._knowledge_base)} reports into knowledge base")

        # Build or load index
        self._build_index(rebuild=rebuild_index, cache_path=index_cache_path)
        return self

    def _build_index(self, rebuild: bool = False, cache_path: Optional[str] = None) -> None:
        """Build retrieval index from loaded knowledge base."""
        if self.backend == "clip":
            self._build_clip_index(rebuild=rebuild, cache_path=cache_path)
        elif self.backend == "colpali":
            self._build_colpali_index(rebuild=rebuild, cache_path=cache_path)

    def _build_clip_index(self, rebuild: bool = False, cache_path: Optional[str] = None) -> None:
        """Build FAISS index using CLIP text embeddings."""
        import numpy as np
        from .vector_store import FAISSVectorStore

        if cache_path and Path(cache_path + ".faiss").exists() and not rebuild:
            self.vector_store = FAISSVectorStore().load(
                cache_path + ".faiss",
                cache_path + "_meta.json",
            )
            self._indexed = True
            return

        if self.clip_model is None:
            logger.warning("No CLIP model provided; using dummy embeddings for demo mode.")
            import numpy as np
            dim = 1024
            embs = np.random.randn(len(self._knowledge_base), dim).astype(np.float32)
            norms = np.linalg.norm(embs, axis=-1, keepdims=True)
            embs /= norms
            self.vector_store = FAISSVectorStore(embedding_dim=dim)
            self.vector_store.build(embs, self._knowledge_base)
            self._indexed = True
            return

        texts = [doc["text"][:500] for doc in self._knowledge_base]  # truncate
        embeddings = self.clip_model.embed_texts(texts)              # (N, dim)

        self.vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.build(embeddings, self._knowledge_base)

        if cache_path:
            self.vector_store.save(cache_path + ".faiss", cache_path + "_meta.json")

        self._indexed = True
        logger.info(f"Built CLIP FAISS index for {len(self._knowledge_base)} reports")

    def _build_colpali_index(self, rebuild: bool = False, cache_path: Optional[str] = None) -> None:
        """Build ColPali multi-vector index from text-rendered report images."""
        from .vector_store import ColPaliVectorStore

        if cache_path and Path(cache_path + "_colpali.pkl").exists() and not rebuild:
            self.vector_store = ColPaliVectorStore().load(cache_path + "_colpali.pkl")
            self._indexed = True
            return

        if self.colpali_model is None:
            logger.warning("No ColPali model provided; using CLIP-style fallback for demo.")
            self._build_clip_index(rebuild=rebuild, cache_path=cache_path)
            return

        # Render reports as images for ColPali
        report_images = [
            self._text_to_image(doc["text"])
            for doc in self._knowledge_base
        ]

        doc_embeddings = []
        for i in range(0, len(report_images), self.colpali_model.batch_size):
            batch = report_images[i : i + self.colpali_model.batch_size]
            embs = self.colpali_model.embed_images(batch, is_query=False)
            doc_embeddings.extend([embs[j] for j in range(embs.shape[0])])

        self.vector_store = ColPaliVectorStore()
        self.vector_store.add(doc_embeddings, self._knowledge_base)

        if cache_path:
            self.vector_store.save(cache_path + "_colpali.pkl")

        self._indexed = True
        logger.info(f"Built ColPali index for {len(self._knowledge_base)} reports")

    #  Retrieval 

    def retrieve(
        self,
        image: Optional[Union[Image.Image, str, Path]] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-K relevant reports for a CXR image + query.

        Args:
            image: CXR image (used as primary query for ColPali)
            query: Text question (used as fallback / combined query)
            top_k: Override default top_k

        Returns:
            List of top-K result dicts with "text", "score", "rank"
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call load_knowledge_base() first.")

        k = top_k or self.top_k

        if self.backend == "clip":
            return self._retrieve_clip(image=image, query=query, top_k=k)
        elif self.backend == "colpali":
            return self._retrieve_colpali(image=image, query=query, top_k=k)

    def _retrieve_clip(
        self,
        image=None,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """CLIP-based retrieval using image and/or text embeddings."""
        import numpy as np

        if self.clip_model is None:
            # Demo fallback: return random docs
            import random
            docs = random.sample(self._knowledge_base, min(top_k, len(self._knowledge_base)))
            return [{"rank": i + 1, "score": 0.5, **d} for i, d in enumerate(docs)]

        if image is not None:
            img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
            query_emb = self.clip_model.embed_images([img])  # (1, dim)
        elif query is not None:
            query_emb = self.clip_model.embed_texts([query])  # (1, dim)
        else:
            raise ValueError("Must provide image or query")

        results = self.vector_store.search(query_emb, top_k=top_k)
        return results[0]  # Single query → first result list

    def _retrieve_colpali(
        self,
        image=None,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """ColPali MaxSim retrieval using image query embeddings."""
        if self.colpali_model is None:
            return self._retrieve_clip(image=image, query=query, top_k=top_k)

        import torch
        if image is not None:
            img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
            query_embs = self.colpali_model.embed_images([img], is_query=True)
        elif query is not None:
            query_embs = self.colpali_model.embed_text_queries([query])
        else:
            raise ValueError("Must provide image or query")

        results = self.vector_store.search_with_query_embs(query_embs, top_k=top_k)
        return results[0]

    #  Context Formatting 

    def format_context(
        self,
        results: list[dict],
        max_chars: int = 2000,
    ) -> str:
        """
        Format retrieved results into a context string for MedGemma.

        Args:
            results: List of result dicts from retrieve()
            max_chars: Maximum total characters in context

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0

        for r in results:
            text = r.get("text", "")
            header = f"[Report {r['rank']} | Similarity: {r.get('score', 0):.3f}]"
            entry = f"{header}\n{text}"

            if total_chars + len(entry) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(entry[:remaining] + "...")
                break

            context_parts.append(entry)
            total_chars += len(entry) + 2

        return "\n\n".join(context_parts)

    #  Helpers 

    @staticmethod
    def _text_to_image(text: str, width: int = 512, height: int = 512) -> Image.Image:
        """Render a text report as a simple image (for ColPali document indexing)."""
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Word-wrap text
            words = text.split()
            lines, line = [], []
            for word in words:
                line.append(word)
                if len(" ".join(line)) > 60:
                    lines.append(" ".join(line[:-1]))
                    line = [word]
            if line:
                lines.append(" ".join(line))
            y = 10
            for line in lines[:30]:
                draw.text((10, y), line, fill=(0, 0, 0))
                y += 16
                if y > height - 20:
                    break
            return img
        except Exception:
            return Image.new("RGB", (width, height), color=(200, 200, 200))

    @staticmethod
    def _get_demo_reports() -> list[dict]:
        """Demo radiology reports for testing without MIMIC-CXR."""
        return [
            {
                "id": "demo_001",
                "text": (
                    "Findings: The lungs are clear bilaterally. No focal consolidation, "
                    "effusion, or pneumothorax is identified. The cardiomediastinal silhouette "
                    "is within normal limits. The bony thorax is intact. "
                    "Impression: No acute cardiopulmonary process."
                ),
            },
            {
                "id": "demo_002",
                "text": (
                    "Findings: There is a right lower lobe opacity consistent with pneumonia. "
                    "Mild cardiomegaly is present. No pleural effusion or pneumothorax. "
                    "The mediastinum is unremarkable. "
                    "Impression: Right lower lobe pneumonia. Mild cardiomegaly."
                ),
            },
            {
                "id": "demo_003",
                "text": (
                    "Findings: Bilateral pleural effusions, right greater than left. "
                    "Enlarged cardiac silhouette. Pulmonary vascular engorgement. "
                    "Perihilar haziness consistent with pulmonary edema. "
                    "Impression: Congestive heart failure with pulmonary edema and bilateral pleural effusions."
                ),
            },
            {
                "id": "demo_004",
                "text": (
                    "Findings: Linear opacity at the right lung base consistent with atelectasis. "
                    "No focal consolidation or effusion. Heart size normal. Mediastinum midline. "
                    "Impression: Bibasilar atelectasis, right greater than left."
                ),
            },
            {
                "id": "demo_005",
                "text": (
                    "Findings: Endotracheal tube tip approximately 4 cm above the carina. "
                    "Nasogastric tube courses to the stomach. Right internal jugular central "
                    "venous catheter tip at the cavoatrial junction. Bilateral patchy opacities "
                    "consistent with aspiration or pneumonia. "
                    "Impression: Support devices in appropriate position. Bilateral infiltrates."
                ),
            },
            {
                "id": "demo_006",
                "text": (
                    "Findings: Hyperinflated lungs with flattened hemidiaphragms. Increased AP "
                    "diameter. No consolidation or effusion. Bullous changes in the upper lobes. "
                    "Cardiac size normal. "
                    "Impression: Findings consistent with emphysema/COPD."
                ),
            },
            {
                "id": "demo_007",
                "text": (
                    "Findings: Left-sided pneumothorax measuring approximately 20%. Trachea "
                    "midline. Lungs otherwise clear. No pleural effusion. "
                    "Impression: Left pneumothorax, moderate."
                ),
            },
            {
                "id": "demo_008",
                "text": (
                    "Findings: Cardiomegaly with cardiothoracic ratio of 0.6. Mild pulmonary "
                    "vascular prominence. Small bilateral pleural effusions. No consolidation. "
                    "Impression: Cardiomegaly with mild pulmonary venous hypertension."
                ),
            },
        ]
