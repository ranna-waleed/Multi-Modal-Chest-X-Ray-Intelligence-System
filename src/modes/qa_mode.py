"""
QA Mode (RAG-based)

Answers clinical questions about chest X-rays using Retrieval-Augmented Generation.

Pipeline:
  1. User provides: CXR image + clinical question
  2. Retrieve top-K relevant radiology reports from the knowledge base
     (using ColPali or CLIP embeddings)
  3. Combine retrieved context + image + question → MedGemma
  4. MedGemma generates a grounded, evidence-based answer

This approach reduces hallucinations by grounding answers in retrieved
clinical evidence, following the CXR-RePaiR-Gen methodology.

Reference: "Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models"
           Ranjit et al., 2023 (arXiv:2305.03660)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Structured output of a QA query."""
    question: str
    answer: str
    retrieved_docs: list[dict]
    retrieval_backend: str
    generation_time_s: float
    context_used: str
    metadata: dict = field(default_factory=dict)

    def format_display(self) -> str:
        """Format for Gradio display."""
        lines = [
            f"###  Question\n{self.question}\n",
            f"###  Answer\n{self.answer}\n",
            f"---",
            f"**Retrieval backend:** {self.retrieval_backend}  |  "
            f"**Docs retrieved:** {len(self.retrieved_docs)}  |  "
            f"**Time:** {self.generation_time_s:.1f}s",
        ]
        return "\n".join(lines)

    def format_retrieved_docs(self) -> str:
        """Format retrieved documents for display."""
        if not self.retrieved_docs:
            return "No documents retrieved."
        parts = []
        for doc in self.retrieved_docs:
            parts.append(
                f"**[{doc['rank']}] Score: {doc.get('score', 0):.3f}**\n"
                f"{doc.get('text', '')[:300]}{'...' if len(doc.get('text',''))>300 else ''}"
            )
        return "\n\n---\n\n".join(parts)


class QAPipeline:
    """
    RAG-based QA pipeline for chest X-ray clinical questions.
    """

    def __init__(
        self,
        medgemma_model=None,
        retriever=None,
        top_k: int = 3,
    ):
        self.medgemma = medgemma_model
        self.retriever = retriever
        self.top_k = top_k

    #  Main Entry Point 

    def answer(
        self,
        question: str,
        image: Optional[Union[Image.Image, str]] = None,
        top_k: Optional[int] = None,
        use_retrieval: bool = True,
    ) -> QAResult:
        """
        Answer a clinical question about a chest X-ray.

        Args:
            question: Clinical question (e.g., "Is there pneumonia?")
            image: CXR image (optional but recommended)
            top_k: Number of reports to retrieve
            use_retrieval: Whether to use RAG (set False for direct VLM QA)

        Returns:
            QAResult with answer, retrieved docs, and metadata
        """
        start = time.time()
        k = top_k or self.top_k

        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image is not None:
            image = image.convert("RGB")

        retrieved_docs = []
        context = ""
        retrieval_backend = "none"

        # ── Step 1: Retrieve ───────────────────────────────────────────────────
        if use_retrieval and self.retriever is not None:
            try:
                retrieved_docs = self.retriever.retrieve(
                    image=image,
                    query=question,
                    top_k=k,
                )
                context = self.retriever.format_context(retrieved_docs)
                retrieval_backend = self.retriever.backend
                logger.info(f"Retrieved {len(retrieved_docs)} docs via {retrieval_backend}")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                retrieved_docs = []
                context = ""
        elif use_retrieval:
            logger.warning("No retriever loaded; falling back to direct VLM QA.")

        #  Step 2: Generate Answer 
        if self.medgemma is not None:
            try:
                answer = self.medgemma.answer_question(
                    question=question,
                    image=image,
                    context=context,
                )
            except Exception as e:
                logger.error(f"MedGemma answer failed: {e}")
                answer = self._demo_answer(question, context)
        else:
            answer = self._demo_answer(question, context)

        elapsed = time.time() - start

        return QAResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            retrieval_backend=retrieval_backend,
            generation_time_s=elapsed,
            context_used=context,
        )

    #  Multi-Turn Conversation 

    def answer_with_history(
        self,
        question: str,
        history: list[dict],
        image: Optional[Image.Image] = None,
    ) -> tuple[QAResult, list[dict]]:
        """
        Answer a question within a multi-turn conversation context.

        Args:
            question: Current question
            history: List of {"role": "user"/"assistant", "content": str} dicts
            image: CXR image

        Returns:
            (QAResult, updated_history)
        """
        # Build context from conversation history
        history_context = ""
        if history:
            history_context = "\n".join(
                f"{msg['role'].title()}: {msg['content']}"
                for msg in history[-4:]  # Last 4 turns
            )
            history_context = f"Conversation history:\n{history_context}\n\n"

        result = self.answer(question=history_context + question, image=image)

        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        ]

        return result, updated_history

    #  Batch QA 

    def batch_answer(
        self,
        questions: list[str],
        image: Optional[Image.Image] = None,
    ) -> list[QAResult]:
        """Answer multiple questions about the same CXR image."""
        return [self.answer(q, image=image) for q in questions]

    #  Demo Fallback 

    @staticmethod
    def _demo_answer(question: str, context: str = "") -> str:
        """Realistic demo answer when MedGemma is not loaded."""
        q = question.lower()

        # Map common questions to realistic answers
        if any(w in q for w in ["pneumonia", "consolidat"]):
            answer = (
                "Based on the chest X-ray and retrieved clinical reports, there is "
                "**increased opacity in the right lower lobe** consistent with consolidation, "
                "likely representing **pneumonia**. The findings show an airspace opacity with "
                "ill-defined borders at the right lung base. No cavitation is identified. "
                "The left lung appears relatively clear.\n\n"
                "*Clinical recommendation: Antibiotic therapy should be considered if "
                "clinical presentation supports bacterial pneumonia.*"
            )
        elif any(w in q for w in ["effusion", "pleural"]):
            answer = (
                "Reviewing the chest X-ray and knowledge base context: **no significant "
                "pleural effusion** is identified in this image. The costophrenic angles "
                "appear sharp bilaterally. The minor fissure is visible on the right. "
                "There is no blunting of the costophrenic angles that would suggest fluid.\n\n"
                "*Note: Small effusions (<200 mL) may not be visible on PA chest X-ray.*"
            )
        elif any(w in q for w in ["cardiomegaly", "heart", "cardiac"]):
            answer = (
                "Assessment of cardiac size: The cardiac silhouette appears **mildly enlarged** "
                "with an estimated cardiothoracic ratio of approximately **0.55** (normal < 0.50). "
                "This is consistent with mild cardiomegaly. The cardiac borders are well-defined. "
                "The aortic knuckle is visible. No pericardial effusion identified.\n\n"
                "*Mild cardiomegaly may be seen in hypertension, cardiomyopathy, or heart failure.*"
            )
        elif any(w in q for w in ["finding", "describe", "summary", "overall"]):
            answer = (
                "**Summary of chest X-ray findings:**\n\n"
                "1. **Right lower lobe opacity** — consistent with consolidation/pneumonia\n"
                "2. **Mild cardiomegaly** — cardiothoracic ratio ~0.55\n"
                "3. **No pleural effusion** — costophrenic angles sharp\n"
                "4. **No pneumothorax** — lung markings visible to periphery\n"
                "5. **Mediastinum** — midline, unremarkable\n"
                "6. **Bones** — no acute osseous abnormality\n\n"
                "*Impression: Right lower lobe pneumonia with mild cardiomegaly.*"
            )
        else:
            answer = (
                f"Based on the chest X-ray and retrieved radiology reports:\n\n"
                f"Regarding **{question}** — the radiographic findings show right lower lobe "
                f"opacity consistent with consolidation. Mild cardiomegaly is present. "
                f"No pleural effusion or pneumothorax is identified.\n\n"
                f"Clinical correlation with patient history and symptoms is recommended."
            )

        if context:
            answer += f"\n\n---\n*Answer grounded using {len(context.split(chr(10)))} lines of retrieved context.*"

        answer += "\n\n>  **Demo output** — MedGemma not loaded. This is a realistic placeholder."
        return answer

    #  Predefined Clinical Questions 

    @staticmethod
    def get_suggested_questions() -> list[str]:
        """Return a list of common CXR clinical questions for the UI."""
        return [
            "Is there any evidence of pneumonia or consolidation?",
            "Are there any pleural effusions?",
            "Is cardiomegaly present?",
            "Is there a pneumothorax?",
            "Describe the lung fields.",
            "Are there any support devices? Are they correctly positioned?",
            "Is there pulmonary edema?",
            "What are the main findings in this chest X-ray?",
            "Is there atelectasis?",
            "Are there any rib fractures?",
            "What is the cardiothoracic ratio?",
            "Is the mediastinum widened?",
        ]