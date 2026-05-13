"""
MedGemma Model Wrapper

Wraps Google's MedGemma (Medical Gemma) vision-language model for:
  - Chest X-ray report generation
  - Clinical question answering

Model: google/medgemma-4b-it or google/medgemma-27b-it
Paper: https://deepmind.google/models/gemma/medgemma/
HF:    https://huggingface.co/google/medgemma-4b-it
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


#  Default Prompts 

REPORT_SYSTEM_PROMPT = """You are an expert radiologist. Analyze the provided chest X-ray image and generate a structured radiology report.

Your report must include:
1. **Findings**: Describe all visible structures and abnormalities systematically (lungs, heart, mediastinum, bones, soft tissues, support devices if present).
2. **Impression**: Provide a concise clinical summary of the key findings.
3. **Recommendations**: Suggest follow-up actions if clinically warranted.

Be precise, evidence-based, and use standard radiological terminology. Only report what is visible in the image."""

QA_SYSTEM_PROMPT = """You are a clinical radiology assistant. Answer the user's question about a chest X-ray based on:
1. The provided chest X-ray image (if given)
2. The retrieved context from the medical knowledge base

Be concise, accurate, and evidence-based. Do not hallucinate findings that are not present."""


class MedGemmaModel:
    """
    MedGemma vision-language model wrapper.

    Supports:
    - 4-bit quantization (bitsandbytes) for reduced VRAM usage
    - Report generation from CXR images
    - Clinical QA with optional RAG context
    - Text-only mode (no image)
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        load_in_4bit: bool = True,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        self.model = None
        self.processor = None
        self._loaded = False

        logger.info(f"MedGemmaModel initialized (model_id={model_id}, 4bit={load_in_4bit})")

    #  Loading 

    def load(self) -> "MedGemmaModel":
        """Load model and processor. Call before inference."""
        if self._loaded:
            return self

        logger.info(f"Loading MedGemma: {self.model_id}")

        # Processor (handles both text & image tokenization)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.hf_token,
        )

        # Quantization config
        bnb_config = None
        if self.load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Using 4-bit quantization (bitsandbytes nf4)")

        # Model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
            token=self.hf_token,
        )
        self.model.eval()

        self._loaded = True
        logger.info("MedGemma loaded successfully")
        return self

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call model.load() first.")

    #  Core Generation 

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text from prompt (and optionally an image).

        Args:
            prompt: User prompt / question
            image: PIL Image, path to image file, or None for text-only
            system_prompt: Override default system prompt
            max_new_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated text string
        """
        self._ensure_loaded()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Build conversation messages (Gemma chat template)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # Tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                do_sample=(self.temperature > 0),
            )

        # Decode only newly generated tokens
        generated = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )
        return generated.strip()

    #  High-Level Tasks 

    def generate_report(
        self,
        image: Union[Image.Image, str, Path],
        additional_context: str = "",
    ) -> dict:
        """
        Generate a structured radiology report from a chest X-ray.

        Args:
            image: CXR image
            additional_context: Optional clinical context (patient history, etc.)

        Returns:
            dict with keys: raw_text, findings, impression, recommendations
        """
        prompt = "Please analyze this chest X-ray and generate a complete radiology report."
        if additional_context:
            prompt += f"\n\nClinical context: {additional_context}"

        raw_text = self.generate(
            prompt=prompt,
            image=image,
            system_prompt=REPORT_SYSTEM_PROMPT,
            max_new_tokens=600,
        )

        return {
            "model": self.model_id,
            "raw_text": raw_text,
            **self._parse_report_sections(raw_text),
        }

    def answer_question(
        self,
        question: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        context: str = "",
    ) -> str:
        """
        Answer a clinical question about a chest X-ray using RAG context.

        Args:
            question: Clinical question
            image: Optional CXR image
            context: Retrieved context from knowledge base

        Returns:
            Answer string
        """
        if context:
            prompt = (
                f"Context from medical knowledge base:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context and image provided:"
            )
        else:
            prompt = f"Question: {question}"

        return self.generate(
            prompt=prompt,
            image=image,
            system_prompt=QA_SYSTEM_PROMPT,
            max_new_tokens=400,
        )

    #  Helpers 

    @staticmethod
    def _parse_report_sections(text: str) -> dict:
        """Parse findings/impression/recommendations from generated report."""
        sections = {"findings": "", "impression": "", "recommendations": ""}
        text_lower = text.lower()

        def extract_section(start_key: str, end_keys: list[str]) -> str:
            start = text_lower.find(start_key)
            if start == -1:
                return ""
            start = text.find(":", start) + 1
            end = len(text)
            for ek in end_keys:
                idx = text_lower.find(ek, start)
                if idx != -1:
                    end = min(end, idx)
            return text[start:end].strip()

        sections["findings"] = extract_section(
            "findings", ["impression", "recommendation", "conclusion"]
        )
        sections["impression"] = extract_section(
            "impression", ["recommendation", "conclusion", "follow"]
        )
        sections["recommendations"] = extract_section(
            "recommendation", ["note:", "disclaimer"]
        )

        # Fallback: return full text if parsing fails
        if not any(sections.values()):
            sections["findings"] = text

        return sections

    def get_model_info(self) -> dict:
        """Return model metadata for UI display."""
        return {
            "name": "MedGemma",
            "model_id": self.model_id,
            "quantization": "4-bit NF4" if self.load_in_4bit else "BF16",
            "task": "Vision-Language (CXR Report Generation & QA)",
            "loaded": self._loaded,
        }
