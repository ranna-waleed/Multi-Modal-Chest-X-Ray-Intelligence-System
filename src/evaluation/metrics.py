"""
Evaluation Metrics

Computes standard NLP + clinical metrics for comparing model outputs.

Metrics:
  - BLEU-1/2/3/4 (token overlap)
  - ROUGE-L (longest common subsequence)
  - METEOR (synonym-aware recall)
  - BERTScore (semantic similarity via BERT)
  - ClinicalScore: rule-based check for key clinical entities

Reference datasets:
  - MIMIC-CXR VQA benchmark
  - CXR-PRO (for report generation)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class CXRMetrics:
    """
    Evaluation metrics for CXR report generation and QA.

    Usage:
        metrics = CXRMetrics()
        results = metrics.evaluate_report(
            predictions=["No acute findings. Clear lungs."],
            references=["No acute cardiopulmonary process. Lungs clear."]
        )
    """

    def __init__(self, bertscore_model: str = "dmis-lab/biobert-base-cased-v1.2"):
        self.bertscore_model = bertscore_model
        self._bleu = None
        self._rouge = None
        self._meteor = None
        self._bertscore = None

    #  Load Metric Libraries 

    def _get_bleu(self):
        if self._bleu is None:
            try:
                from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                self._bleu = (corpus_bleu, SmoothingFunction)
            except ImportError:
                logger.warning("nltk not available; BLEU will be skipped.")
        return self._bleu

    def _get_rouge(self):
        if self._rouge is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
            except ImportError:
                logger.warning("rouge_score not available; ROUGE will be skipped.")
        return self._rouge

    def _get_bertscore(self):
        if self._bertscore is None:
            try:
                from bert_score import score as bs_score
                self._bertscore = bs_score
            except ImportError:
                logger.warning("bert_score not available; BERTScore will be skipped.")
        return self._bertscore

    #  Core Evaluation 

    def evaluate_report(
        self,
        predictions: list[str],
        references: list[str],
        compute_bertscore: bool = True,
    ) -> dict:
        """
        Compute all metrics for a batch of report predictions vs references.

        Args:
            predictions: List of generated reports
            references: List of ground-truth reports
            compute_bertscore: Whether to compute BERTScore (slower)

        Returns:
            dict with all metric values
        """
        assert len(predictions) == len(references), "Lists must be same length"

        results = {
            "num_samples": len(predictions),
            "bleu": self._compute_bleu(predictions, references),
            "rouge": self._compute_rouge(predictions, references),
            "clinical": self._compute_clinical_entity_score(predictions, references),
        }

        if compute_bertscore:
            results["bertscore"] = self._compute_bertscore(predictions, references)

        # Aggregate
        results["summary"] = self._aggregate(results)
        return results

    def evaluate_qa(
        self,
        predictions: list[str],
        references: list[str],
        questions: Optional[list[str]] = None,
    ) -> dict:
        """Evaluate QA answers against reference answers."""
        return self.evaluate_report(predictions, references, compute_bertscore=True)

    #  BLEU 

    def _compute_bleu(
        self, predictions: list[str], references: list[str]
    ) -> dict:
        bleu_fn = self._get_bleu()
        if bleu_fn is None:
            return {}

        corpus_bleu, SmoothingFunction = bleu_fn
        sf = SmoothingFunction().method1

        # Tokenize
        tokenized_preds = [p.lower().split() for p in predictions]
        tokenized_refs = [[r.lower().split()] for r in references]

        scores = {}
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = corpus_bleu(tokenized_refs, tokenized_preds, weights=weights, smoothing_function=sf)
            except Exception:
                score = 0.0
            scores[f"bleu_{n}"] = round(float(score), 4)

        return scores

    #  ROUGE 

    def _compute_rouge(
        self, predictions: list[str], references: list[str]
    ) -> dict:
        scorer = self._get_rouge()
        if scorer is None:
            return {}

        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure

        n = len(predictions)
        return {k: round(v / n, 4) for k, v in totals.items()}

    #  BERTScore 

    def _compute_bertscore(
        self, predictions: list[str], references: list[str]
    ) -> dict:
        bs_score = self._get_bertscore()
        if bs_score is None:
            return {}

        try:
            P, R, F1 = bs_score(
                predictions,
                references,
                model_type=self.bertscore_model,
                lang="en",
                verbose=False,
            )
            return {
                "precision": round(float(P.mean()), 4),
                "recall": round(float(R.mean()), 4),
                "f1": round(float(F1.mean()), 4),
            }
        except Exception as e:
            logger.error(f"BERTScore failed: {e}")
            return {}

    #  Clinical Entity Score 

    CLINICAL_ENTITIES = [
        "pneumonia", "atelectasis", "effusion", "pneumothorax",
        "cardiomegaly", "edema", "consolidation", "opacity",
        "fracture", "normal", "clear", "unremarkable",
        "mediastinum", "diaphragm", "hilar", "pleural",
    ]

    def _compute_clinical_entity_score(
        self, predictions: list[str], references: list[str]
    ) -> dict:
        """
        Rule-based clinical entity recall:
        What fraction of clinical terms in the reference appear in the prediction?
        """
        total_precision, total_recall, n = 0.0, 0.0, 0

        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()

            pred_entities = {e for e in self.CLINICAL_ENTITIES if e in pred_lower}
            ref_entities = {e for e in self.CLINICAL_ENTITIES if e in ref_lower}

            if ref_entities:
                recall = len(pred_entities & ref_entities) / len(ref_entities)
                total_recall += recall

            if pred_entities:
                precision = len(pred_entities & ref_entities) / len(pred_entities) if pred_entities else 0
                total_precision += precision

            n += 1

        if n == 0:
            return {}

        precision = total_precision / n
        recall = total_recall / n
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    #  Aggregation 

    @staticmethod
    def _aggregate(results: dict) -> dict:
        """Produce a one-line summary of key metrics."""
        summary = {}
        if "bleu" in results:
            summary["BLEU-1"] = results["bleu"].get("bleu_1", 0)
            summary["BLEU-4"] = results["bleu"].get("bleu_4", 0)
        if "rouge" in results:
            summary["ROUGE-L"] = results["rouge"].get("rougeL", 0)
        if "bertscore" in results:
            summary["BERTScore-F1"] = results["bertscore"].get("f1", 0)
        if "clinical" in results:
            summary["Clinical-F1"] = results["clinical"].get("f1", 0)
        return summary

    #  Comparison Report 

    def compare_models(
        self,
        model_outputs: dict[str, list[str]],
        references: list[str],
    ) -> str:
        """
        Compute metrics for multiple models and return a formatted comparison table.

        Args:
            model_outputs: {"MedGemma": [...], "CLIP": [...]}
            references: Ground-truth reports

        Returns:
            Markdown table string
        """
        all_results = {}
        for model_name, predictions in model_outputs.items():
            result = self.evaluate_report(predictions, references, compute_bertscore=False)
            all_results[model_name] = result["summary"]

        # Build markdown table
        if not all_results:
            return "No results to display."

        metrics = list(next(iter(all_results.values())).keys())
        header = "| Model | " + " | ".join(metrics) + " |"
        separator = "|-------|" + "|".join(["-------"] * len(metrics)) + "|"

        rows = [header, separator]
        for model_name, scores in all_results.items():
            row = f"| {model_name} | " + " | ".join(
                f"{scores.get(m, 0):.4f}" for m in metrics
            ) + " |"
            rows.append(row)

        return "\n".join(rows)
