"""
QA Dataset Creation Script
===========================
Creates a QA dataset from MIMIC-CXR radiology reports.

The MIMIC-CXR dataset has a "text" column containing radiology reports.
This script converts those reports into QA pairs for Mode 2 (Clinical QA).

Method:
  1. Load reports from MIMIC-CXR (text column = report)
  2. For each report, generate questions based on 15 clinical categories
     (following MIMIC-CXR-VQA methodology)
  3. Generate answers from the report text using rule-based extraction
     OR using an LLM (LLaMA / MedGemma) for richer answers
  4. Save as JSON for use in the RAG knowledge base

Usage:
    # Rule-based (no GPU needed):
    python scripts/create_qa_dataset.py \
        --reports_csv data/mimic_cxr/reports.csv \
        --output data/qa_dataset.json \
        --method rules

    # LLM-based (needs GPU, better quality):
    python scripts/create_qa_dataset.py \
        --reports_csv data/mimic_cxr/reports.csv \
        --output data/qa_dataset.json \
        --method llm \
        --hf_token hf_xxx

Reference: MIMIC-CXR-VQA Dataset Creation
    https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Clinical Categories & Question Templates ───────────────────────────────────
# Following MIMIC-CXR-VQA methodology (Aas-Alas et al., 2026)

QUESTION_TEMPLATES = {
    "Pneumonia": [
        "Is there any evidence of pneumonia?",
        "Can pneumonia be identified in this chest X-ray?",
        "Are there signs of pneumonia?",
    ],
    "Consolidation": [
        "Is consolidation observed?",
        "Are there signs of consolidation?",
        "What evidence of consolidation is present?",
    ],
    "Pleural Effusion": [
        "Are there any pleural effusions?",
        "Can pleural effusion be detected?",
        "What evidence of pleural effusion is visible?",
    ],
    "Cardiomegaly": [
        "Is cardiomegaly present?",
        "Does the heart appear enlarged?",
        "Is the cardiac silhouette within normal limits?",
    ],
    "Atelectasis": [
        "Is atelectasis observed?",
        "Are there signs of atelectasis?",
        "Can atelectasis be identified?",
    ],
    "Pneumothorax": [
        "Is there a pneumothorax?",
        "Can pneumothorax be detected?",
        "Are there signs of pneumothorax?",
    ],
    "Edema": [
        "Is pulmonary edema present?",
        "Are there signs of edema?",
        "What evidence of edema is visible?",
    ],
    "No Finding": [
        "Are there any abnormal findings?",
        "What are the main findings in this chest X-ray?",
        "Is this a normal chest X-ray?",
    ],
}

# Keywords to detect findings in report text
FINDING_KEYWORDS = {
    "Pneumonia":       ["pneumonia", "infectious", "bacterial", "infection"],
    "Consolidation":   ["consolidation", "opacity", "opacification", "airspace"],
    "Pleural Effusion":["effusion", "pleural fluid", "pleural"],
    "Cardiomegaly":    ["cardiomegaly", "enlarged heart", "cardiac enlargement", "cardiothoracic ratio"],
    "Atelectasis":     ["atelectasis", "atelectatic", "collapse", "linear opacity"],
    "Pneumothorax":    ["pneumothorax", "pneumothoraces"],
    "Edema":           ["edema", "pulmonary edema", "vascular congestion", "engorgement"],
    "No Finding":      ["no acute", "unremarkable", "normal", "clear", "no evidence"],
}


class RuleBasedQAGenerator:
    """
    Generates QA pairs from radiology reports using rule-based text extraction.

    This approach:
    1. Detects which findings are present/absent using keyword matching
    2. Generates appropriate questions for each finding
    3. Extracts relevant sentences from the report as answers
    4. Labels each QA pair as positive/negative/uncertain

    Advantages: No GPU needed, fast, deterministic
    Limitations: Less natural language variety than LLM-generated answers
    """

    def generate(self, report_text: str, image_id: str) -> list[dict]:
        """Generate QA pairs for one report."""
        report_lower = report_text.lower()
        qa_pairs = []

        for category, keywords in FINDING_KEYWORDS.items():
            # Detect if finding is present
            found = any(kw in report_lower for kw in keywords)

            # Pick a random question template
            questions = QUESTION_TEMPLATES.get(category, [])
            if not questions:
                continue
            question = random.choice(questions)

            # Generate answer from report sentences
            answer = self._extract_answer(
                report_text, keywords, category, found
            )

            qa_pairs.append({
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "category": category,
                "label": "positive" if found else "negative",
                "source": "rule_based",
            })

        return qa_pairs

    def _extract_answer(
        self,
        report: str,
        keywords: list[str],
        category: str,
        found: bool,
    ) -> str:
        """Extract relevant sentences from report as answer."""
        sentences = re.split(r'[.!?]\s+', report)

        # Find sentences mentioning the category
        relevant = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in keywords):
                relevant.append(sent.strip())

        if relevant:
            answer = ". ".join(relevant[:2]) + "."
            # Rephrase to observational style (per MIMIC-CXR-VQA guidelines)
            answer = answer.replace("mentioned in the report", "observed in the radiograph")
            answer = answer.replace("the report shows", "the radiograph demonstrates")
            return answer
        elif found:
            return f"{category} findings are observed in the radiograph."
        else:
            return f"No {category.lower()} is identified in the radiograph."


class LLMQAGenerator:
    """
    Generates QA pairs using MedGemma (or any LLM).

    Produces richer, more natural answers following the MIMIC-CXR-VQA
    prompt structure (Aas-Alas et al., 2026).

    Requires GPU + HF_TOKEN.
    """

    def __init__(self, hf_token: str = ""):
        self.hf_token = hf_token
        self.model = None

    def load(self):
        from src.models.medgemma import MedGemmaModel
        self.model = MedGemmaModel(hf_token=self.hf_token).load()
        return self

    SYSTEM_PROMPT = """You are a radiologist assistant. Given a radiology report and a question,
generate a concise, evidence-based answer strictly from the report findings.
Do not reference the report directly — describe findings as if observing the radiograph.
Do not use comparison terms like 'unchanged' or 'stable'.
Return only the answer, no preamble."""

    def generate(self, report_text: str, image_id: str) -> list[dict]:
        """Generate QA pairs using LLM for richer answers."""
        qa_pairs = []
        rule_gen = RuleBasedQAGenerator()

        # Get questions from rule-based (reuse question generation)
        rule_pairs = rule_gen.generate(report_text, image_id)

        for pair in rule_pairs:
            prompt = (
                f"Report findings: {report_text}\n\n"
                f"Question: {pair['question']}\n\n"
                "Answer strictly based on the report:"
            )
            try:
                answer = self.model.generate(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    max_new_tokens=150,
                )
                pair["answer"] = answer.strip()
                pair["source"] = "llm_medgemma"
            except Exception as e:
                logger.warning(f"LLM failed for {image_id}: {e} — using rule answer")
                pair["source"] = "rule_based_fallback"

            qa_pairs.append(pair)

        return qa_pairs


def load_reports(reports_path: Path, text_col: str = "text", limit: int = None) -> list[dict]:
    """
    Load reports from MIMIC-CXR dataset.

    The MIMIC-CXR dataset CSV has a 'text' column containing the report.
    Each row = one study with: study_id, subject_id, text (report), split
    """
    import csv

    reports = []
    suffix = reports_path.suffix.lower()

    if suffix == ".csv":
        with open(reports_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                text = row.get(text_col, row.get("report", "")).strip()
                if len(text) > 50:
                    reports.append({
                        "image_id": row.get("study_id", f"study_{i}"),
                        "subject_id": row.get("subject_id", ""),
                        "split": row.get("split", "train"),
                        "report_text": text,
                    })

    elif suffix == ".json":
        with open(reports_path) as f:
            data = json.load(f)
        for i, item in enumerate(data[:limit] if limit else data):
            reports.append({
                "image_id": item.get("study_id", item.get("id", f"study_{i}")),
                "report_text": item.get("text", item.get("report", "")),
                "split": item.get("split", "train"),
            })

    elif reports_path.is_dir():
        # Directory of .txt files
        for i, f in enumerate(sorted(reports_path.glob("*.txt"))):
            if limit and i >= limit:
                break
            text = f.read_text(encoding="utf-8").strip()
            if len(text) > 50:
                reports.append({
                    "image_id": f.stem,
                    "report_text": text,
                    "split": "train",
                })

    logger.info(f"Loaded {len(reports)} reports from {reports_path}")
    return reports


def create_qa_dataset(
    reports: list[dict],
    method: str = "rules",
    hf_token: str = "",
) -> list[dict]:
    """
    Create QA dataset from reports.

    Args:
        reports: List of report dicts with 'image_id' and 'report_text'
        method: 'rules' (fast, no GPU) or 'llm' (better quality, needs GPU)
        hf_token: HuggingFace token for LLM method

    Returns:
        List of QA pair dicts
    """
    if method == "llm":
        logger.info("Using LLM-based QA generation (MedGemma)...")
        generator = LLMQAGenerator(hf_token=hf_token).load()
    else:
        logger.info("Using rule-based QA generation...")
        generator = RuleBasedQAGenerator()

    all_qa_pairs = []
    for report in reports:
        pairs = generator.generate(
            report_text=report["report_text"],
            image_id=report["image_id"],
        )
        # Add split info
        for p in pairs:
            p["split"] = report.get("split", "train")
        all_qa_pairs.extend(pairs)

    logger.info(
        f"Created {len(all_qa_pairs)} QA pairs from {len(reports)} reports "
        f"({len(all_qa_pairs)/len(reports):.1f} pairs/report)"
    )
    return all_qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Create QA Dataset from MIMIC-CXR reports")
    parser.add_argument("--reports_csv",  type=Path,
                        default=ROOT / "data/sample_reports",
                        help="Path to MIMIC-CXR CSV, JSON, or directory of .txt files")
    parser.add_argument("--output",       type=Path,
                        default=ROOT / "data/qa_dataset.json")
    parser.add_argument("--method",       choices=["rules", "llm"], default="rules",
                        help="rules=fast/no-GPU, llm=MedGemma/needs-GPU")
    parser.add_argument("--text_col",     default="text",
                        help="Column name for report text in CSV")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--hf_token",     default="")
    args = parser.parse_args()

    # Load reports
    reports = load_reports(args.reports_csv, text_col=args.text_col, limit=args.limit)
    if not reports:
        logger.error("No reports loaded.")
        sys.exit(1)

    # Create QA pairs
    qa_pairs = create_qa_dataset(reports, method=args.method, hf_token=args.hf_token)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    # Stats
    categories = {}
    for pair in qa_pairs:
        cat = pair["category"]
        categories[cat] = categories.get(cat, 0) + 1

    logger.info(f"Saved {len(qa_pairs)} QA pairs to {args.output}")
    logger.info("Category distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()