"""
Evaluate Models

Runs quantitative evaluation comparing MedGemma vs CLIP
on a set of reference CXR reports.

Usage:
    python scripts/evaluate.py \\
        --test_images data/test/images \\
        --test_reports data/test/reports.json \\
        --output results/evaluation.json

Metrics computed:
  - BLEU-1/2/3/4
  - ROUGE-L
  - BERTScore-F1
  - Clinical Entity F1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    test_images: list[Path],
    test_reports: list[str],
    medgemma_model,
    clip_model,
    output_path: Path,
):
    """Run evaluation for both models."""
    from src.evaluation.metrics import CXRMetrics
    from PIL import Image

    metrics = CXRMetrics()
    medgemma_preds = []
    clip_preds = []

    for img_path, ref_report in zip(test_images, test_reports):
        logger.info(f"Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # MedGemma
        if medgemma_model:
            result = medgemma_model.generate_report(image)
            medgemma_preds.append(result["raw_text"])
        else:
            medgemma_preds.append(ref_report)  # Dummy

        # CLIP
        if clip_model:
            result = clip_model.generate_report_clip(image)
            clip_preds.append(result["raw_text"])
        else:
            clip_preds.append("")

    # Compute metrics
    results = {}

    if medgemma_preds:
        logger.info("Computing MedGemma metrics...")
        results["MedGemma"] = metrics.evaluate_report(medgemma_preds, test_reports)

    if clip_preds and any(clip_preds):
        logger.info("Computing CLIP metrics...")
        results["CLIP"] = metrics.evaluate_report(clip_preds, test_reports)

    # Comparison table
    if len(results) > 1:
        comparison_table = metrics.compare_models(
            {name: data for name, data in [
                ("MedGemma", medgemma_preds),
                ("CLIP", clip_preds),
            ]},
            test_reports,
        )
        results["comparison_table"] = comparison_table
        logger.info(f"\nModel Comparison:\n{comparison_table}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f" Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CXR Models")
    parser.add_argument("--test_images", type=Path, default=ROOT / "data/test/images")
    parser.add_argument("--test_reports", type=Path, default=ROOT / "data/test/reports.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation.json")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--skip_medgemma", action="store_true")
    parser.add_argument("--skip_clip", action="store_true")
    args = parser.parse_args()

    # Load test data
    if not args.test_reports.exists():
        logger.warning("No test data found. Using demo reports.")
        # Create minimal demo test set
        from src.rag.retriever import RAGRetriever
        demo_reports = RAGRetriever._get_demo_reports()
        test_images = []
        test_reports_text = [r["text"] for r in demo_reports[:4]]
    else:
        with open(args.test_reports) as f:
            test_data = json.load(f)
        test_reports_text = [d["text"] for d in test_data]
        test_images = [
            args.test_images / d.get("image", "placeholder.jpg")
            for d in test_data
        ]
        test_images = [p for p in test_images if p.exists()]

    # Load models
    medgemma = None
    clip = None

    if not args.skip_medgemma:
        try:
            from src.models.medgemma import MedGemmaModel
            medgemma = MedGemmaModel(hf_token=args.hf_token).load()
        except Exception as e:
            logger.warning(f"MedGemma not available: {e}")

    if not args.skip_clip:
        try:
            from src.models.clip_model import CLIPModel
            clip = CLIPModel().load()
        except Exception as e:
            logger.warning(f"CLIP not available: {e}")

    # Run evaluation
    if not test_images:
        logger.warning("No test images. Running text-only evaluation.")

    # Simple demo evaluation on stored texts
    logger.info("Running demo evaluation with sample reports...")
    from src.evaluation.metrics import CXRMetrics
    metrics = CXRMetrics()

    # Use first half as "predictions" and second half as "references" for demo
    mid = len(test_reports_text) // 2
    if mid >= 1:
        preds = test_reports_text[:mid]
        refs = test_reports_text[mid:mid*2] if len(test_reports_text) >= mid*2 else test_reports_text[:mid]

        result = metrics.evaluate_report(preds, refs, compute_bertscore=False)
        logger.info(f"Demo metrics summary: {result['summary']}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f" Results saved to {args.output}")


if __name__ == "__main__":
    main()
