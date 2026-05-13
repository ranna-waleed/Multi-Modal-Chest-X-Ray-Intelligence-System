"""
Build Knowledge Base
Offline script to index radiology reports into the vector store
for use by the RAG retrieval pipeline.

Usage:
    # Using CLIP (fast, CPU-friendly):
    python scripts/build_knowledge_base.py \\
        --reports_dir data/mimic_cxr/reports \\
        --output_dir data/knowledge_base \\
        --backend clip

    # Using ColPali (better retrieval, needs GPU):
    python scripts/build_knowledge_base.py \\
        --reports_dir data/mimic_cxr/reports \\
        --output_dir data/knowledge_base \\
        --backend colpali

    # From MIMIC-CXR-VQA JSON:
    python scripts/build_knowledge_base.py \\
        --json_file data/mimic_cxr_vqa/train.json \\
        --output_dir data/knowledge_base \\
        --backend clip \\
        --limit 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_reports_from_dir(reports_dir: Path, limit: int = None) -> list[dict]:
    """Load reports from a directory of .txt files."""
    files = sorted(reports_dir.glob("**/*.txt"))
    if limit:
        files = files[:limit]

    reports = []
    for f in tqdm(files, desc="Loading reports"):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > 20:  # Skip empty/very short reports
            reports.append({
                "id": f.stem,
                "text": text,
                "source": str(f.relative_to(ROOT)),
            })

    logger.info(f"Loaded {len(reports)} reports from {reports_dir}")
    return reports


def load_reports_from_json(json_file: Path, limit: int = None) -> list[dict]:
    """Load reports from MIMIC-CXR-VQA JSON format."""
    with open(json_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        reports = data
    elif isinstance(data, dict) and "reports" in data:
        reports = data["reports"]
    else:
        raise ValueError("Unrecognized JSON format")

    if limit:
        reports = reports[:limit]

    # Normalize keys
    normalized = []
    for r in reports:
        normalized.append({
            "id": r.get("id", r.get("study_id", str(len(normalized)))),
            "text": r.get("text", r.get("report", r.get("findings", ""))),
            "source": "mimic_cxr_vqa",
        })

    logger.info(f"Loaded {len(normalized)} reports from {json_file}")
    return normalized


def build_clip_index(reports: list[dict], output_dir: Path, hf_token: str = ""):
    """Build CLIP FAISS index."""
    from src.models.clip_model import CLIPModel
    from src.rag.vector_store import FAISSVectorStore
    import numpy as np

    logger.info("Loading CLIP model...")
    clip = CLIPModel(hf_token=hf_token).load()

    logger.info(f"Embedding {len(reports)} reports...")
    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(reports), batch_size), desc="Embedding"):
        batch_texts = [r["text"][:500] for r in reports[i:i+batch_size]]
        embs = clip.embed_texts(batch_texts)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embedding shape: {embeddings.shape}")

    # Build and save FAISS index
    output_dir.mkdir(parents=True, exist_ok=True)
    store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    store.build(embeddings, reports)
    store.save(
        str(output_dir / "clip_index.faiss"),
        str(output_dir / "clip_metadata.json"),
    )
    logger.info(f" CLIP index saved to {output_dir}")


def build_colpali_index(reports: list[dict], output_dir: Path, hf_token: str = ""):
    """Build ColPali multi-vector index."""
    from src.models.colpali import ColPaliModel
    from src.rag.vector_store import ColPaliVectorStore
    from src.rag.retriever import RAGRetriever

    logger.info("Loading ColPali model...")
    colpali = ColPaliModel(hf_token=hf_token).load()

    logger.info(f"Rendering and embedding {len(reports)} reports as images...")
    retriever = RAGRetriever(backend="colpali", colpali_model=colpali)

    # Render reports as images and embed
    report_images = [
        retriever._text_to_image(r["text"])
        for r in tqdm(reports, desc="Rendering reports")
    ]

    doc_embeddings = []
    batch_size = colpali.batch_size
    for i in tqdm(range(0, len(report_images), batch_size), desc="Embedding"):
        batch = report_images[i:i+batch_size]
        embs = colpali.embed_images(batch, is_query=False)
        doc_embeddings.extend([embs[j] for j in range(embs.shape[0])])

    store = ColPaliVectorStore()
    store.add(doc_embeddings, reports)

    output_dir.mkdir(parents=True, exist_ok=True)
    store.save(str(output_dir / "colpali_store.pkl"))
    logger.info(f" ColPali store saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build CXR RAG Knowledge Base")
    parser.add_argument("--reports_dir", type=Path, help="Directory of .txt report files")
    parser.add_argument("--json_file", type=Path, help="JSON file with reports")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data/knowledge_base")
    parser.add_argument("--backend", choices=["clip", "colpali"], default="clip")
    parser.add_argument("--limit", type=int, default=None, help="Max number of reports")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace token")
    args = parser.parse_args()

    # Load reports
    if args.json_file:
        reports = load_reports_from_json(args.json_file, limit=args.limit)
    elif args.reports_dir:
        reports = load_reports_from_dir(args.reports_dir, limit=args.limit)
    else:
        logger.error("Provide --reports_dir or --json_file")
        sys.exit(1)

    if not reports:
        logger.error("No reports loaded")
        sys.exit(1)

    # Build index
    if args.backend == "clip":
        build_clip_index(reports, args.output_dir, hf_token=args.hf_token)
    elif args.backend == "colpali":
        build_colpali_index(reports, args.output_dir, hf_token=args.hf_token)

    logger.info(" Knowledge base built successfully!")


if __name__ == "__main__":
    main()
