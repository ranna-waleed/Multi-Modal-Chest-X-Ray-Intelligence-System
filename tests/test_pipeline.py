"""
Unit Tests for CXR Intelligence System
=======================================
Tests core pipeline logic without requiring GPU/models (uses mocks).

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from PIL import Image
import numpy as np


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """Create a dummy 512x512 chest X-ray placeholder."""
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_reports():
    """Sample radiology reports for testing."""
    return [
        {"id": "r001", "text": "No acute cardiopulmonary process."},
        {"id": "r002", "text": "Right lower lobe pneumonia. Mild cardiomegaly."},
        {"id": "r003", "text": "Bilateral pleural effusions. Pulmonary edema."},
    ]


# ─── Test Report Result ────────────────────────────────────────────────────────

def test_report_result_format():
    from src.modes.report_generation import ReportResult
    r = ReportResult(
        model_name="TestModel",
        raw_text="Findings: Clear lungs. Impression: Normal.",
        findings="Clear lungs.",
        impression="Normal.",
        recommendations="None.",
        generation_time_s=1.5,
    )
    display = r.format_display()
    assert "TestModel" in display
    assert "Clear lungs" in display


def test_report_result_to_dict():
    from src.modes.report_generation import ReportResult
    r = ReportResult(
        model_name="TestModel",
        raw_text="Normal CXR.",
        findings="Clear.",
        impression="Normal.",
        recommendations="",
        generation_time_s=0.5,
    )
    d = r.to_dict()
    assert d["model"] == "TestModel"
    assert d["generation_time_s"] == 0.5


# ─── Test QA Result ────────────────────────────────────────────────────────────

def test_qa_result_format():
    from src.modes.qa_mode import QAResult
    r = QAResult(
        question="Is there pneumonia?",
        answer="No pneumonia is identified.",
        retrieved_docs=[{"rank": 1, "score": 0.8, "text": "Clear lungs."}],
        retrieval_backend="clip",
        generation_time_s=2.0,
        context_used="Clear lungs.",
    )
    display = r.format_display()
    assert "pneumonia" in display.lower()
    assert "No pneumonia" in display


# ─── Test Vector Store (No GPU) ───────────────────────────────────────────────

def test_faiss_vector_store():
    try:
        import faiss
    except ImportError:
        pytest.skip("FAISS not installed")

    from src.rag.vector_store import FAISSVectorStore

    dim = 128
    N = 10
    embeddings = np.random.randn(N, dim).astype(np.float32)
    # Normalize
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

    metadata = [{"id": f"doc_{i}", "text": f"Report {i}"} for i in range(N)]

    store = FAISSVectorStore(embedding_dim=dim)
    store.build(embeddings, metadata)

    # Search
    query = np.random.randn(1, dim).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, top_k=3)

    assert len(results) == 1          # 1 query
    assert len(results[0]) == 3       # top-3 results
    assert results[0][0]["rank"] == 1
    assert "text" in results[0][0]


# ─── Test Retriever (Demo Mode) ───────────────────────────────────────────────

def test_retriever_demo_mode(dummy_image, sample_reports):
    try:
        import faiss
    except ImportError:
        pytest.skip("FAISS not installed")
    _test_retriever_demo_mode_inner(dummy_image, sample_reports)

def _test_retriever_demo_mode_inner(dummy_image, sample_reports):
    from src.rag.retriever import RAGRetriever

    retriever = RAGRetriever(backend="clip", clip_model=None)
    # Load demo knowledge base (no model needed)
    retriever._knowledge_base = sample_reports
    retriever._build_index(rebuild=True)  # Will use dummy embeddings

    results = retriever.retrieve(query="pneumonia", top_k=2)
    assert len(results) >= 1


def test_retriever_format_context():
    from src.rag.retriever import RAGRetriever

    retriever = RAGRetriever()
    results = [
        {"rank": 1, "score": 0.9, "text": "Normal CXR. No acute findings."},
        {"rank": 2, "score": 0.7, "text": "Bilateral pleural effusions."},
    ]
    context = retriever.format_context(results)
    assert "Normal CXR" in context
    assert "Report 1" in context


# ─── Test Evaluation Metrics ──────────────────────────────────────────────────

def test_bleu_metric():
    from src.evaluation.metrics import CXRMetrics
    metrics = CXRMetrics()

    preds = ["no acute cardiopulmonary findings clear lungs"]
    refs = ["no acute cardiopulmonary process lungs clear bilaterally"]

    result = metrics.evaluate_report(preds, refs, compute_bertscore=False)
    assert "bleu" in result
    assert result["bleu"]["bleu_1"] > 0


def test_rouge_metric():
    from src.evaluation.metrics import CXRMetrics
    metrics = CXRMetrics()

    preds = ["bilateral pleural effusions cardiomegaly pulmonary edema"]
    refs = ["bilateral pleural effusions with cardiomegaly and pulmonary edema"]

    result = metrics.evaluate_report(preds, refs, compute_bertscore=False)
    assert "rouge" in result
    assert result["rouge"]["rougeL"] > 0.5


def test_clinical_entity_score():
    from src.evaluation.metrics import CXRMetrics
    metrics = CXRMetrics()

    preds = ["pneumonia and effusion present"]
    refs = ["right lower lobe pneumonia with pleural effusion"]

    result = metrics._compute_clinical_entity_score(preds, refs)
    assert result["recall"] > 0


def test_compare_models_table():
    from src.evaluation.metrics import CXRMetrics
    metrics = CXRMetrics()

    refs = ["clear lungs no acute findings", "bilateral effusions cardiomegaly"]
    preds_a = ["clear lungs bilaterally no findings", "bilateral effusions enlarged heart"]
    preds_b = ["normal chest", "effusions heart enlarged"]

    table = metrics.compare_models({"ModelA": preds_a, "ModelB": preds_b}, refs)
    assert "ModelA" in table
    assert "ModelB" in table


# ─── Test Image Processing ────────────────────────────────────────────────────

def test_preprocess_for_model(dummy_image):
    from src.utils.image_processing import preprocess_for_model

    processed = preprocess_for_model(dummy_image, model_type="medgemma", max_size=512)
    assert processed.mode == "RGB"
    assert max(processed.size) <= 512


def test_large_image_resize():
    from src.utils.image_processing import preprocess_for_model

    big_img = Image.new("RGB", (2048, 1536))
    processed = preprocess_for_model(big_img, max_size=512)
    assert max(processed.size) <= 512


# ─── Test MedGemma Parser ─────────────────────────────────────────────────────

def test_medgemma_parse_sections():
    from src.models.medgemma import MedGemmaModel

    text = """
**Findings:**
The lungs are clear bilaterally. No consolidation or effusion.
Cardiac size is normal.

**Impression:**
No acute cardiopulmonary process.

**Recommendations:**
Routine follow-up as clinically indicated.
"""

    result = MedGemmaModel._parse_report_sections(text)
    assert "clear" in result["findings"].lower() or len(result["findings"]) > 0
    # At minimum the raw text is preserved
    assert result["findings"] or result["impression"] or text in result.get("findings", "")


# ─── Test QA Suggested Questions ─────────────────────────────────────────────

def test_suggested_questions():
    from src.modes.qa_mode import QAPipeline
    questions = QAPipeline.get_suggested_questions()
    assert len(questions) >= 5
    assert all(isinstance(q, str) for q in questions)
