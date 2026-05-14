"""
CXR Intelligence System — Gradio Demo Application
Dual-mode multi-modal chest X-ray AI system.

Run:
    python app/app.py --demo     # Demo mode (no GPU required)
    python app/app.py --share    # Public Gradio link
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image
import numpy as np

from src.modes.report_generation import ReportGenerationPipeline
from src.modes.qa_mode import QAPipeline
from src.rag.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEMO_MODE = True
report_pipeline: ReportGenerationPipeline = None
qa_pipeline: QAPipeline = None
retriever: RAGRetriever = None


#  Image Helper 

def to_pil(image) -> Image.Image:
    """
    Safely convert Gradio image input to PIL Image.
    Gradio 6 may return: numpy array, PIL Image, dict, or file path.
    """
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if isinstance(image, dict):
        # Gradio 6 returns {"path": ..., "url": ..., "size": ...}
        path = image.get("path") or image.get("url")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
    if isinstance(image, str) and Path(image).exists():
        return Image.open(image).convert("RGB")
    try:
        return Image.open(image).convert("RGB")
    except Exception:
        return None


# Model Initialization 

def initialize_models(demo_mode: bool = True, hf_token: str = ""):
    global report_pipeline, qa_pipeline, retriever

    logger.info(f"Initializing models (demo_mode={demo_mode})")

    medgemma_model = None
    clip_model = None
    colpali_model = None

    if not demo_mode:
        token = hf_token or os.getenv("HF_TOKEN", "")

        try:
            from src.models.medgemma import MedGemmaModel
            medgemma_model = MedGemmaModel(hf_token=token).load()
            logger.info(" MedGemma loaded")
        except Exception as e:
            logger.error(f" MedGemma failed: {e}")

        try:
            from src.models.clip_model import CLIPModel
            clip_model = CLIPModel().load()
            logger.info(" CLIP loaded")
        except Exception as e:
            logger.error(f" CLIP failed: {e}")

        try:
            from src.models.colpali import ColPaliModel
            colpali_model = ColPaliModel(hf_token=token).load()
            logger.info("ColPali loaded")
        except Exception as e:
            logger.error(f" ColPali failed: {e}")

    retriever = RAGRetriever(
        backend="colpali" if colpali_model else "clip",
        clip_model=clip_model,
        colpali_model=colpali_model,
    )
    retriever.load_knowledge_base(str(ROOT / "data" / "sample_reports"))

    report_pipeline = ReportGenerationPipeline(
        medgemma_model=medgemma_model,
        clip_model=clip_model,
    )

    qa_pipeline = QAPipeline(
        medgemma_model=medgemma_model,
        retriever=retriever,
    )

    status = "\n".join([
        f"MedGemma: {' Loaded' if medgemma_model else ' Demo mode'}",
        f"CLIP:     {' Loaded' if clip_model else ' Demo mode'}",
        f"ColPali:  {' Loaded' if colpali_model else ' Demo mode'}",
        f"RAG KB:    {len(retriever._knowledge_base)} reports indexed",
    ])
    logger.info("Model initialization complete.")
    return status


#  Tab 1: Report Generation 

def generate_report(image, clinical_context, run_medgemma, run_clip):
    pil_image = to_pil(image)
    if pil_image is None:
        return (
            " Please upload a chest X-ray image first.",
            "",
            "",
        )

    try:
        result = report_pipeline.generate(
            image=pil_image,
            clinical_context=clinical_context or "",
            run_medgemma=run_medgemma,
            run_clip=run_clip,
        )

        medgemma_out = result.get("medgemma_result", None)
        clip_out     = result.get("clip_result", None)
        comparison   = result.get("comparison", {})

        return (
            medgemma_out.format_display() if medgemma_out else "*Not run.*",
            clip_out.format_display()     if clip_out     else "*Not run.*",
            comparison.get("analysis", "*Run both models to see comparison.*"),
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        return f" Error: {e}", "", ""


#  Tab 2: QA Mode 

def answer_question(image, question, use_retrieval, top_k):
    if not question or not question.strip():
        return " Please enter a clinical question.", ""

    pil_image = to_pil(image)

    try:
        result = qa_pipeline.answer(
            question=question,
            image=pil_image,
            top_k=int(top_k),
            use_retrieval=use_retrieval,
        )
        return result.format_display(), result.format_retrieved_docs()
    except Exception as e:
        logger.error(f"QA error: {e}", exc_info=True)
        return f" Error: {e}", ""


#  Model Info 

def get_model_info():
    return """
##  Models Used

### 1. MedGemma — Primary Model (Mandatory)
- **ID**: `google/medgemma-4b-it`
- **Type**: Vision-Language Model (multimodal, instruction-tuned)
- **Task**: Report generation + Clinical QA
- **Architecture**: Gemma 2 + SigLIP vision encoder
- **Quantization**: 4-bit NF4 via bitsandbytes (~4 GB VRAM)
- **Strengths**: Full narrative reports, clinical reasoning, structured output
- **Limitations**: Needs GPU, potential hallucinations

### 2. CLIP — Comparison Model (Suggested)
- **ID**: `openai/clip-vit-large-patch14`
- **Type**: Contrastive Vision-Language
- **Task**: Zero-shot report generation (template ranking) + FAISS retrieval
- **Strengths**: Fast, interpretable, low hallucination risk, CPU-friendly
- **Limitations**: Template-only output, no free-text generation

### 3. ColPali — Retrieval Model (Mandatory)
- **ID**: `vidore/colpali-v1.2`
- **Type**: Multi-vector late-interaction retrieval
- **Task**: RAG knowledge base retrieval (MaxSim scoring)
- **Strengths**: Rich patch-level image understanding, state-of-the-art retrieval
- **Limitations**: Slower than CLIP single-vector retrieval

---

## 📊 Comparison Table

| Aspect | MedGemma | CLIP | ColPali |
|--------|----------|------|---------|
| Output | Full narrative | Template labels | Retrieval scores |
| Medical fine-tune |  Yes |  General | Yes |
| Report generation |  Full |  Template |  No |
| QA support | Yes |  No | No |
| RAG retrieval |  No |  Single-vector |  Multi-vector |
| GPU required |  ~8 GB |  Optional |  ~6 GB |
| Speed | Slow | Fast | Medium |
| Hallucination risk | Moderate | Low | N/A |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│              CXR Intelligence System                 │
├──────────────────────┬──────────────────────────────┤
│  MODE 1: REPORT GEN  │   MODE 2: QA (RAG)           │
│                      │                              │
│  CXR Image           │   CXR Image + Question       │
│       ↓              │         ↓                    │
│  MedGemma (primary)  │   ColPali Retriever          │
│  CLIP (comparison)   │         ↓                    │
│       ↓              │   Top-K Reports (Context)    │
│  Structured Report   │         ↓                    │
│  + Model Comparison  │   MedGemma Generator         │
│                      │         ↓                    │
│                      │   Grounded Answer            │
└──────────────────────┴──────────────────────────────┘
```
"""


#  Build UI 

def build_ui():

    css = """
    .warning-banner {
        background: #fef3c7; border: 1px solid #f59e0b;
        padding: 10px 16px; border-radius: 8px;
        margin: 0 0 12px 0; font-size: 0.9em;
    }
    """

    with gr.Blocks(title="CXR Intelligence System") as demo:

        # Header
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px 0">
            <h1 style="font-size:2.2em; font-weight:700; margin-bottom:6px">
                🫁 CXR Intelligence System
            </h1>
            <p style="color:#6b7280; font-size:1.05em">
                Multi-Modal Chest X-Ray Analysis · Report Generation &amp; Clinical QA
            </p>
            <p style="color:#9ca3af; font-size:0.9em">
                Models: MedGemma · CLIP · ColPali &nbsp;|&nbsp; DSAI 413 – Assignment 2
            </p>
        </div>
        """)

        if DEMO_MODE:
            gr.HTML("""
            <div style="background:#fef3c7; border:1px solid #f59e0b; padding:10px 16px;
                        border-radius:8px; margin:0 0 12px 0; font-size:0.9em">
                ⚠️ <strong>Demo Mode</strong> — Models not loaded.
                Outputs are placeholder examples showing system structure.
                To enable real inference: set <code>HF_TOKEN</code> and run on a GPU machine.
            </div>
            """)

        with gr.Tabs():

            #  Tab 1: Report Generation 
            with gr.Tab("📄 Report Generation"):
                gr.Markdown(
                    "Upload a chest X-ray → get a structured radiology report "
                    "from **MedGemma** compared with **CLIP**."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Chest X-Ray Image (JPG / PNG / JFIF)",
                            type="numpy",
                            height=320,
                        )
                        clinical_ctx = gr.Textbox(
                            label="Clinical Context (optional)",
                            placeholder="e.g. 65-year-old male, fever, productive cough...",
                            lines=3,
                        )
                        with gr.Row():
                            cb_medgemma = gr.Checkbox(label="Run MedGemma", value=True)
                            cb_clip     = gr.Checkbox(label="Run CLIP",     value=True)

                        btn_report = gr.Button(
                            " Generate Report", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("MedGemma Report"):
                                out_medgemma = gr.Markdown(
                                    value="*Upload an image and click Generate Report...*"
                                )
                            with gr.Tab("CLIP Report"):
                                out_clip = gr.Markdown(
                                    value="*CLIP output will appear here...*"
                                )
                            with gr.Tab("⚖️ Comparison"):
                                out_comparison = gr.Markdown(
                                    value="*Comparison appears after both models run...*"
                                )

                btn_report.click(
                    fn=generate_report,
                    inputs=[img_input, clinical_ctx, cb_medgemma, cb_clip],
                    outputs=[out_medgemma, out_clip, out_comparison],
                )

            #  Tab 2: QA Mode 
            with gr.Tab(" Clinical QA (RAG)"):
                gr.Markdown(
                    "Ask a clinical question. **ColPali** retrieves relevant reports, "
                    "then **MedGemma** generates a grounded answer."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        qa_img = gr.Image(
                            label="Chest X-Ray Image (optional)",
                            type="numpy",
                            height=280,
                        )
                        qa_question = gr.Textbox(
                            label="Clinical Question",
                            placeholder="e.g. Is there any evidence of pneumonia?",
                            lines=3,
                        )

                        gr.Markdown("**Quick Questions:**")
                        with gr.Row():
                            q1 = gr.Button("Pneumonia?",    size="sm")
                            q2 = gr.Button(" Effusion?",     size="sm")
                        with gr.Row():
                            q3 = gr.Button(" Cardiomegaly?", size="sm")
                            q4 = gr.Button(" All findings",  size="sm")

                        with gr.Row():
                            cb_rag  = gr.Checkbox(label="Use RAG retrieval", value=True)
                            sl_topk = gr.Slider(1, 8, value=3, step=1, label="Top-K docs")

                        btn_ask = gr.Button(
                            " Ask Question", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Answer"):
                                out_answer = gr.Markdown(
                                    value="*Answer will appear here...*"
                                )
                            with gr.Tab(" Retrieved Documents"):
                                out_docs = gr.Markdown(
                                    value="*Retrieved reports will appear here...*"
                                )

                # Quick question wiring
                q1.click(lambda: "Is there any evidence of pneumonia or consolidation?", outputs=qa_question)
                q2.click(lambda: "Are there any pleural effusions present?",             outputs=qa_question)
                q3.click(lambda: "Is cardiomegaly present? What is the cardiothoracic ratio?", outputs=qa_question)
                q4.click(lambda: "Please describe all findings in this chest X-ray.",    outputs=qa_question)

                btn_ask.click(
                    fn=answer_question,
                    inputs=[qa_img, qa_question, cb_rag, sl_topk],
                    outputs=[out_answer, out_docs],
                )

            #  Tab 3: Models 
            with gr.Tab(" Models & Architecture"):
                gr.Markdown(get_model_info())


        gr.HTML("""
        <div style="text-align:center;padding:12px;color:#9ca3af;font-size:0.85em;
                    border-top:1px solid #e5e7eb;margin-top:16px">
            CXR Intelligence · DSAI 413 Assignment 2 ·
            <a href="https://arxiv.org/abs/2305.03660">CXR-RePaiR-Gen</a> ·
            <a href="https://huggingface.co/google/medgemma-4b-it">MedGemma</a> ·
            <a href="https://huggingface.co/vidore/colpali-v1.2">ColPali</a>
        </div>
        """)

    return demo


#  Entry Point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",     action="store_true")
    parser.add_argument("--port",      type=int, default=7860)
    parser.add_argument("--demo",      action="store_true", default=False)
    parser.add_argument("--hf_token",  type=str, default="")
    args = parser.parse_args()

    global DEMO_MODE
    DEMO_MODE = args.demo or os.getenv("DEMO_MODE", "true").lower() == "true"

    status = initialize_models(demo_mode=DEMO_MODE, hf_token=args.hf_token)
    logger.info(f"Initialization:\n{status}")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()