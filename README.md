# 🫁 CXR Intelligence System
### Multi-Modal Chest X-Ray Analysis — Report Generation & Clinical QA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DSAI 413](https://img.shields.io/badge/Course-DSAI%20413-green.svg)]()

> **DSAI 413 — Assignment 2** | Multi-Modal Chest X-Ray Intelligence System (Dual-Mode: Report Generation & QA)

---

## 📋 Overview

This system provides two independent AI-powered modes for chest X-ray (CXR) analysis:

| Mode | Input | Output | Models |
|------|-------|--------|--------|
| **Report Generation** | CXR Image | Structured medical report | MedGemma + CLIP |
| **Clinical QA (RAG)** | CXR Image + Question | Grounded clinical answer | ColPali → MedGemma |

### Key Features
- ✅ **Dual-mode system** — independently operable report generation and QA
- ✅ **Multi-modal pipeline** — image + text processing
- ✅ **RAG-based QA** — retrieval-grounded answers from MIMIC-CXR knowledge base
- ✅ **Model comparison** — MedGemma vs CLIP with quantitative metrics
- ✅ **Gradio UI** — interactive demo with both modes
- ✅ **Modular codebase** — clean, extensible architecture

---

## 🏗️ Architecture

```
cxr-intelligence/
├── app/
│   └── app.py                 # Gradio demo application
├── config/
│   └── config.yaml            # Model & pipeline configuration
├── data/
│   └── sample_reports/        # Demo radiology reports for RAG
├── report/
│   └── report.md              # Short assignment report
├── scripts/
│   ├── build_knowledge_base.py  # Index reports into vector store
│   └── evaluate.py            # Quantitative model evaluation
├── src/
│   ├── models/
│   │   ├── medgemma.py        # MedGemma VLM wrapper
│   │   ├── colpali.py         # ColPali retrieval wrapper
│   │   └── clip_model.py      # CLIP vision-language wrapper
│   ├── modes/
│   │   ├── report_generation.py  # Report generation pipeline
│   │   └── qa_mode.py            # RAG-based QA pipeline
│   ├── rag/
│   │   ├── retriever.py       # RAG orchestration
│   │   └── vector_store.py    # FAISS + ColPali vector stores
│   ├── evaluation/
│   │   └── metrics.py         # BLEU, ROUGE, BERTScore, Clinical-F1
│   └── utils/
│       └── image_processing.py  # DICOM/image loading & preprocessing
└── requirements.txt
```

---

## 🤖 Models

### 1. MedGemma (Primary Model — Mandatory)
- **HuggingFace**: [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
- **Type**: Vision-Language Model (multimodal, instruction-tuned)
- **Architecture**: Gemma 2 + SigLIP vision encoder
- **Usage**: Report generation (primary), clinical QA (generator)
- **VRAM**: ~4GB (4-bit quantization) / ~8GB (BF16)

### 2. ColPali (Mandatory)
- **HuggingFace**: [`vidore/colpali-v1.2`](https://huggingface.co/vidore/colpali-v1.2)
- **Type**: Multi-vector Late-Interaction Retrieval
- **Architecture**: PaliGemma + MaxSim scoring
- **Usage**: RAG knowledge base retrieval
- **Paper**: [ColPali: Efficient Document Retrieval with VLMs](https://arxiv.org/abs/2407.01449)

### 3. CLIP (Suggested — for comparison)
- **HuggingFace**: [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)
- **Type**: Contrastive Vision-Language
- **Usage**: Zero-shot report generation (comparison baseline), FAISS retrieval
- **Paper**: [Learning Transferable Visual Models (Radford 2021)](https://arxiv.org/abs/2103.00020)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/cxr-intelligence.git
cd cxr-intelligence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. HuggingFace Token

MedGemma and ColPali require accepting their model licenses on HuggingFace:

1. Visit [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) → Accept license
2. Visit [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) → Accept license
3. Create a HF token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Run Demo Application

```bash
# Demo mode (no GPU/token required — placeholder outputs)
python app/app.py --demo

# Full mode (requires GPU + HF_TOKEN)
python app/app.py --hf_token $HF_TOKEN

# Public sharing link
python app/app.py --share
```

Open http://localhost:7860 in your browser.

---

## 📊 Pipeline Details

### Mode 1: Report Generation

```python
from src.models.medgemma import MedGemmaModel
from src.modes.report_generation import ReportGenerationPipeline
from PIL import Image

# Load models
medgemma = MedGemmaModel(hf_token="hf_...").load()

# Create pipeline
pipeline = ReportGenerationPipeline(medgemma_model=medgemma)

# Generate report
image = Image.open("chest_xray.jpg")
result = pipeline.generate(image, clinical_context="65-year-old with cough")

print(result["medgemma_result"].findings)
print(result["medgemma_result"].impression)
```

### Mode 2: Clinical QA (RAG)

```python
from src.models.medgemma import MedGemmaModel
from src.models.colpali import ColPaliModel
from src.rag.retriever import RAGRetriever
from src.modes.qa_mode import QAPipeline
from PIL import Image

# Load models
medgemma = MedGemmaModel(hf_token="hf_...").load()
colpali = ColPaliModel(hf_token="hf_...").load()

# Setup RAG retriever
retriever = RAGRetriever(backend="colpali", colpali_model=colpali)
retriever.load_knowledge_base("data/sample_reports")

# Create QA pipeline
qa = QAPipeline(medgemma_model=medgemma, retriever=retriever)

# Answer question
image = Image.open("chest_xray.jpg")
result = qa.answer(
    question="Is there any evidence of pneumonia?",
    image=image,
    top_k=3
)

print(result.answer)
print(f"Retrieved {len(result.retrieved_docs)} documents")
```

---

## 📦 Dataset: MIMIC-CXR

This system is designed to work with the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) dataset.

### Access
1. Create an account at [PhysioNet](https://physionet.org/)
2. Complete the required training (CITI course)
3. Request access to MIMIC-CXR
4. Or use the [Kaggle mirror](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)

### Build Knowledge Base from MIMIC-CXR

```bash
# Index reports for RAG (CLIP backend — fast):
python scripts/build_knowledge_base.py \
    --reports_dir /path/to/mimic_cxr/reports \
    --output_dir data/knowledge_base \
    --backend clip \
    --limit 50000

# ColPali backend (better retrieval, slower):
python scripts/build_knowledge_base.py \
    --reports_dir /path/to/mimic_cxr/reports \
    --output_dir data/knowledge_base \
    --backend colpali \
    --hf_token $HF_TOKEN
```

### MIMIC-CXR-VQA Dataset
For QA evaluation, use the [MIMIC-CXR-VQA dataset](https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation):
- 3.2M QA pairs across 15 clinical categories
- Generated using LLaMA 3.1 with structured prompts

---

## 📏 Evaluation

```bash
# Run evaluation on test set
python scripts/evaluate.py \
    --test_images data/test/images \
    --test_reports data/test/reports.json \
    --output results/evaluation.json

# View results
cat results/evaluation.json
```

Metrics computed:
| Metric | Description |
|--------|-------------|
| BLEU-1/2/3/4 | N-gram overlap |
| ROUGE-L | Longest common subsequence F1 |
| BERTScore-F1 | Semantic similarity (BioBERT) |
| Clinical Entity F1 | Medical term overlap |

---

## 🔧 Configuration

Edit `config/config.yaml` to change:
- Model IDs and quantization settings
- RAG top-K retrieval count
- Knowledge base paths
- Evaluation metrics

---

## 🌐 References

| Resource | Link |
|---------|------|
| MedGemma Model | [HuggingFace](https://huggingface.co/google/medgemma-4b-it) |
| MedGemma Paper | [DeepMind](https://deepmind.google/models/gemma/medgemma/) |
| ColPali Paper | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| CXR-RePaiR-Gen | [arXiv:2305.03660](https://arxiv.org/abs/2305.03660) |
| MIMIC-CXR-VQA | [GitHub](https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation) |
| MIMIC-CXR Dataset | [Kaggle](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset) |
| ColPali Cookbook | [HuggingFace](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) |
| Radiology Basics | [Radiology Assistant](https://radiologyassistant.nl/chest/chest-x-ray/basic-interpretation) |

---

## 🛠️ Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.10 | 3.11 |
| GPU VRAM | 8GB (4-bit) | 16GB |
| RAM | 16GB | 32GB |
| CUDA | 11.8 | 12.1 |
| Disk | 20GB | 50GB |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Medical Disclaimer**: This system is for educational purposes only. It is not intended for clinical use and should not replace professional medical judgment.
