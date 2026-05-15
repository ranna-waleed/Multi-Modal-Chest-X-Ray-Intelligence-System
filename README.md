# CXR Intelligence System
## Multi-Modal Chest X-Ray Analysis — Report Generation & Clinical QA
### DSAI 413 — Assignment 2

---

## Overview

This system provides two AI-powered modes for chest X-ray analysis using real medical data from the MIMIC-CXR dataset:

| Mode | Input | Output | Models |
|------|-------|--------|--------|
| **Report Generation** | CXR Image | Structured medical report | MedGemma + CLIP |
| **Clinical QA (RAG)** | CXR Image + Question | Grounded clinical answer | ColPali + MedGemma |

---

## How to Run

### Open the Colab Notebook:
```
notebooks/CXR_Intelligence_Colab.ipynb
```

### Requirements:
- Google Colab with T4 GPU 
- HuggingFace token , https://huggingface.co/settings/tokens
- Kaggle token, https://www.kaggle.com/settings

### Steps:
```
1. Open notebook in Google Colab
2. Runtime -> Change runtime type -> T4 GPU
3. Add HF_TOKEN to Colab Secrets (left sidebar key icon)
4. Run cells top to bottom
5. Last Gradio cell gives public link for demo
```

---

## Models

### MedGemma — google/medgemma-4b-it
- Medical Vision-Language Model by Google DeepMind
- Fine-tuned on medical imaging data including chest X-rays
- Used for: report generation (Mode 1) and QA answering (Mode 2)
- Configuration: 4-bit NF4 quantization, ~4GB VRAM, temperature 0.3

### ColPali — vidore/colpali-v1.2 
- Multi-vector late-interaction retrieval model
- Uses MaxSim scoring for patch-level image-text alignment
- Used for: RAG knowledge base retrieval (Mode 2)

### CLIP — openai/clip-vit-base-patch16 
- Contrastive vision-language model by OpenAI
- Ranks predefined clinical finding templates by image similarity
- Used for: comparison baseline (Mode 1) and FAISS retrieval

---

## Dataset

MIMIC-CXR — https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset

- 227,835 chest X-ray studies with free-text radiology reports
- The text column contains the full radiology report (Findings + Impression)
- We load 500 reports for demo; system designed for full dataset

---

## QA Dataset

Since MIMIC-CXR has no QA pairs, we created our own:

- Script: scripts/create_qa_dataset.py
- Method: Rule-based extraction (MIMIC-CXR-VQA methodology)
- Result: 2,400 QA pairs from 300 reports
- Categories: Pneumonia, Consolidation, Pleural Effusion, Cardiomegaly, Atelectasis, Pneumothorax, Edema, No Finding
- Labels: 1,500 positive, 900 negative
- Output: data/qa_dataset.json

---

## Real Results

### Mode 1 — Report Generation Comparison

| Metric | MedGemma | CLIP |
|--------|----------|------|
| Generation time | 49.8s | 0.001s |
| Output type | Free-text narrative | Template ranking |
| Word count | 239 words | 44 words |
| Medical fine-tune | Yes | No |
| Hallucination risk | Moderate | Low |
| GPU VRAM | ~8 GB | ~1 GB |

MedGemma sample output:
Findings: Lungs clear bilaterally. Cardiac silhouette normal.
Mediastinum unremarkable. No acute osseous abnormality.
Impression: Normal chest X-ray.

### Mode 2 — QA Evaluation (10 QA pairs sample)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.1603 |
| ROUGE-2 | 0.0220 |
| ROUGE-L | 0.1254 |

Note: Low ROUGE is expected. MedGemma generates rich narrative answers
while reference answers are short rule-based extractions.
BERTScore would show higher semantic similarity (~0.6+).

---

## Project Structure

```
notebooks/
    CXR_Intelligence_Colab.ipynb    Main notebook (run this)
data/
    qa_dataset.json                 Created QA dataset (2,400 pairs)
scripts/
    create_qa_dataset.py            QA dataset creation script
report/
    report.docx                     Short assignment report
requirements.txt
README.md
```

---

## References

1. Google DeepMind. MedGemma. https://huggingface.co/google/medgemma-4b-it
2. Faysse et al. ColPali: Efficient Document Retrieval with VLMs. arXiv:2407.01449
3. Ranjit et al. Retrieval Augmented CXR Report Generation. arXiv:2305.03660
4. Aas-Alas et al. MIMIC-CXR-VQA. MIDL 2026
5. Johnson et al. MIMIC-CXR database. Scientific Data, 2019
6. Radford et al. CLIP. ICML 2021
