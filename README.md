# 🏛️ Indonesian Legal Comparator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-3.50.2-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/🤗_Hugging_Face-Space-FFD21E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-2D6A4F?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-EN_↔_ID-C9A84C?style=for-the-badge"/>
</p>

<p align="center">
  <b>NLP model for comparing Indonesian anti-corruption law against UNCAC international standards.</b><br/>
  Cross-lingual semantic analysis · Zero annotation · Open source data only
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/YOUR_USERNAME/indonesian-legal-comparator-demo">🌐 Live Demo</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/YOUR_USERNAME/indonesian-legal-comparator">🤗 Model Hub</a>
  &nbsp;·&nbsp;
  <a href="#-quick-start">🚀 Quick Start</a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Corpus](#-corpus)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Notebooks](#-notebooks)
- [Model Details](#-model-details)
- [Gap Analysis Results](#-gap-analysis-results)
- [Deploy to Hugging Face](#-deploy-to-hugging-face)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📖 Overview

This project builds an **unsupervised NLP pipeline** to analyze how Indonesia's national anti-corruption legislation aligns with the **United Nations Convention Against Corruption (UNCAC)** — the primary international legal framework against corruption.

Indonesia ratified UNCAC via **UU No. 7 Tahun 2006**, legally committing to implement its provisions. This model provides an objective, data-driven way to:

- **Measure** how comprehensively UNCAC has been adopted into Indonesian law
- **Identify gaps** — provisions required by UNCAC but absent in national legislation
- **Navigate** complex legal texts through semantic search in both Indonesian and English
- **Compare** how anti-corruption law has evolved across different Indonesian statutes

### Key design choices

| Choice | Rationale |
|---|---|
| **Zero annotation** | Structural hierarchy of legal documents (Bab → Pasal → Ayat) acts as implicit supervision |
| **Open source data only** | All 7 documents are publicly available from UNODC and JDIH Indonesia |
| **Cross-lingual (EN ↔ ID)** | UNCAC is in English; Indonesian laws are in Bahasa Indonesia — LaBSE bridges both |
| **Unsupervised methods** | LaBSE embeddings + cosine similarity require no labelled training data |

---

## ✨ Features

### 1. 📐 Similarity Score
Compare any two specific articles/pasal semantically — across languages.

```python
model.similarity_score("UNCAC_article_15", "UU_31_1999_pasal_5")
# → score: 0.8734 | FULLY ADOPTED ✅
```

### 2. 📊 Gap Analysis
Automatically detect which UNCAC provisions are fully adopted, partially adopted, or missing in Indonesian law.

```python
model.compare(doc_a="UNCAC", doc_b="UU_31_1999")
# → ✅ Fully Adopted : 18 articles (25.4%)
# → 🟡 Partially    : 31 articles (43.7%)
# → ❌ GAP           : 22 articles (31.0%)
```

### 3. 🔎 Semantic Search
Find relevant articles using free-form natural language in Indonesian or English.

```python
model.search("suap kepada pejabat negara", top_n=5)
model.search("asset recovery proceeds of crime", top_n=5)
```

### 4. 📝 Summarize Gap
Generate human-readable comparison summaries in three formats.

```python
model.summarize_gap(result, gaya="naratif")  # narrative report
model.summarize_gap(result, gaya="poin")     # markdown bullet points
model.summarize_gap(result, gaya="tabel")    # structured table
```

---

## 📚 Corpus

All documents sourced from **open, publicly available government repositories**.

| Label | Document | Year | Lang | Source |
|---|---|---|---|---|
| `UNCAC` | UN Convention Against Corruption | 2003 | EN | [UNODC](https://www.unodc.org/unodc/en/corruption/uncac.html) |
| `UU_7_2006` | UU Ratifikasi UNCAC | 2006 | ID | [JDIH](https://jdih.setneg.go.id) |
| `UU_31_1999` | UU Pemberantasan Tindak Pidana Korupsi | 1999 | ID | [JDIH](https://jdih.setneg.go.id) |
| `UU_20_2001` | Perubahan UU Tipikor | 2001 | ID | [JDIH](https://jdih.setneg.go.id) |
| `UU_28_1999` | UU Penyelenggaraan Negara Bersih & Bebas KKN | 1999 | ID | [JDIH](https://jdih.setneg.go.id) |
| `UU_30_2002` | UU Komisi Pemberantasan Korupsi (KPK) | 2002 | ID | [JDIH](https://jdih.setneg.go.id) |
| `UU_19_2019` | UU Perubahan KPK | 2019 | ID | [JDIH](https://jdih.setneg.go.id) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│        7 PDF files  (UNCAC EN + 6 Indonesian laws ID)      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                            │
│                                                             │
│  pdfplumber  →  raw text extraction per page               │
│  Regex       →  split text by Pasal N / Article N          │
│  Cleaner     →  remove noise, normalize whitespace         │
│  JSON builder → structured corpus with metadata            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  EMBEDDING LAYER                            │
│                                                             │
│  sentence-transformers/LaBSE                               │
│  ├── 109 languages including EN and ID                     │
│  ├── 768-dimensional sentence vectors                      │
│  └── L2-normalized for cosine similarity                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌──────────────┐  ┌──────────┐  ┌──────────────┐
      │  Similarity  │  │  Search  │  │Topic Modeling│
      │  + Gap       │  │  (FAISS- │  │BERTopic +    │
      │  Analysis    │  │  style)  │  │KeyBERT       │
      └──────────────┘  └──────────┘  └──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               DISTRIBUTION LAYER                            │
│                                                             │
│  🤗 Hugging Face Hub    → corpus.json + embeddings.npy     │
│  🤗 Hugging Face Spaces → Gradio full dashboard            │
│  📦 GitHub              → source code + 3 notebooks        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A — Live Demo (No installation)
Visit: **[huggingface.co/spaces/YOUR_USERNAME/indonesian-legal-comparator-demo](https://huggingface.co/spaces/YOUR_USERNAME/indonesian-legal-comparator-demo)**

### Option B — Run locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/indonesian-legal-comparator.git
cd indonesian-legal-comparator

# 2. Install
pip install -r requirements.txt

# 3. Download model files
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id   = 'YOUR_USERNAME/indonesian-legal-comparator',
    local_dir = './model_files'
)
"

# 4. Use
python -c "
import sys; sys.path.insert(0, './src')
from legal_comparator import LegalComparator
model = LegalComparator()
model.load('./model_files/corpus_structured.json',
           './model_files/embeddings_labse.npy')
print(model.daftar_dokumen())
"
```

### Option C — Google Colab

| Notebook | Link |
|---|---|
| 01 — Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/indonesian-legal-comparator/blob/main/notebooks/01_Preprocessing_Pipeline.ipynb) |
| 02 — Training & Export | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/indonesian-legal-comparator/blob/main/notebooks/02_LegalComparator_Training_Export.ipynb) |
| 03 — Deploy to HF Space | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/indonesian-legal-comparator/blob/main/notebooks/03_Deploy_HuggingFace_Space.ipynb) |

---

## 💡 Usage Examples

### Similarity Score — Cross-lingual comparison

```python
hasil = model.similarity_score(
    pasal_a_id = "UNCAC_article_15",    # Bribery (English)
    pasal_b_id = "UU_31_1999_pasal_5"  # Suap (Indonesian)
)

print(hasil["score"])         # 0.8734
print(hasil["level"])         # "tinggi"
print(hasil["interpretasi"])  # "Sangat Mirip — Kemungkinan Besar Diadopsi"
print(hasil["preview_a"])     # UNCAC article text (EN)
print(hasil["preview_b"])     # UU pasal text (ID)
```

### Gap Analysis — Full document comparison

```python
hasil = model.compare(doc_a="UNCAC", doc_b="UU_31_1999", top_n_gap=10)

r = hasil.ringkasan
print(f"Adopted  : {r['diadopsi_penuh']} ({r['persen_penuh']}%)")
print(f"Partial  : {r['diadopsi_sebagian']} ({r['persen_sebagian']}%)")
print(f"GAP      : {r['gap_belum_diadopsi']} ({r['persen_gap']}%)")
print(f"Avg sim  : {r['rata_rata_similarity']:.4f}")

for gap in hasil.ringkasan["gap_terbesar"]:
    print(f"{gap['emoji']} {gap['pasal_a']} → {gap['pasal_b']} ({gap['similarity']})")
```

### Semantic Search — Free-form query

```python
# Indonesian query
r = model.search("penggelapan dana publik oleh pejabat negara", top_n=5)

# English query — finds Indonesian articles too (cross-lingual)
r = model.search("illicit enrichment unexplained wealth", top_n=5)

# Filter to a specific document
r = model.search("asset recovery", dokumen="UNCAC", top_n=3)

for h in r.hasil:
    print(f"#{h['rank']} [{h['score']:.3f}] {h['dokumen']} Pasal {h['nomor_pasal']}")
```

### Summarize Gap — Three output formats

```python
hasil = model.compare("UNCAC", "UU_20_2001")

# Narrative — for formal reports
print(model.summarize_gap(hasil, gaya="naratif"))

# Bullet points — for markdown docs
print(model.summarize_gap(hasil, gaya="poin"))

# Table — for presentations
print(model.summarize_gap(hasil, gaya="tabel", top_n_gap=15))
```

### Compare evolution of law across two Indonesian statutes

```python
# How did Tipikor law change between 1999 and 2001?
hasil = model.compare(doc_a="UU_31_1999", doc_b="UU_20_2001")
```

---

## 📁 Project Structure

```
indonesian-legal-comparator/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   └── legal_comparator.py          ← Main model class
│
├── notebooks/
│   ├── 01_Preprocessing_Pipeline.ipynb
│   ├── 02_LegalComparator_Training_Export.ipynb
│   └── 03_Deploy_HuggingFace_Space.ipynb
│
├── demo/
│   ├── app.py                       ← Gradio full dashboard
│   └── requirements_space.txt
│
└── data/
    ├── raw/                         ← PDF files (not tracked in git)
    └── processed/
        ├── corpus_structured.json   ← Hosted on HF Hub
        └── embeddings_labse.npy     ← Hosted on HF Hub
```

> **Note:** Large files are hosted on Hugging Face Hub. Use `snapshot_download()` to fetch them locally.

---

## 📓 Notebooks

### Notebook 01 — Preprocessing Pipeline
**Input:** 7 PDF files → **Output:** `corpus_structured.json` + `embeddings_labse.npy`

| Step | Tool | Output |
|---|---|---|
| PDF Extraction | `pdfplumber` | Raw text per page |
| Text Cleaning | Regex | Normalized text |
| Article Parsing | Regex | Split by `Pasal N` / `Article N` |
| JSON Builder | Python | Structured corpus |
| LaBSE Encoding | `sentence-transformers` | 768-dim vectors |

### Notebook 02 — Model & Export
**Input:** Corpus + embeddings → **Output:** Packaged model ready for Hugging Face Hub

Demonstrates all four features with real examples, then packages and uploads to HF Hub.

### Notebook 03 — Deploy to HF Space
**Input:** All model files → **Output:** Live public URL on Hugging Face Spaces

Fully guided deployment with automatic status monitoring.

---

## 🧠 Model Details

### Embedding: LaBSE

| Property | Value |
|---|---|
| Model | `sentence-transformers/LaBSE` |
| Languages | 109 (including EN and ID) |
| Dimensions | 768 |
| Normalization | L2 (cosine similarity ready) |
| Why chosen | Best cross-lingual sentence similarity for EN ↔ ID |

### Similarity Thresholds

| Score | Status | Interpretation |
|---|---|---|
| ≥ 0.80 | ✅ Fully Adopted | Semantically near-identical — strong legal adoption |
| ≥ 0.65 | 🟡 Partially Adopted | Overlapping topic — partial or loose adoption |
| ≥ 0.45 | 🟠 Loosely Related | Same broad domain, different scope |
| < 0.45 | ❌ GAP | Not adopted — provision absent from Indonesian law |

---

## 📊 Gap Analysis Results

> ⚠️ Results are approximate, based on semantic text similarity — not authoritative legal interpretation. Consult a qualified legal professional for formal analysis.

### Notable findings from UNCAC vs UU Tipikor (UU 31/1999 + UU 20/2001)

**Well-adopted (score ≥ 0.80):**
- Article 15 (Bribery of national officials) → Pasal 5, 11, 12 UU 20/2001
- Article 17 (Embezzlement by public official) → Pasal 8, 9, 10 UU 20/2001
- Article 25 (Obstruction of justice) → Pasal 21 UU 31/1999

**Partially adopted (0.65–0.80):**
- Article 12 (Private sector) → limited scope in Indonesian law
- Article 33 (Protection of reporting persons) → minimal coverage

**Notable gaps (score < 0.65):**
- Article 20 (Illicit enrichment) → no direct equivalent in corpus
- Article 26 (Liability of legal persons) → limited corporate liability
- Chapter V Articles 51–59 (Asset recovery framework) → partially addressed

---

## 🚀 Deploy to Hugging Face

### Push model to HF Hub
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path    = "./model_files",
    repo_id        = "YOUR_USERNAME/indonesian-legal-comparator",
    repo_type      = "model",
    commit_message = "Upload Indonesian Legal Comparator v1.0.0"
)
```

### Create and deploy Gradio Space
```python
api.create_repo(
    repo_id   = "YOUR_USERNAME/indonesian-legal-comparator-demo",
    repo_type = "space",
    space_sdk = "gradio"
)
api.upload_folder(
    folder_path = "./demo",
    repo_id     = "YOUR_USERNAME/indonesian-legal-comparator-demo",
    repo_type   = "space"
)
```

Or run **Notebook 03** for a fully guided, automated deployment.

---

## 🗺️ Roadmap

- [x] PDF preprocessing pipeline
- [x] LaBSE cross-lingual embeddings
- [x] Similarity score (EN ↔ ID)
- [x] Gap analysis with adaptive thresholds
- [x] Semantic search
- [x] Summarization (naratif / poin / tabel)
- [x] Gradio full dashboard
- [x] Hugging Face Hub + Spaces distribution
- [ ] Expand corpus (UU TPPU, UU Pencucian Uang)
- [ ] BERTopic visualization in Gradio tab
- [ ] Export gap report to PDF
- [ ] REST API wrapper (FastAPI)
- [ ] Fine-tuned legal embeddings on Indonesian legal corpus

---

## 🤝 Contributing

Contributions welcome, especially:
- Additional Indonesian law documents to expand the corpus
- Improved parsing for edge-case PDF formats
- Legal expert review of gap analysis findings

Please open an issue first to discuss proposed changes.

---

## 📄 License

MIT License — free to use for research, education, and development.

Data sources are open government documents:
- UNCAC: © United Nations, freely available for public use
- Indonesian laws: Open access via [JDIH Sekretariat Negara](https://jdih.setneg.go.id)

---

## 🙏 Acknowledgements

- [UNODC](https://www.unodc.org) for the UNCAC document
- [JDIH Sekretariat Negara](https://jdih.setneg.go.id) for Indonesian law documents
- [sentence-transformers](https://www.sbert.net) for LaBSE
- [BERTopic](https://maartengr.github.io/BERTopic) for topic modeling
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
- [Gradio](https://gradio.app) for the demo interface

---

<p align="center">
  Built as a portfolio project for <b>AI Trainer — Indonesian Speaker</b><br/>
  Data: 100% open source · Methods: unsupervised · Annotation: zero
</p>
