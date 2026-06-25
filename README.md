# APEX2 — Offline Document AI

> Ask questions across your confidential documents. Runs entirely on your laptop. No internet required. No data uploaded anywhere.

Built for engineers, analysts, and researchers who work with documents they cannot share with cloud services.

---

## Demo

*[Add screenshot or screen recording here]*

---

## What It Does

Upload any PDF. Ask questions in plain English. Get answers with exact source references showing which document, which section, and which page the answer came from.

```
You:    "What are the SF6 gas pressure requirements?"
APEX2:  "The SF6 gas pressure shall be maintained at 0.5 MPa..."
        Source: abb_gis_guide.pdf → Components → Page 12
```

Multiple documents can be indexed simultaneously. Search across all of them or restrict to a specific document.

---

## Why Offline Matters

Most document AI tools (ChatPDF, NotebookLM, Adobe AI) upload your files to cloud servers. For engineers working with IEC standards, internal specifications, R&D reports, legal contracts, or financial documents — that is not acceptable.

APEX2 processes everything locally:
- PDF parsing runs on your machine
- Embeddings computed by Ollama locally
- Vectors stored in a local Qdrant database
- LLM inference runs via Ollama
- Zero API calls to any external service

---

## Architecture

```
PDF Upload
    │
    ▼
[Docling — Deep Learning Layout Parser]
    │  Detects columns, tables, headings, captions
    │  Extracts tables as Markdown
    │  Preserves section hierarchy as breadcrumbs
    ▼
[Chunk JSON] ──→ saved to chunks/ folder (cached)
    │
    ▼
[nomic-embed-text via Ollama]
    │  "search_document: " prefix applied per chunk
    │  768-dimension dense vectors
    ▼
[Qdrant Vector Database] ──→ stored in qdrant_storage/
    │  Single collection, filtered by source filename
    │  Cosine similarity, threshold 0.6
    ▼
[User Question]
    │
    ▼
[nomic-embed-text] — "search_query: " prefix
    │
    ▼
[Qdrant Search] — returns top-K chunks above threshold
    │
    ▼
[qwen2.5:1.5b via Ollama] — answers using only retrieved context
    │  Strict prompt prevents hallucination
    │  Sequential chat memory — last N turns included
    ▼
[Answer + Source Provenance]
```

---

## Key Engineering Decisions

### Why Docling over pdfplumber

The first version of APEX used a hand-built statistical chunker based on pdfplumber. By analyzing line gap distributions, font size clustering, and x-coordinate histograms across 20 test documents, the statistical approach hit a fundamental ceiling — thresholds derived from one document's layout failed on others.

Docling uses a deep learning layout detection model trained on millions of documents. It detects the same signals (gaps, font sizes, column positions) but learned from data rather than hardcoded rules. Switching to Docling gave correct chunking on all 20 test documents including double-column academic papers and 135-page IEC standards — without any per-document tuning.

The statistical work was not wasted. Understanding *why* chunking is hard made it clear *what* Docling solves.

### Why nomic-embed-text with task prefixes

Two embedding models were compared on domain-specific sentences:

|  | nomic-embed-text | qwen3-0.6b |
|--|--|--|
| Same topic, different framing | 0.675 | 0.646 |
| Completely unrelated | 0.399 | 0.286 |
| Same person, different domain | 0.530 | 0.263 |

nomic captures semantic intent more broadly — it found similarity between GIS engineering and trading sentences (0.53) where qwen treated them as unrelated (0.26). For a general Q&A system where users phrase questions in natural language, intent-matching outperforms strict domain vocabulary matching.

nomic-embed-text v1.5 requires task-specific prefixes: `"search_document: "` for indexing and `"search_query: "` for retrieval. Without these prefixes, retrieval quality degrades measurably. This is enforced throughout the pipeline.

### Why cosine similarity at threshold 0.6

A direct comparison of cosine vs L2 distance was run on the same document corpus:

- **L2 outperformed cosine** on procedural and fault-condition questions — physical proximity in vector space captures step-by-step similarity better
- **Cosine outperformed L2** on conceptual questions — directional similarity captures intent without being misled by vector magnitude

Threshold 0.6 was derived from observed score distributions: clearly relevant chunks scored ~0.67, weakly relevant ~0.53, unrelated ~0.40. Setting threshold at 0.6 correctly rejected off-topic queries while retrieving relevant content.

### Single Qdrant collection with source filtering

Each document's chunks are stored in one shared Qdrant collection with a `source` field in the payload. This enables:
- Search across all documents simultaneously
- Filter to a single document by source filename
- Filter to a selected subset using `MatchAny`
- Remove one document without rebuilding the entire index

Alternative (one collection per document) was rejected because it requires a new connection per search operation, which conflicts with Qdrant local mode's single-connection constraint.

### Incremental indexing — no full rebuilds

`index_document()` checks if a source already exists in Qdrant before embedding. If found, it skips. This means:
- Second upload of same document is instant
- Adding a new document does not re-embed existing ones
- App startup does not require rebuilding the index

---

## Document Corpus Tested

The chunking pipeline was validated on 20 documents across 5 categories:

| Category | Layout types | Notes |
|--|--|--|
| Academic papers (arxiv) | Single column | Tables, equations |
| Wikipedia articles | Single column, PPT-style | Images, varied length |
| Technical standards | Single column | 135-page IEC standard included |
| Legal documents | Single column | Numbered hierarchy (1.a.i) |
| Financial reports | Mixed, multi-column | Annual reports, balance sheets |

**Known limitations:**
- Mathematical equations extract incompletely — PDF equation rendering loses structure
- Scanned PDFs (images of text) are not supported — digital PDFs only
- Page numbers shown are PDF-internal, not printed page numbers
- Processing speed: ~5 seconds per page on CPU

---

## Features

**Core:**
- Upload and index any PDF
- Ask questions with source provenance (document → section → page)
- Sequential chat memory — references previous answers in conversation
- Multi-document search or restrict to selected documents

**Document management:**
- Document library showing all indexed PDFs
- Remove individual documents without rebuilding index
- Caching — previously processed PDFs skip re-extraction

**Chat:**
- Save and load chat history
- New chat button with automatic save
- Context window usage indicator

**Developer settings:**
- Temperature (0.0 — 1.0)
- Retrieval threshold (0.3 — 0.9)
- Chunks to retrieve (top-k, 1 — 8)
- Chat history turns to include (0 — 6)

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- 8GB RAM minimum (16GB recommended for large documents)
- AMD or NVIDIA GPU optional (CPU works, GPU faster via Vulkan/CUDA)

---

## Installation

```bash
# Clone repository
git clone https://github.com/rushabhsparekh/offline-pdf-chat.git
cd offline-pdf-chat

# Install dependencies
pip install -r requirements.txt

# One-time tokenizer cache
python tokenizer.py

# Pull required Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

---

## Running

```bash
# Terminal 1 — keep open
ollama serve

# Terminal 2
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

**AMD GPU acceleration (RX 560X and similar):**
```bash
# Set before running ollama serve
set OLLAMA_VULKAN=1
ollama serve
```

---

## Project Structure

```
offline-pdf-chat/
├── main.py          # Streamlit UI — all user interaction
├── retrieve.py      # Qdrant operations, search, RAG pipeline
├── pdf_to_json.py   # Docling PDF chunker
├── tokenizer.py     # One-time nomic tokenizer cache setup
├── requirements.txt
├── chunks/          # Generated chunk JSON files (gitignored)
├── pdfs/            # Uploaded PDFs (gitignored)
├── models/          # Cached tokenizer (gitignored)
└── qdrant_storage/  # Vector index (gitignored)
```

---

## Known Limitations

| Limitation | Impact | Status |
|--|--|--|
| Mathematical equations | Incomplete formula extraction | Known, no fix planned |
| Scanned PDFs | Not supported | Out of scope |
| PDF page numbers | Internal numbering only | Documented |
| qwen2.5:1.5b knowledge bleed | Occasional hallucination on weak prompts | Mitigated via strict prompt |
| Single machine only | No multi-user support | Out of scope |

---

## What I Learned Building This

This project was built from scratch over 3 weeks as a structured learning exercise in applied LLM engineering. Every architectural decision was derived from experiments, not tutorials.

Key discoveries made through observation before reading about them:

- **Semantic chunking problem** — fixed-size chunking loses paragraph context. Discovered by reading extracted text and noticing meaning boundaries don't align with character counts.
- **Knowledge bleed** — small LLMs mix training knowledge into RAG answers. Caught because domain expertise (SF6 GWP is 23,000x CO2, not 25x) flagged the hallucination.
- **Cosine vs L2 tradeoffs** — discovered through controlled retrieval experiments that different question types benefit from different distance metrics.
- **PDF parsing order** — pypdf reads in storage order not visual order, causing sequence corruption. Diagnosed by saving extracted text to file and reading it.
- **nomic task prefix requirement** — embedding quality difference with and without prefixes measured directly.

---

## Tech Stack

| Component | Technology |
|--|--|
| PDF parsing | Docling (IBM, deep learning layout detection) |
| Embeddings | nomic-embed-text via Ollama |
| Vector database | Qdrant (local mode) |
| LLM | qwen2.5:1.5b via Ollama |
| UI | Streamlit |
| Language | Python 3.11 |
