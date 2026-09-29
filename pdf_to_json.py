"""
APEX2 — Universal PDF Chunker
Converts PDF → clean chunk JSON ready for nomic-embed-text + Qdrant.

Usage:
    python pdf_to_json.py                        # chunks all PDFs in ./pdfs/
    python pdf_to_json.py path/to/file.pdf       # single file
    python pdf_to_json.py path/to/dir/           # all PDFs in a directory

Output:
    ./chunks/<filename>_chunks.json  per PDF

Embedding (run after this):
    Each chunk["text"] should be prefixed with "search_document: " before
    passing to ollama.embeddings(model="nomic-embed-text", prompt=...)
    Each query should be prefixed with "search_query: " at retrieval time.

One-time setup (run once to cache tokenizer locally, then never again):
    python tokenizer.py

Docling and transformers are imported inside build_converter() /
build_chunker() so the text helpers below can be imported (and tested)
without loading the ML stack.
"""

import json
import re
import sys
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────

TOKENIZER_PATH  = "./models/nomic-tokenizer"  # local cache (see one-time setup above)
MAX_TOKENS      = 512
MIN_CHUNK_CHARS = 100
OUTPUT_DIR      = "./chunks"

NOISE_HEADINGS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "index",
    "foreword",
    "preface",
    "table of contents",
    "contents",
    "notational conventions",
}
NOISE_PATTERN   = re.compile(r'<!--.*?-->', re.DOTALL)
SECTION_NUMBER  = re.compile(r'^(?:\d+(?:\.\d+)*[.)]?|[ivxlc]+[.)])\s+', re.IGNORECASE)


# ── Build converter (call once, reuse across files) ───────────────────────────

def build_converter():
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr                  = False                    # digital PDFs only — biggest speedup
    opts.do_table_structure      = True                     # keep — GIS specs need tables
    opts.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.FAST,
        do_cell_matching=False,
    )
    opts.generate_page_images    = False
    opts.generate_picture_images = False
    opts.do_picture_classification = False
    opts.do_picture_description    = False
    opts.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.AUTO,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


# ── Build chunker (call once, reuse across files) ─────────────────────────────

def build_chunker():
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
    from transformers import AutoTokenizer

    # Serializer: Markdown tables, plain text for everything else
    class MDTableProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(compact_tables=True),
            )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH),
        max_tokens=MAX_TOKENS,
    )
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MDTableProvider(),
    )



# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = NOISE_PATTERN.sub("", text)
    text = re.sub(r'\r\n', '\n', text)        # normalize line endings
    text = re.sub(r'\n{3,}', '\n\n', text)    # max two consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces/tabs not newlines
    text = re.sub(r'\[\d+\]', '', text)       # citation markers like [12]
    return text.strip()



NOISE_LABELS = {"page_header", "page_footer", "picture"}

def normalize_heading(heading: str) -> str:
    """ "7. References" / "IV References" → "references" """
    return SECTION_NUMBER.sub("", heading.strip()).strip().lower()


def is_noise_chunk(chunk: dict) -> bool:
    headings = [normalize_heading(h) for h in chunk["headings"]]

    # Chunk sits under a references / contents / index style section
    if any(h in NOISE_HEADINGS for h in headings):
        return True

    # Too short
    if len(chunk["text"].strip()) < MIN_CHUNK_CHARS:
        return True

    # Only noise items from Docling — a chunk that merely contains a
    # picture alongside real text is kept
    labels = chunk["labels"]
    if labels and all(label in NOISE_LABELS for label in labels):
        return True

    return False


# ── Core: PDF → chunk JSON ────────────────────────────────────────────────────

def pdf_to_chunks(
    pdf_path: str,
    converter,
    chunker,
    output_dir: str = OUTPUT_DIR,
) -> list[dict]:

    print(f"Processing: {pdf_path}")
    result = converter.convert(pdf_path)
    doc    = result.document

    chunks = []
    for chunk in chunker.chunk(dl_doc=doc):
        raw_text = chunker.contextualize(chunk)
        cleaned  = clean_text(raw_text)
        if not cleaned:
            continue

        meta = chunk.meta

        page_nums = sorted({
            prov.page_no
            for item in (meta.doc_items or [])
            for prov in item.prov
        })

        labels = list({
            item.label.value
            for item in (meta.doc_items or [])
        })

        headings = [
            re.sub(r'\r\n', ' ', h).strip()
            for h in (meta.headings or [])
        ]

        captions = [
            item.text
            for item in (meta.doc_items or [])
            if hasattr(item, "label")
            and item.label.value == "caption"
            and hasattr(item, "text")
        ]

        record = {
            "chunk_id": len(chunks),  # auto-increment based on position
            "text":     cleaned,
            "source":   str(pdf_path),
            "page":     page_nums,
            "headings": headings,
            "captions": captions,
            "labels":   labels,
        }

        if not is_noise_chunk(record):
            chunks.append(record)

    out_path = Path(output_dir) / (Path(pdf_path).stem + "_chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"  → {len(chunks)} chunks saved to {out_path}")
    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    converter = build_converter()
    chunker   = build_chunker()

    # Resolve input paths from CLI args or default to ./pdfs/
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            pdf_files = sorted(target.glob("*.pdf"))
        elif target.suffix.lower() == ".pdf":
            pdf_files = [target]
        else:
            print(f"Error: {target} is not a PDF or directory.")
            sys.exit(1)
    else:
        pdf_files = sorted(Path("./pdfs").glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    total_chunks = 0
    for pdf in pdf_files:
        chunks      = pdf_to_chunks(str(pdf), converter, chunker)
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} total chunks across {len(pdf_files)} file(s).")
    print(f"Chunk JSONs saved to: {Path(OUTPUT_DIR).resolve()}")


def preview_chunks(json_path, how_many=5):
    """
    Human-readable preview of chunk JSON.
    Run this separately to verify quality.
    """
    with open(json_path, encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"\nTotal chunks: {len(chunks)}")
    print("=" * 60)
    
    for chunk in chunks[:how_many]:
        print(f"\nChunk {chunk['chunk_id']}")
        print(f"Pages:    {chunk['page']}")
        print(f"Headings: {' > '.join(chunk['headings'])}")
        print(f"Labels:   {chunk['labels']}")
        print(f"Length:   {len(chunk['text'])} chars")
        print(f"Text:\n{chunk['text'][:400]}")
        print("-" * 60)

if __name__ == "__main__":
    main()