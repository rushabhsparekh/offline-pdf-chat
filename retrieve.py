from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny
)
from pathlib import Path
import ollama
import json
import uuid

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

COLLECTION_NAME = "apex2_gis"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL       = "qwen2.5:1.5b"
LLM_CTX_WINDOW  = 4096   # num_ctx sent to Ollama — prompt is truncated beyond this
THRESHOLD       = 0.6
TOP_K           = 3
VECTOR_SIZE     = 768
EMBED_BATCH     = 32     # chunks per ollama.embed call


# ─────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────

def get_embeddings(texts):
    """
    Converts a list of texts to embedding vectors in one Ollama call.
    Caller is responsible for the nomic task prefix.
    """
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


def get_embedding(text, is_query=False):
    """
    Converts text to embedding vector via Ollama.
    is_query=True  → adds "search_query: " prefix
    is_query=False → caller is responsible for prefix
                     (index_document adds "search_document: ")
    """
    if is_query:
        text = "search_query: " + text
    return get_embeddings([text])[0]


def point_id_for(source_name, chunk_index):
    """
    Deterministic point ID for a chunk.
    uuid5 is stable across runs — Python's hash() is salted per process.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_name}#{chunk_index}"))


# ─────────────────────────────────────────
# COLLECTION MANAGEMENT
# ─────────────────────────────────────────

def ensure_collection(client):
    """
    Creates Qdrant collection if it does not exist.
    Safe to call multiple times — idempotent.
    Called once at app startup via get_qdrant_client().
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection created: {COLLECTION_NAME}")


def is_indexed(filename, client):
    """True if any vector with source == filename exists in Qdrant."""
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=filename)
            )]
        ),
        limit=1
    )
    return bool(points)


def index_document(filename, client):
    """
    Indexes ONE document's chunks into the existing collection.

    filename: PDF filename exactly as shown in the UI e.g. "gis_basics.pdf"
              (chunk file must exist at chunks/gis_basics_chunks.json)
              Stored as the "source" payload, so filters and removal
              match it exactly — including ".PDF" extensions.

    Skips silently if this document is already indexed.
    Returns number of chunks indexed (or existing count if skipped).

    Why not rebuild_index():
    - build_index() deleted and rebuilt everything on each call
    - That caused the Qdrant duplicate client error on second upload
    - index_document() adds incrementally — one doc at a time
    - get_qdrant_client() in main.py holds the single connection
    """
    pdf_stem   = Path(filename).stem
    chunk_file = Path("chunks") / (pdf_stem + "_chunks.json")
    if not chunk_file.exists():
        print(f"Chunk file not found: {chunk_file}")
        return 0

    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"Empty chunk file: {chunk_file}")
        return 0

    source_name = filename
    if is_indexed(source_name, client):
        print(f"Already indexed: {source_name} — skipping")
        return len(chunks)

    # Build and upload points
    points = []
    print(f"Embedding {len(chunks)} chunks — {source_name}...")

    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        embeddings.extend(get_embeddings(
            ["search_document: " + c["text"] for c in batch]
        ))

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=point_id_for(source_name, i),
            vector=embedding,
            payload={
                "chunk_id": chunk.get("chunk_id", i),
                "text":     chunk["text"],
                "source":   source_name,
                "page":     chunk.get("page", []),
                "headings": chunk.get("headings", []),
                "labels":   chunk.get("labels", []),
            }
        )
        points.append(point)

    # Upload in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + 100]
        )
        print(f"  Uploaded {min(i + 100, len(points))}/{len(points)}")

    print(f"Done — {len(points)} chunks indexed for {source_name}")
    return len(points)


def remove_document(filename, client):
    """
    Removes all Qdrant vectors where source == filename.
    filename: full filename with extension e.g. "gis_basics.pdf"

    Note: chunk JSON file deletion is handled in main.py
    so the UI can confirm before deleting from disk.
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=filename)
            )]
        )
    )
    print(f"Removed from Qdrant: {filename}")


# ─────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────

def search(question, client=None, source_filter=None,
           multi_sources=None, threshold=None, top_k=None):
    """
    Embeds question and searches Qdrant for similar chunks.

    source_filter:  single filename — restricts to one document
    multi_sources:  list of filenames — restricts to selected docs
    threshold:      override default 0.6 (from developer settings)
    top_k:          override default 3 (from developer settings)

    If neither source_filter nor multi_sources is set,
    searches across ALL indexed documents.
    """
    if client is None:
        client = QdrantClient(path="qdrant_storage")

    _threshold = threshold if threshold is not None else THRESHOLD
    _top_k     = top_k     if top_k     is not None else TOP_K

    question_vector = get_embedding(question, is_query=True)

    # Build filter based on scope
    query_filter = None
    if source_filter:
        query_filter = Filter(must=[FieldCondition(
            key="source",
            match=MatchValue(value=source_filter)
        )])
    elif multi_sources:
        query_filter = Filter(must=[FieldCondition(
            key="source",
            match=MatchAny(any=multi_sources)
        )])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        query_filter=query_filter,
        limit=_top_k,
        score_threshold=_threshold
    ).points

    return [{
        "score":    round(r.score, 4),
        "chunk_id": r.payload.get("chunk_id"),
        "text":     r.payload["text"],
        "source":   r.payload.get("source", ""),
        "page":     r.payload.get("page", []),
        "headings": r.payload.get("headings", []),
    } for r in results]


# ─────────────────────────────────────────
# ASK — FULL RAG PIPELINE
# ─────────────────────────────────────────

def ask(question, client, source_filter=None, multi_sources=None,
        chat_history=None, temperature=0.2, threshold=None,
        top_k=None, context_turns=3):
    """
    Full RAG pipeline with sequential chat memory.

    Steps:
    1. Retrieve relevant chunks via search()
    2. Build context string from chunks
    3. Build conversation history from last N turns
    4. Send prompt to LLM
    5. Return answer + sources

    context_turns: how many previous Q&A pairs to include.
                   Controlled by developer settings slider.
                   Kept small to avoid context window overflow.
    """
    retrieved_chunks = search(
        question, client,
        source_filter, multi_sources,
        threshold, top_k
    )

    if not retrieved_chunks:
        return {
            "answer":      "I could not find relevant information in the "
                           "document to answer this question.",
            "sources":     [],
            "chunks_used": 0
        }

    # Build context block from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        headings   = chunk.get("headings", [])
        breadcrumb = " > ".join(headings) if headings else "Document"
        pages      = chunk.get("page", [])
        page_str   = ", ".join(str(p) for p in pages) if pages else "?"

        context += (
            f"\n[Source {i+1}: {breadcrumb} "
            f"| {chunk.get('source', '?')} p.{page_str}]\n"
        )
        context += chunk["text"] + "\n"

    # Build conversation history block — last N turns only
    # Prevents context window overflow on long conversations
    history_text = ""
    if chat_history and context_turns > 0:
        recent = chat_history[-(context_turns * 2):]
        for msg in recent:
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['answer']}\n"

    history_block = (
        f"Previous conversation:\n{history_text}\n"
        if history_text else ""
    )

    prompt = f"""You are a document retrieval assistant.
Your ONLY job is to answer using the document context provided.

{history_block}STRICT RULES:
- Use ONLY the exact information present in the context below
- Do not use any knowledge from your training
- If a specific detail is not in the context, say so explicitly
- You may refer to previous conversation for continuity
- Answer in 3-5 sentences maximum

Context:
{context}
Question: {question}

Answer strictly using only the context above:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "num_predict": 300,
            "num_ctx":     LLM_CTX_WINDOW,
        }
    )

    return {
        "answer":      response["message"]["content"],
        "sources":     [{
            "score":    c["score"],
            "source":   c.get("source", ""),
            "page":     c.get("page", []),
            "headings": c.get("headings", []),
            "text":     c["text"],
        } for c in retrieved_chunks],
        "chunks_used": len(retrieved_chunks)
    }


# ─────────────────────────────────────────
# PRINT HELPERS — terminal testing only
# ─────────────────────────────────────────

def print_full_answer(question, result):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used ({result['chunks_used']} chunks):")
    for i, s in enumerate(result["sources"]):
        headings   = s.get("headings", [])
        breadcrumb = " > ".join(headings) if headings else "—"
        print(f"  {i+1}. {s.get('source','?')} "
              f"p.{s.get('page','?')} — score {s['score']}")
        print(f"     {breadcrumb}")
    print(f"{'='*60}")


# ─────────────────────────────────────────
# MAIN — terminal testing without UI
# ─────────────────────────────────────────

if __name__ == "__main__":
    """
    Terminal test mode.
    Connects to existing qdrant_storage (must already be indexed via main.py).
    Does NOT rebuild index — use main.py UI to index documents.
    """
    print("=" * 60)
    print("APEX2 — Terminal Test Mode")
    print("Connecting to existing Qdrant index...")
    print("=" * 60)

    client = QdrantClient(path="qdrant_storage")
    ensure_collection(client)

    count = client.count(collection_name=COLLECTION_NAME)
    print(f"Points in index: {count.count}")

    if count.count == 0:
        print("\nIndex is empty.")
        print("Run: streamlit run main.py")
        print("Upload and index a PDF first, then re-run this script.")
    else:
        questions = [
            "What insulating medium is used in GIS?",
            "Is SF6 gas harmful to the environment?",
            "What precautions should be taken near switchgear?",
            "What is the history of switchgear development?",
            "How do I make biryani?"
        ]

        for question in questions:
            result = ask(question, client)
            print_full_answer(question, result)