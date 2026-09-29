import streamlit as st
from pathlib import Path
from qdrant_client import QdrantClient
import hashlib
import json
import sys
import os
import datetime

sys.path.append(os.path.dirname(__file__))
from pdf_to_json import build_converter, build_chunker, pdf_to_chunks, MAX_TOKENS
from retrieve import (
    ensure_collection, index_document, remove_document, is_indexed,
    ask, LLM_CTX_WINDOW
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

REGISTRY_PATH  = Path("doc_registry.json")
CHUNKS_DIR     = Path("chunks")
PDFS_DIR       = Path("pdfs")
CHAT_LOGS_DIR  = Path("chat_logs")


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="APEX2 — Offline Document AI",
    page_icon="🔬",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES — loaded once, never reloaded
# @st.cache_resource persists across ALL reruns for this session
# Solves: fan noise on every rerun, duplicate Qdrant client error
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_docling_models():
    """
    Loads Docling converter and chunker exactly once per session.
    Heavy ML models — 15-30 seconds first time, instant after.
    """
    return build_converter(), build_chunker()


@st.cache_resource
def get_qdrant_client():
    """
    Single Qdrant client for entire app session.
    Solves: 'Storage folder already accessed' error on multi-PDF upload.
    Only one connection to qdrant_storage/ ever exists.
    """
    PDFS_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)
    CHAT_LOGS_DIR.mkdir(exist_ok=True)

    client = QdrantClient(path="qdrant_storage")
    ensure_collection(client)
    return client


# ─────────────────────────────────────────────────────────────
# REGISTRY HELPERS
# Tracks which PDFs are indexed across sessions
# ─────────────────────────────────────────────────────────────

def load_registry() -> dict:
    """Loads doc_registry.json. Returns empty registry if not found."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def add_to_registry(filename: str, chunk_count: int):
    registry = load_registry()
    existing = [d["filename"] for d in registry["documents"]]
    if filename not in existing:
        registry["documents"].append({
            "filename":    filename,
            "indexed_at":  datetime.datetime.now().isoformat(timespec="seconds"),
            "chunk_count": chunk_count,
            "status":      "indexed"
        })
        save_registry(registry)


def remove_from_registry(filename: str):
    registry = load_registry()
    registry["documents"] = [
        d for d in registry["documents"]
        if d["filename"] != filename
    ]
    save_registry(registry)


def get_indexed_docs() -> list[dict]:
    return load_registry().get("documents", [])


def sync_registry_with_index(client):
    """
    Drops registry entries whose vectors are no longer in Qdrant
    (e.g. qdrant_storage/ was deleted, or indexing crashed midway).
    Qdrant is the source of truth for what is searchable.
    """
    registry = load_registry()
    docs     = registry.get("documents", [])
    kept     = [d for d in docs if is_indexed(d["filename"], client)]
    if len(kept) != len(docs):
        registry["documents"] = kept
        save_registry(registry)


def delete_document(filename: str, client):
    """Removes a document's vectors, registry entry and cached chunks."""
    remove_document(filename, client)
    remove_from_registry(filename)
    cp = CHUNKS_DIR / (Path(filename).stem + "_chunks.json")
    if cp.exists():
        cp.unlink()


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────
# CHAT LOG HELPERS
# ─────────────────────────────────────────────────────────────

def save_chat_log(chat_history: list, name: str = ""):
    CHAT_LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label     = f"_{name}" if name else ""
    log_path  = CHAT_LOGS_DIR / f"chat{label}_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2, ensure_ascii=False)
    return log_path


def load_chat_logs() -> list[Path]:
    if not CHAT_LOGS_DIR.exists():
        return []
    return sorted(CHAT_LOGS_DIR.glob("chat_*.json"), reverse=True)


# ─────────────────────────────────────────────────────────────
# TOKEN COUNTER (approximate)
# 1 token ≈ 4 characters — fast, no tokenizer needed
# ─────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_context_usage(chat_history: list, context_turns: int,
                           top_k: int) -> tuple[int, int]:
    """
    Estimates prompt tokens for the next question, using the same
    history window and top-k that ask() will use.
    Chunks are counted at their worst-case size (MAX_TOKENS each).
    Returns (used_tokens, total_available) — used may exceed total.
    """
    recent        = chat_history[-(context_turns * 2):] if context_turns > 0 else []
    history_toks  = sum(
        approx_tokens(m.get("content", "") + m.get("answer", ""))
        for m in recent
    )
    prompt_overhead = 350               # system prompt + formatting
    chunk_reserve   = top_k * MAX_TOKENS
    used            = history_toks + prompt_overhead + chunk_reserve
    return used, LLM_CTX_WINDOW


# ─────────────────────────────────────────────────────────────
# SOURCES DISPLAY
# ─────────────────────────────────────────────────────────────

def render_sources(sources: list):
    with st.expander(f"📚 Sources — {len(sources)} chunks used"):
        for i, src in enumerate(sources):
            headings   = src.get("headings", [])
            breadcrumb = " > ".join(headings) if headings else "—"
            pages      = src.get("page", [])
            page_str   = ", ".join(str(p) for p in pages) if pages else "?"

            st.markdown(
                f"**{i+1}.** `{src.get('source','?')}` "
                f"— p.{page_str} "
                f"— score `{src['score']}`"
            )
            st.caption(f"📍 {breadcrumb}")


# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────

_defaults = {
    "chat_history":   [],
    "current_doc":    "",       # last indexed doc
    "scope":          "All documents",
    "selected_docs":  [],       # for multi-select scope
    "chat_name":      "",       # optional label for saving
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────

with st.spinner("Loading document AI models — one time only..."):
    converter, chunker = load_docling_models()

client = get_qdrant_client()

if "registry_synced" not in st.session_state:
    sync_registry_with_index(client)
    st.session_state.registry_synced = True

indexed_docs = get_indexed_docs()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("🔬 APEX2")
    st.caption("Offline Document AI")
    st.markdown("---")

    # ── Upload & Index ────────────────────────────────────────
    st.markdown("### Add Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Processed entirely on your machine — nothing leaves your laptop"
    )

    if uploaded_file:
        filename      = Path(uploaded_file.name).name   # strip any path parts
        pdf_path      = PDFS_DIR / filename
        chunk_path    = CHUNKS_DIR / (pdf_path.stem + "_chunks.json")
        pdf_bytes     = uploaded_file.getvalue()

        # Same name on disk but different content → cached chunks are stale
        on_disk_same  = (
            pdf_path.exists()
            and file_sha256(pdf_path.read_bytes()) == file_sha256(pdf_bytes)
        )
        replacing     = pdf_path.exists() and not on_disk_same
        already_done  = chunk_path.exists() and on_disk_same

        if already_done:
            st.info("Already processed — click Index to add to search.")
        elif replacing:
            st.warning(
                f"A different file named {filename} is already in the "
                "library — indexing will replace it."
            )
        else:
            st.success(f"Uploaded: {filename}")

        if st.button("⚡ Index Document", use_container_width=True):

            # Step 1 — Chunk (skip if cached)
            if not already_done:
                # Clear any stale vectors/chunks stored under this name
                delete_document(filename, client)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                with st.spinner("Processing PDF with Docling..."):
                    chunks = pdf_to_chunks(
                        str(pdf_path), converter, chunker
                    )
                st.success(f"✅ {len(chunks)} chunks extracted")
            else:
                with open(chunk_path, encoding="utf-8") as f:
                    chunks = json.load(f)
                st.success(f"✅ {len(chunks)} chunks loaded from cache")

            # Step 2 — Embed & index this document only
            with st.spinner("Embedding and indexing..."):
                count = index_document(filename, client)

            add_to_registry(filename, count)
            st.session_state.current_doc = filename

            # Refresh indexed docs list
            indexed_docs = get_indexed_docs()
            st.success("✅ Ready — ask your first question below")
            st.rerun()

    st.markdown("---")

    # ── Document Library ──────────────────────────────────────
    st.markdown("### Document Library")

    if not indexed_docs:
        st.caption("No documents indexed yet.")
    else:
        for doc in indexed_docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(
                    f"📄 **{doc['filename']}**\n\n"
                    f"{doc['chunk_count']} chunks · {doc['indexed_at'][:10]}"
                )
            with col2:
                if st.button("✕", key=f"remove_{doc['filename']}",
                             help=f"Remove {doc['filename']}"):
                    with st.spinner(f"Removing {doc['filename']}..."):
                        delete_document(doc["filename"], client)

                    st.success(f"Removed {doc['filename']}")
                    indexed_docs = get_indexed_docs()
                    st.rerun()

    st.markdown("---")

    # ── Search Scope ──────────────────────────────────────────
    st.markdown("### Search Scope")

    doc_names     = [d["filename"] for d in indexed_docs]
    has_docs      = len(doc_names) > 0

    scope_options = ["All documents", "Current document", "Select documents"]
    st.session_state.scope = st.radio(
        "Search in",
        scope_options,
        disabled=not has_docs,
        label_visibility="collapsed"
    )

    if st.session_state.scope == "Select documents" and has_docs:
        st.session_state.selected_docs = st.multiselect(
            "Choose documents",
            doc_names,
            default=st.session_state.selected_docs or doc_names[:1]
        )

    st.markdown("---")

    # ── Context Window Indicator ──────────────────────────────
    # Slider values live in session_state (keys below) — read them here
    # so the estimate matches the settings the next question will use
    used, total = estimate_context_usage(
        st.session_state.chat_history,
        st.session_state.get("context_turns", 3),
        st.session_state.get("top_k", 3),
    )
    pct         = int((used / total) * 100)

    st.markdown("### Context Window")
    st.progress(
        min(pct, 100) / 100,
        text=f"~{used} / {total} tokens ({pct}%)"
    )
    if pct > 100:
        st.error(
            "Prompt may exceed the model's context window and be truncated. "
            "Lower top-k or history turns, or start a new chat."
        )
    elif pct >= 80:
        st.warning("Context nearly full — start a new chat to reset.")

    st.markdown("---")

    # ── Chat Controls ─────────────────────────────────────────
    st.markdown("### Chat")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 New Chat", use_container_width=True,
                     disabled=not st.session_state.chat_history):
            if st.session_state.chat_history:
                save_chat_log(
                    st.session_state.chat_history,
                    st.session_state.chat_name
                )
            st.session_state.chat_history = []
            st.rerun()

    with col_b:
        if st.button("💾 Save Chat", use_container_width=True,
                     disabled=not st.session_state.chat_history):
            path = save_chat_log(
                st.session_state.chat_history,
                st.session_state.chat_name
            )
            st.success(f"Saved: {path.name}")

    # Chat history browser
    chat_logs = load_chat_logs()
    if chat_logs:
        st.markdown("**Saved chats:**")
        selected_log = st.selectbox(
            "Load a previous chat",
            ["— select —"] + [p.name for p in chat_logs],
            label_visibility="collapsed"
        )
        if selected_log and selected_log != "— select —":
            log_path = CHAT_LOGS_DIR / selected_log
            with open(log_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if st.button("📂 Load this chat", use_container_width=True):
                st.session_state.chat_history = loaded
                st.rerun()

    st.markdown("---")

    # ── Developer Settings (collapsible) ─────────────────────
    with st.expander("⚙️ Developer Settings"):
        st.caption("These settings affect answer quality and speed.")

        temperature = st.slider(
            "Temperature",
            key="temperature",
            min_value=0.0, max_value=1.0,
            value=0.2, step=0.05,
            help="Higher = more creative. Lower = more consistent."
        )
        threshold = st.slider(
            "Retrieval threshold",
            key="threshold",
            min_value=0.3, max_value=0.9,
            value=0.6, step=0.05,
            help="Minimum similarity score to include a chunk."
        )
        top_k = st.slider(
            "Chunks to retrieve (top-k)",
            key="top_k",
            min_value=1, max_value=8,
            value=3, step=1,
            help="More chunks = more context but slower."
        )
        context_turns = st.slider(
            "Chat history turns to include",
            key="context_turns",
            min_value=0, max_value=6,
            value=3, step=1,
            help="How many previous Q&A pairs to send to the LLM."
        )

    st.markdown("---")
    st.caption("🔒 Nothing leaves your laptop.")
    st.caption("Runs on Ollama + Qdrant, fully offline.")


# ─────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────

st.markdown("## Ask your documents")

if not indexed_docs:
    st.info(
        "👈 Upload a PDF in the sidebar to get started. "
        "All processing happens on your machine."
    )
else:
    active_scope = st.session_state.scope
    current_doc  = st.session_state.current_doc

    # Scope label shown above chat
    if active_scope == "All documents":
        scope_label = f"Searching across **{len(indexed_docs)} documents**"
    elif active_scope == "Current document":
        scope_label = f"Searching in **{current_doc or 'last indexed doc'}**"
    else:
        n = len(st.session_state.selected_docs)
        scope_label = f"Searching in **{n} selected documents**"
    st.caption(scope_label)

# ── Render chat history ───────────────────────────────────────

for message in st.session_state.chat_history:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])

    else:
        with st.chat_message("assistant"):
            st.write(message["answer"])

            sources = message.get("sources", [])
            if sources:
                render_sources(sources)

# ── Chat input ────────────────────────────────────────────────

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not indexed_docs
)

if question:

    # Resolve source filter from scope
    source_filter  = None
    multi_sources  = None

    if active_scope == "Current document" and current_doc:
        source_filter = current_doc
    elif active_scope == "Select documents":
        multi_sources = st.session_state.selected_docs or None

    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching and reasoning..."):

            # Pass developer settings into ask()
            result = ask(
                question=question,
                client=client,
                source_filter=source_filter,
                multi_sources=multi_sources,
                chat_history=st.session_state.chat_history,
                temperature=temperature,
                threshold=threshold,
                top_k=top_k,
                context_turns=context_turns,
            )

        st.write(result["answer"])

        sources = result.get("sources", [])
        if sources:
            render_sources(sources)
        else:
            st.warning(
                "No relevant content found above the confidence threshold. "
                "Try rephrasing, lowering the threshold in Developer Settings, "
                "or check if the document covers this topic."
            )

    # Save to session history
    st.session_state.chat_history.append({
        "role":    "user",
        "content": question
    })
    st.session_state.chat_history.append({
        "role":    "assistant",
        "answer":  result["answer"],
        "sources": result.get("sources", [])
    })