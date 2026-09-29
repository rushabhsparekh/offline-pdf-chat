import hashlib
import json

import pytest
from qdrant_client import QdrantClient

import retrieve


def fake_vector(text):
    """Deterministic 768-d vector; texts sharing a topic word stay close."""
    topic = text.split(": ", 1)[-1].split()[0].lower()
    seed  = hashlib.sha256(topic.encode()).digest()
    return [((seed[i % 32] + i) % 17) / 17 + 0.01 for i in range(retrieve.VECTOR_SIZE)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_embed(model, input):
        calls.append(list(input))
        return {"embeddings": [fake_vector(t) for t in input]}

    monkeypatch.setattr(retrieve.ollama, "embed", fake_embed)
    c = QdrantClient(":memory:")
    retrieve.ensure_collection(c)
    c.embed_calls = calls
    return c


def write_chunks(stem, texts):
    chunks = [
        {"chunk_id": i, "text": t, "page": [i + 1], "headings": ["H"], "labels": ["text"]}
        for i, t in enumerate(texts)
    ]
    path = retrieve.Path("chunks") / f"{stem}_chunks.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(chunks), encoding="utf-8")


def count(client, source=None):
    flt = None
    if source:
        flt = retrieve.Filter(must=[retrieve.FieldCondition(
            key="source", match=retrieve.MatchValue(value=source))])
    return client.count(retrieve.COLLECTION_NAME, count_filter=flt).count


def test_point_ids_are_deterministic_and_distinct():
    assert retrieve.point_id_for("a.pdf", 0) == retrieve.point_id_for("a.pdf", 0)
    assert retrieve.point_id_for("a.pdf", 0) != retrieve.point_id_for("b.pdf", 0)
    assert retrieve.point_id_for("a.pdf", 0) != retrieve.point_id_for("a.pdf", 1)


def test_index_uses_exact_filename_as_source(client):
    write_chunks("Report", ["gas pressure", "breaker rating"])
    assert retrieve.index_document("Report.PDF", client) == 2
    assert count(client, "Report.PDF") == 2

    retrieve.remove_document("Report.PDF", client)
    assert count(client) == 0


def test_index_batches_embeddings_with_document_prefix(client, monkeypatch):
    monkeypatch.setattr(retrieve, "EMBED_BATCH", 2)
    write_chunks("doc", ["gas a", "gas b", "gas c"])
    retrieve.index_document("doc.pdf", client)

    assert [len(c) for c in client.embed_calls] == [2, 1]
    assert all(t.startswith("search_document: ") for c in client.embed_calls for t in c)


def test_reindex_is_skipped(client):
    write_chunks("doc", ["gas a"])
    retrieve.index_document("doc.pdf", client)
    retrieve.index_document("doc.pdf", client)
    assert count(client) == 1
    assert len(client.embed_calls) == 1


def test_search_filters_by_source_and_returns_payload_chunk_id(client):
    write_chunks("one", ["gas one", "breaker one"])
    write_chunks("two", ["gas two"])
    retrieve.index_document("one.pdf", client)
    retrieve.index_document("two.pdf", client)

    hits = retrieve.search("gas", client, source_filter="two.pdf", threshold=0.0)
    assert [h["source"] for h in hits] == ["two.pdf"]
    assert hits[0]["chunk_id"] == 0

    hits = retrieve.search("gas", client, multi_sources=["one.pdf", "two.pdf"],
                           threshold=0.0, top_k=10)
    assert {h["source"] for h in hits} == {"one.pdf", "two.pdf"}


def test_ask_sends_context_history_and_num_ctx(client, monkeypatch):
    write_chunks("doc", ["gas pressure is 0.5 MPa"])
    retrieve.index_document("doc.pdf", client)

    sent = {}

    def fake_chat(model, messages, options):
        sent.update(model=model, prompt=messages[0]["content"], options=options)
        return {"message": {"content": "0.5 MPa"}}

    monkeypatch.setattr(retrieve.ollama, "chat", fake_chat)
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "answer": "old answer"},
    ]
    result = retrieve.ask("gas pressure?", client, chat_history=history,
                          threshold=0.0, context_turns=1)

    assert result["answer"] == "0.5 MPa"
    assert result["chunks_used"] == 1
    assert sent["model"] == retrieve.LLM_MODEL
    assert sent["options"]["num_ctx"] == retrieve.LLM_CTX_WINDOW
    assert "gas pressure is 0.5 MPa" in sent["prompt"]
    assert "Assistant: old answer" in sent["prompt"]


def test_ask_without_matches_skips_llm(client, monkeypatch):
    monkeypatch.setattr(retrieve.ollama, "chat",
                        lambda **_: pytest.fail("LLM should not be called"))
    result = retrieve.ask("anything", client)
    assert result["chunks_used"] == 0
    assert result["sources"] == []
