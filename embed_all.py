import ollama
import numpy as np
import json

def get_embedding(text):
    """
    Converts a text string into a list of numbers (embedding vector).
    Uses nomic-embed-text model running locally via Ollama.
    Returns a list of floats.
    """
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


def cosine_similarity(vector_a, vector_b):
    """
    Measures similarity between two vectors.
    Returns a score between 0 and 1.
    1.0 = identical meaning
    0.0 = completely unrelated
    
    Formula: dot product divided by product of magnitudes
    """
    a = np.array(vector_a)
    b = np.array(vector_b)
    
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    return dot_product / (magnitude_a * magnitude_b)


def embed_all_chunks(chunks):
    """
    Takes your list of chunk dicts from chunking step.
    Adds an 'embedding' field to each chunk.
    Returns updated chunks list.
    """
    total = len(chunks)
    
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{total}...", end="\r")
        chunk["embedding"] = get_embedding(chunk["text"])
    
    print(f"\nAll {total} chunks embedded successfully")
    return chunks


if __name__ == "__main__":
    
    # ─────────────────────────────────────────
    # EXPERIMENT 1: See what an embedding looks like
    # ─────────────────────────────────────────
    print("=" * 60)
    print("EXPERIMENT 1: What does an embedding look like?")
    print("=" * 60)
    
    sample_text = "SF6 gas is used as insulating medium in switchgear"
    embedding = get_embedding(sample_text)
    
    print(f"\nText: '{sample_text}'")
    print(f"Embedding length: {len(embedding)} dimensions")
    print(f"First 10 numbers: {[round(x, 4) for x in embedding[:10]]}")
    print(f"Min value: {round(min(embedding), 4)}")
    print(f"Max value: {round(max(embedding), 4)}")
    
    
    # ─────────────────────────────────────────
    # EXPERIMENT 2: Prove similar text = similar vectors
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Similar text = similar vectors?")
    print("=" * 60)
    
    sentence_1 = "The rapid expansion of the Gas Insulated Switchgear market necessitates a robust revision of our current R&D design validation protocols."
    sentence_2 = "Current trends in high-voltage infrastructure development require that we accelerate the implementation of updated testing standards for GIS equipment."
    sentence_3 = "The intricate pruning techniques required for a healthy Ficus retusa bonsai depend heavily on seasonal light exposure and precise soil moisture management."
    sentence_4 = "A comprehensive analysis of the latest algorithmic trading shifts suggests that volatility in the Nifty 50 index is driving increased reliance on automated hedging strategies."
    
    emb_1 = get_embedding(sentence_1)
    emb_2 = get_embedding(sentence_2)
    emb_3 = get_embedding(sentence_3)
    emb_4 = get_embedding(sentence_4)
    
    sim_1_2 = cosine_similarity(emb_1, emb_2)
    sim_1_3 = cosine_similarity(emb_1, emb_3)
    sim_1_4 = cosine_similarity(emb_1, emb_4)
    
    print(f"\nSentence 1: '{sentence_1}'")
    print(f"Sentence 2: '{sentence_2}'")
    print(f"Sentence 3: '{sentence_3}'")
    print(f"Sentence 4: '{sentence_4}'")
    
    print(f"\nSimilarity 1 vs 2 (same topic):      {round(sim_1_2, 4)}")
    print(f"Similarity 1 vs 3 (unrelated):        {round(sim_1_3, 4)}")
    print(f"Similarity 1 vs 4 (related domain):   {round(sim_1_4, 4)}")
    
    print(f"\nDoes similar text score higher? {'YES' if sim_1_2 > sim_1_3 else 'NO - investigate!'}")
    

    # ─────────────────────────────────────────
    # EXPERIMENT 3: Embed your actual chunks
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Embed your real document chunks")
    print("=" * 60)
    
    with open("resources/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from chunks.json")
    print("Starting embedding... this will take 1-3 minutes")
    
    embedded_chunks = embed_all_chunks(chunks)
    
    # Save embedded chunks - embeddings included
    with open("resources/embedded_chunks.json", "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False)
    
    print(f"Saved to resources/embedded_chunks.json")
    print(f"File contains {len(embedded_chunks)} chunks with embeddings")
    

    # ─────────────────────────────────────────
    # BONUS: Find most similar chunk to a question
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BONUS: Manual retrieval without a vector database")
    print("=" * 60)
    
    question = "What gas is used for insulation in GIS?"
    question_embedding = get_embedding(question)
    
    # Compare question to every chunk manually
    results = []
    for chunk in embedded_chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        results.append((score, chunk["chunk_id"], chunk["section"], chunk["text"][:100]))
    
    # Sort by score, highest first
    results.sort(reverse=True)
    
    print(f"\nQuestion: '{question}'")
    print(f"\nTop 3 most similar chunks:")
    for score, chunk_id, section, preview in results[:3]:
        print(f"\nScore: {round(score, 4)} | Chunk {chunk_id} | Section: {section}")
        print(f"Preview: {preview}...")
        