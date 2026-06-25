import pypdf
import os
def extract_text_from_pdf(pdf_path):
    """
    Opens aPDF and extracts all text page by page .
    returns a list of tupples: (page_number, page_text)
    """
    pages = []

    reader = pypdf.PdfReader(pdf_path)

    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()

        if text and len(text.strip()) > 0:
            pages.append((page_num + 1, text))

    print(f"Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages

def chunk_text(pages, chunk_size=500, overlap=50):
    """
    Takes lsit of (page_num, text) tuples.
    Split each page's text into overlapping chunk.
    Returns list of dicts with chunk text, page number, chunk index
    """
    all_chunks = []
    chunk_index = 0
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) > 50:
                all_chunks.append({
                    "chunk_id": chunk_index,
                    "page_num": page_num,
                    "text": chunk.strip()
                })
                chunk_index += 1
            start = start + chunk_size - overlap
    print(f"Created {len(all_chunks)} chunks total")
    return all_chunks

def preview_chunks(chunks, how_many=3):
    """
    prints first N chunks so you can see what they look like
    """
    print("\n--- CHUNK PREVIEW ---")
    for i in range(min(how_many, len(chunks))):
        print(f"\nChunk {i} | Page {chunks[i]['page_num']}")
        print(f"Length: {len(chunks[i]['text'])} characters")
        print(f"Text: {chunks[i]['text']}")
        print("-"*40)

def save_text_to_file(pages, output_path="resources/extracted_text.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for page_num, text in pages:
            f.write(f"\n{'='*50}\n")
            f.write(f"PAGE {page_num}\n")
            f.write(f"{'='*50}\n")
            f.write(text)
    print(f"Text saved to {output_path}")

def chunk_by_paragraph(pages, min_length=100, max_length=1000):
    """
    Splits text by paragraphs instead of fixed character count.
    Each paragraph becomes one chunk.
    If paragraph is too long, splits it further.
    If paragraph is too short, merges with next one.
    """
    all_chunks = []
    chunk_index = 0

    for page_num, text in pages:

        # Split by double newline = paragraph boundary
        paragraphs = text.split('\n\n')

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()

            # Skip empty paragraphs
            if len(para) < 20:
                continue

            # If adding this paragraph keeps us under max_length, merge it
            if len(current_chunk) + len(para) < max_length:
                current_chunk = current_chunk + " " + para

            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) > min_length:
                    all_chunks.append({
                        "chunk_id": chunk_index,
                        "page_num": page_num,
                        "text": current_chunk.strip()
                    })
                    chunk_index += 1

                # Start fresh with current paragraph
                current_chunk = para

        # Don't forget the last chunk on each page
        if len(current_chunk) > min_length:
            all_chunks.append({
                "chunk_id": chunk_index,
                "page_num": page_num,
                "text": current_chunk.strip()
            })
            chunk_index += 1

    print(f"Paragraph chunking created {len(all_chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    pdf_path = "resources/Switchgear.pdf"
    pages = extract_text_from_pdf(pdf_path)
    save_text_to_file(pages)

    print("\n=== FIXED SIZE CHUNKING ===")
    fixed_chunk = chunk_text(pages, chunk_size=600, overlap=100)
    preview_chunks(fixed_chunk, how_many=3)

    print("\n=== PARAGRAPH CHUNKING ===")
    para_chunk = chunk_by_paragraph(pages, min_length=100, max_length=1000)
    preview_chunks(para_chunk, how_many=3)

    print("\n--- COMPARISON ---")
    print(f"Fixed size chunks: {len(fixed_chunk)}")
    print(f"paragraph chunks: {len(para_chunk)}")

    """
    chunks = chunk_text(pages, chunk_size=600, overlap=100)
    preview_chunks(chunks, how_many=3)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")
    """

