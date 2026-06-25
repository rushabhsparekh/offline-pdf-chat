import pdfplumber
import json

def detect_structure(all_lines, body_font_size):
    """
    Structure detection based on real document observation:
    
    Main header: first large bottom gap in document
        - When gap_below of a line > 30, that line is the main header
        - Only applies once (first occurrence)
    
    Section: line where gap_above > 30 AND font > body_font_size
        - The line ABOVE the gap is the section title
    """
    
    header_found = False
    current_header = "Document"
    current_section = ""
    structured_lines = []
    
    for i, line in enumerate(all_lines):
        
        line_info = {
            "text": line["text"],
            "page_num": line["page_num"],
            "avg_font": line["avg_font"],
            "is_header": False,
            "is_section": False
        }
        
        # Find main header: first line with large gap below
        if not header_found:
            if line["gap_below"] > 30:
                line_info["is_header"] = True
                header_found = True
                structured_lines.append(line_info)
                continue
        
        # Find section: current line has gap_above > 30
        # AND the previous line has font bigger than body
        if line["gap_above"] > 30 and i > 0:
            prev_line = all_lines[i - 1]
            if prev_line["avg_font"] > body_font_size:
                # Mark previous line as section
                # Find it in structured_lines and update it
                for sl in reversed(structured_lines):
                    if sl["text"] == prev_line["text"]:
                        sl["is_section"] = True
                        break
        
        structured_lines.append(line_info)
    
    return structured_lines

def extract_structured_chunks(pdf_path, max_chunk_size=1000):
    """
    Smart chunker v2 - based on real document analysis.
    
    Section detection: large gap on BOTH sides (top >40, bottom >36)
    Paragraph detection: large gap on one side (top >20)
    Chunk size: one section per chunk, split if >max_chunk_size
    """
    
    PARA_GAP_TOP = 20
    SECTION_GAP_TOP = 40
    SECTION_GAP_BOTTOM = 36
    
    # First pass: collect all lines with their properties
    all_lines = []
    
    with pdfplumber.open(pdf_path) as pdf:
        
        # Detect body font size from most common size
        all_sizes = []
        for page in pdf.pages:
            for char in page.chars:
                all_sizes.append(round(char['size'], 1))
        body_font_size = max(set(all_sizes), key=all_sizes.count)
        print(f"Body font size detected: {body_font_size}")
        
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(
                keep_blank_chars=False,
                extra_attrs=["size"]
            )
            if not words:
                continue
            
            # Group words into lines
            lines = {}
            for word in words:
                y_key = round(word['top'] / 2) * 2
                if y_key not in lines:
                    lines[y_key] = {
                        "words": [],
                        "top": word['top'],
                        "sizes": []
                    }
                lines[y_key]["words"].append(word['text'])
                lines[y_key]["sizes"].append(round(word['size'], 1))
            
            sorted_lines = sorted(lines.items())
            
            # Calculate gaps: each line needs gap_above and gap_below
            line_list = []
            for i, (y_pos, line_data) in enumerate(sorted_lines):
                line_list.append({
                    "y_pos": y_pos,
                    "text": " ".join(line_data["words"]).strip(),
                    "avg_font": sum(line_data["sizes"]) / len(line_data["sizes"]),
                    "page_num": page_num + 1,
                    "gap_above": 0,
                    "gap_below": 0
                })
            
            # Now calculate gap_above and gap_below for each line
            for i in range(len(line_list)):
                if i > 0:
                    line_list[i]["gap_above"] = round(
                        line_list[i]["y_pos"] - line_list[i-1]["y_pos"], 1
                    )
                if i < len(line_list) - 1:
                    line_list[i]["gap_below"] = round(
                        line_list[i+1]["y_pos"] - line_list[i]["y_pos"], 1
                    )
            
            all_lines.extend(line_list)
    
    # Second pass: build chunks using detected structure
    structured_lines = detect_structure(all_lines, body_font_size)
    
    chunks = []
    chunk_index = 0
    current_header = "Document"
    current_section = "Introduction"
    current_section_text = ""
    current_page = 1
    
    for line in structured_lines:
        
        # Skip captions
        if line["avg_font"] < body_font_size - 1:
            continue
        
        if not line["text"]:
            continue
        
        # Main header — don't add to chunks, just track it
        if line["is_header"]:
            current_header = line["text"]
            print(f"Header detected: {current_header}")
            continue
        
        # Section title — save current section, start new one
        if line["is_section"]:
            if current_section_text.strip():
                new_chunks = split_section_into_chunks(
                    text=current_section_text.strip(),
                    header=current_header,
                    section=current_section,
                    page_num=current_page,
                    start_index=chunk_index,
                    max_size=max_chunk_size
                )
                chunks.extend(new_chunks)
                chunk_index += len(new_chunks)
            
            current_section = line["text"]
            current_section_text = ""
            print(f"Section detected: {current_section}")
            continue
        
        # Regular line — add to current section
        current_section_text += " " + line["text"]
        current_page = line["page_num"]
    
    # Save final section
    if current_section_text.strip():
        new_chunks = split_section_into_chunks(
            text=current_section_text.strip(),
            header=current_header,
            section=current_section,
            page_num=current_page,
            start_index=chunk_index,
            max_size=max_chunk_size
        )
        chunks.extend(new_chunks)
    
    return chunks


def split_section_into_chunks(text, header, section, page_num, 
                               start_index, max_size=1000):
    """
    Splits section text into parts.
    
    Key fix: collect all sentence boundaries first,
    then build parts by walking sentences in order.
    No character position drift possible.
    """
    chunks = []
    
    # If fits in one chunk, no splitting needed
    if len(text) <= max_size:
        chunks.append({
            "chunk_id": start_index,
            "page_num": page_num,
            "header": header,
            "section": section,
            "text": text
        })
        return chunks
    
    # Step 1: Split entire text into sentences first
    # Split on ". " to avoid splitting decimal numbers like "0.5"
    raw_sentences = text.replace('\n\n', ' ').split('. ')
    
    # Restore the period that split() removed
    sentences = []
    for i, s in enumerate(raw_sentences):
        s = s.strip()
        if not s:
            continue
        # Add period back except for the last sentence
        if i < len(raw_sentences) - 1:
            s = s + '.'
        sentences.append(s)
    
    # Step 2: Walk sentences and fill parts up to max_size
    parts = []
    current_part = ""
    
    for sentence in sentences:
        
        # If adding this sentence keeps us under limit, add it
        if len(current_part) + len(sentence) + 1 <= max_size:
            current_part += " " + sentence
        
        else:
            # Current part is full, save it
            if current_part.strip():
                parts.append(current_part.strip())
            
            # Start new part with current sentence
            current_part = sentence
    
    # Don't forget the last part
    if current_part.strip():
        parts.append(current_part.strip())
    
    # Step 3: Build chunk dicts from parts
    total_parts = len(parts)
    
    for i, part_text in enumerate(parts):
        chunks.append({
            "chunk_id": start_index + i,
            "page_num": page_num,
            "header": header,
            "section": f"{section} (part {i+1} of {total_parts})" if total_parts > 1 else section,
            "text": part_text
        })
    
    return chunks


def preview_smart_chunks(chunks, how_many=5):
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\n--- SMART CHUNK PREVIEW ---")
    for chunk in chunks[:how_many]:
        print(f"\nChunk {chunk['chunk_id']} | Page {chunk['page_num']}")
        print(f"Header: {chunk['header']}")
        print(f"Section: {chunk['section']}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Text: {chunk['text'][:200]}...")
        print("-" * 50)


def save_chunks_to_file(chunks, output_path="resources/chunks.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nChunks saved to {output_path}")


if __name__ == "__main__":
    
    pdf_path = "resources/Switchgear.pdf"
    
    chunks = extract_structured_chunks(pdf_path)
    preview_smart_chunks(chunks, how_many=5)
    save_chunks_to_file(chunks)
    
    # Summary statistics
    lengths = [len(c['text']) for c in chunks]
    print(f"\n--- CHUNK STATISTICS ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"Shortest chunk: {min(lengths)} chars")
    print(f"Longest chunk: {max(lengths)} chars")
    print(f"Average chunk: {sum(lengths)//len(lengths)} chars")
    
    # Show all unique sections found
    sections = list(set(c['section'] for c in chunks))
    print(f"\nSections detected: {len(sections)}")
    for s in sections:
        print(f"  - {s}")