from pdf_to_json import clean_text, is_noise_chunk, normalize_heading

LONG_TEXT = "SF6 gas pressure shall be maintained at 0.5 MPa. " * 5


def chunk(text=LONG_TEXT, headings=(), labels=("text",)):
    return {"text": text, "headings": list(headings), "labels": list(labels)}


def test_clean_text_strips_comments_citations_and_whitespace():
    raw = "Intro <!-- image -->  text[12]\r\n\n\n\nNext\t\tline  "
    assert clean_text(raw) == "Intro text\n\nNext line"


def test_normalize_heading_removes_section_numbers():
    assert normalize_heading("7. References") == "references"
    assert normalize_heading("2.3 Contents") == "contents"
    assert normalize_heading("IV. Index") == "index"
    assert normalize_heading("1) Preface") == "preface"


def test_normalize_heading_keeps_words_made_of_roman_letters():
    assert normalize_heading("Civil works") == "civil works"
    assert normalize_heading("Mix design") == "mix design"


def test_numbered_reference_section_is_noise():
    assert is_noise_chunk(chunk(headings=["Paper", "8 References"]))


def test_short_chunk_is_noise():
    assert is_noise_chunk(chunk(text="too short"))


def test_chunk_with_picture_and_text_is_kept():
    assert not is_noise_chunk(chunk(labels=["text", "picture"]))


def test_chunk_with_only_noise_labels_is_dropped():
    assert is_noise_chunk(chunk(labels=["page_header", "picture"]))


def test_regular_chunk_is_kept():
    assert not is_noise_chunk(chunk(headings=["Components", "Gas system"]))
