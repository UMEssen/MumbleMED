from mumblemed.utils.chunking import chunk_document, replace_special_characters, split_into_chunks
from mumblemed.utils.charmap import general_char_map


def test_replace_special_characters_german():
    text = "A. B"
    out = replace_special_characters(text, general_char_map, "de")
    assert "Punkt" in out
    assert "A" in out


def test_chunk_document_splits_sentences():
    doc = "First sentence here. Second sentence follows."
    chunks = chunk_document(document=doc, words_per_30s=500)
    assert len(chunks) >= 1
    assert "First sentence" in chunks[0]


def test_split_into_chunks_splits_overlong_sentence_by_word_window():
    chunks = split_into_chunks(["one two three four"], words_per_30s=2)
    assert chunks == ["one two", "three four"]


def test_chunk_document_preserves_german_report_lines():
    doc = """Diagnosen:
C34.1 Bronchialkarzinom links
TNM cT2a cN1 cM0
Therapie:
Carboplatin AUC5 Paclitaxel 175 mg/m2
Entlassung:
Patient stabil, Wiedervorstellung empfohlen"""

    chunks = chunk_document(document=doc, words_per_30s=6, language_code="de")

    assert len(chunks) > 1
    assert "Diagnosen:" in chunks[0]
    assert all(len(chunk.split()) <= 6 for chunk in chunks)


def test_split_into_chunks_prefers_clinical_separators_for_long_sections():
    sentence = "Befund: links basal Verschattung, kein Erguss, keine Pneumonie, Kontrolle empfohlen"

    chunks = split_into_chunks([sentence], words_per_30s=4)

    assert len(chunks) > 1
    assert chunks[0].startswith("Befund:")
    assert chunks[0].endswith(",")
    assert all(len(chunk.split()) <= 4 for chunk in chunks)
