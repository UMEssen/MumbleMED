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


def test_split_into_chunks_does_not_emit_empty_chunk_for_long_sentence():
    chunks = split_into_chunks(["one two three four"], words_per_30s=2)
    assert chunks == ["one two three four"]
