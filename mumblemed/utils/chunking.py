import re

import nltk
from pydub import AudioSegment


_NLTK_LANGUAGE_BY_CODE = {
    "de": "german",
    "de-de": "german",
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
}

CHUNKING_MODES = {"sentence-strict", "sentence-divide", "word-divide"}
DEFAULT_CHUNKING_MODE = "sentence-divide"


def _nltk_language(language_code: str) -> str:
    """Return the NLTK Punkt language for a TTS language code."""
    return _NLTK_LANGUAGE_BY_CODE.get(language_code.lower(), "english")


def _ensure_punkt(language: str):
    # NLTK 3.9+ uses punkt_tab for sent_tokenize; older releases used punkt.
    try:
        nltk.data.find(f"tokenizers/punkt_tab/{language}")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def replace_special_characters(text, special_char_map, language_code):
    """Replace symbols with pronounceable text for the requested language."""
    for char, translations in special_char_map.items():
        if language_code in translations:
            text = text.replace(char, translations[language_code])
    return text


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_word_window(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _split_on_clinical_separators(text: str) -> list[str]:
    """Split long clinical sections at punctuation that often separates findings."""
    clauses = re.findall(r"[^;,:]+[;,:]?", text)
    return [clause.strip() for clause in clauses if clause.strip()]


def _split_overlong_unit(text: str, max_words: int) -> list[str]:
    """Split one over-budget sentence or line into deterministic smaller units."""
    if _word_count(text) <= max_words:
        return [text]

    clauses = _split_on_clinical_separators(text)
    if len(clauses) <= 1:
        return _split_by_word_window(text, max_words)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for clause in clauses:
        clause_word_count = _word_count(clause)
        if clause_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.extend(_split_by_word_window(clause, max_words))
        elif current_chunk and current_word_count + clause_word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [clause]
            current_word_count = clause_word_count
        else:
            current_chunk.append(clause)
            current_word_count += clause_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _split_lines(document: str) -> list[str]:
    """Preserve non-empty report lines as candidate section boundaries."""
    document = document.replace("\\r\\n", "\n").replace("\\n", "\n")
    document = document.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in document.split("\n"):
        line = re.sub(r"[ \t\xa0]+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def _sentences_from_lines(document: str, language: str) -> list[str]:
    sentences = []
    for line in _split_lines(document):
        line_sentences = nltk.sent_tokenize(line, language=language)
        sentences.extend(sentence.strip() for sentence in line_sentences if sentence.strip())
    return sentences


def _normalize_flat_text(document: str) -> str:
    """Normalize a document into one whitespace-collapsed text stream."""
    return re.sub(r"\s+", " ", " ".join(_split_lines(document))).strip()


def split_into_chunks(sentences, words_per_30s, chunking_mode: str = DEFAULT_CHUNKING_MODE):
    """Group text units according to the selected chunking mode."""
    if chunking_mode not in CHUNKING_MODES:
        raise ValueError(f"chunking_mode must be one of {sorted(CHUNKING_MODES)}")

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        units = [sentence]
        if chunking_mode == "sentence-divide":
            units = _split_overlong_unit(sentence, words_per_30s)

        for unit in units:
            word_count = _word_count(unit)
            if current_chunk and current_word_count + word_count > words_per_30s:
                chunks.append(" ".join(current_chunk))
                current_chunk = [unit]
                current_word_count = word_count
            else:
                current_chunk.append(unit)
                current_word_count += word_count

        if current_chunk and current_word_count >= words_per_30s:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(
    document: str,
    words_per_30s=40,
    language_code: str = "en",
    chunking_mode: str = DEFAULT_CHUNKING_MODE,
):
    """Split a clinical document into chunks using the requested strategy."""
    if chunking_mode not in CHUNKING_MODES:
        raise ValueError(f"chunking_mode must be one of {sorted(CHUNKING_MODES)}")

    if chunking_mode == "word-divide":
        return _split_by_word_window(_normalize_flat_text(document), words_per_30s)

    language = _nltk_language(language_code)
    _ensure_punkt(language)
    sentences = _sentences_from_lines(document, language=language)

    return split_into_chunks(
        sentences=sentences,
        words_per_30s=words_per_30s,
        chunking_mode=chunking_mode,
    )


def get_audio_duration(path):
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0  # milliseconds to seconds
