import nltk
from pydub import AudioSegment


def _ensure_punkt():
    # NLTK 3.9+ uses punkt_tab for sent_tokenize; older releases used punkt.
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

# Function to replace special characters with words for TTS generation based on language
def replace_special_characters(text, special_char_map, language_code):
    for char, translations in special_char_map.items():
        if language_code in translations:
            text = text.replace(char, translations[language_code])
    return text


def split_into_chunks(sentences, words_per_30s):
    """Group sentences into chunks without emitting empty chunks."""
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_chunk and current_word_count + word_count > words_per_30s:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Split text into sentences
def chunk_document(document: str, words_per_30s=40):

    _ensure_punkt()
    # Parse the document to sentences
    sentences = nltk.sent_tokenize(document)

    # Chunk the document
    return split_into_chunks(sentences=sentences, words_per_30s=words_per_30s)




def get_audio_duration(path):
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0  # milliseconds to seconds
