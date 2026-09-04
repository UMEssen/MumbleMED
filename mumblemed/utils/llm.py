import random
from math import floor
from pathlib import Path

import pandas as pd

from mumblemed.utils.chunking import chunk_document

# Built-in coding system names and their default CSV filenames
DEFAULT_CODING_SYSTEM_FILES = {
    "ICD": "icd10gm.csv",
    "OPS": "ops.csv",
    "RADLEX": "radlex.csv",
}
TTS_LENGTH_POLICIES = {"warn", "rechunk", "skip"}
DEFAULT_TTS_LENGTH_POLICY = "rechunk"


def _load_display_table(data_dir: Path, filename: str) -> pd.DataFrame:
    table_path = data_dir / filename
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing coding system file: {table_path}. "
            "Set CODING_SYSTEMS_PATH or place the CSVs under data/coding-systems."
        )
    return pd.read_csv(table_path)[["display", "code"]]


def load_coding_tables(
    data_dir: Path,
    system_names: list[str] | None = None,
    system_files: dict[str, str] | None = None,
) -> dict:
    """Load coding system tables. Each CSV must have columns 'display' and 'code'.

    Args:
        data_dir: Directory containing the CSV files.
        system_names: Which systems to load (e.g. ["ICD", "OPS"]). Default: all built-in.
        system_files: Optional mapping name -> filename for custom systems or overrides.
            Merged with built-in defaults (ICD, OPS, RADLEX). Example: {"CUSTOM": "mycodes.csv"}.
    """
    names = system_names or list(DEFAULT_CODING_SYSTEM_FILES)
    files = dict(DEFAULT_CODING_SYSTEM_FILES)
    if system_files:
        files.update(system_files)
    return {name: _load_display_table(data_dir, files[name]) for name in names if name in files}


# Randomly select a subset of coding systems and their display labels
def generate_random_displays(coding_tables: dict):
    systems = list(coding_tables.keys())
    if not systems:
        return {}, {}
    selected_systems = random.sample(systems, random.randint(1, len(systems)))

    selected_displays = {}
    selected_codes = {}

    for system in selected_systems:
        table = coding_tables[system]
        n = random.randint(0, min(3, len(table)))
        if n > 0:
            rows = table.sample(n)
            selected_displays[system] = rows["display"].tolist()
            selected_codes[system] = rows["code"].tolist()
        else:
            selected_displays[system] = []
            selected_codes[system] = []
    return selected_displays, selected_codes


# Example LLM text generation using the OpenAI-compatible API.

def generate_synthetic_medical_text(client, model_name, displays):
    # LLM prompt (tailored for a clinical narrative)
    prompt = (
        "Denk Dir eine klinische Geschichte oder medizinische Erzählung, die die folgenden medizinischen Fachbegriffe und Formulierungen nutzen. "
        "Der Text soll ein realistisches medizinisches Szenario darstellen, das auf den ausgewählten Anzeigen basiert. Verwende nur reinen Text und kein Markdown! \n\n"
    )
    
    # Dynamically include the selected systems
    for system, display in displays.items():
        prompt += f"{system}: {display}\n"
    
    prompt += "\nBeispieltext:\n\n"
    prompt += (
        "Ein 55-jähriger Patient kommt in die Klinik mit Symptomen von Diabetes mellitus Typ 2. "
        "Er hat eine längere Geschichte von Hypertonie und wurde kürzlich zur Durchführung einer Laparoskopischen Appendektomie in die Klinik überwiesen. "
        "Zusätzlich wurde eine Röntgenaufnahme des Lendenwirbelbereichs gemacht, um mögliche Wirbelsäulenprobleme auszuschließen."
    )

    # Call the OpenAI-compatible API
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the response
    synthetic_text = response.choices[0].message.content.strip()

    return synthetic_text


def process_text_structure(client, model_name, text):
    # LLM prompt (tailored for a clinical narrative)
    intro = r"""
        Du bist ein Assistent zur Vorbereitung medizinischer Texte für das Fine-Tuning eines Speech-to-Text-Modells. Deine Aufgabe ist es, einen gegebenen medizinischen Text in eine lautsprachlich vollständige Form umzuwandeln, bei der:

        Alle Satzzeichen und Sonderzeichen ausgeschrieben werden, zum Beispiel:

        . → „Punkt“

        , → „Komma“

        - → „Bindestrich“

        % → „Prozent“

        ( → „Klammerauf“

        ) → „Klammerzu“

        : → „Doppelpunkt“

        ? → „Fragezeichen“

        ! → „Ausrufezeichen“

        / → „Schrägstrich“

        \ → „Rückslash“

        = → „Gleichzeichen“

        + → „Pluszeichen“

        Medizinische Abkürzungen ausgeschrieben werden, zum Beispiel:

        i.v. → „intravenös“

        DD → „Differenzialdiagnose“

        ED → „Erstdiagnose“

        Z.n. → „Zustand nach“

        KM → „Kontrastmittel“

        z.B. → „zum Beispiel“

        ex. → „Exitus“

        o.g. → „oben genannt“

        a.e. → „am ehesten“

        V. → „Verdacht“

        cava inf. → „Vena cava inferior“

        neg. → „negativ“

        pos. → „positiv“

        Wo → „Woche“ oder „Wochen“ (je nach Kontext)

        Zahlen bleiben erhalten, dürfen aber ggf. ausgeschrieben werden, wenn sie Teil eines Wortes oder einer Redewendung sind.

        Wichtig: Für jedes Ersetzte Wort füge bitte kein zusätzliches Komma oder Satzzeichen an. 

        Bitte gib nur die umgewandelte Version des Textes zurück in klarer, lautschriftlich orientierter Form, vollständig ausgeschrieben, ohne Abkürzungen oder Sonderzeichen. Beachte, dass der Text für Sprachsynthese gedacht ist und daher besonders klar und eindeutig sein muss und gleichzeitig konsistent mit dem Orignal Text.

        Hier ist der Text:
    
    """
    
    prompt = f"""
        {intro}

        {text}
    """

    # Call the OpenAI-compatible API
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the response
    synthetic_text = response.choices[0].message.content.strip()

    return synthetic_text


def count_words(text: str) -> int:
    """Count whitespace-separated words in generated label or TTS text."""
    return len(str(text).split())


def tts_expansion_ratio(label_text: str, tts_text: str) -> float:
    """Return how much the TTS-oriented text expanded relative to the label."""
    label_word_count = count_words(label_text)
    if label_word_count == 0:
        return 0.0
    return count_words(tts_text) / label_word_count


def derive_rechunk_word_budget(label_text: str, tts_text: str, max_tts_words: int) -> int:
    """Choose a smaller label budget from the observed TTS expansion ratio."""
    ratio = tts_expansion_ratio(label_text, tts_text)
    if ratio <= 0:
        return max(1, max_tts_words)
    return max(1, floor(max_tts_words / ratio))


def build_tts_length_metadata(
    label_chunk: str,
    tts_chunk: str,
    max_tts_words: int,
    was_rechunked_after_tts_transform: bool = False,
) -> dict:
    """Create auditable length metadata for one TTS synthesis candidate."""
    label_word_count = count_words(label_chunk)
    tts_word_count = count_words(tts_chunk)
    return {
        "label_chunk": label_chunk,
        "tts_chunk": tts_chunk,
        "label_word_count": label_word_count,
        "tts_word_count": tts_word_count,
        "tts_expansion_ratio": float(tts_word_count / label_word_count) if label_word_count else 0.0,
        "was_rechunked_after_tts_transform": was_rechunked_after_tts_transform,
        "was_over_tts_word_budget": tts_word_count > max_tts_words,
    }


def prepare_tts_segments(
    client,
    model_name: str,
    label_chunk: str,
    language_code: str,
    max_tts_words: int,
    tts_length_policy: str = DEFAULT_TTS_LENGTH_POLICY,
) -> list[dict]:
    """Transform label text for TTS and optionally rechunk if lautschrift expands too far."""
    if tts_length_policy not in TTS_LENGTH_POLICIES:
        raise ValueError(f"tts_length_policy must be one of {sorted(TTS_LENGTH_POLICIES)}")

    tts_chunk = process_text_structure(client=client, model_name=model_name, text=label_chunk)
    metadata = build_tts_length_metadata(
        label_chunk=label_chunk,
        tts_chunk=tts_chunk,
        max_tts_words=max_tts_words,
    )

    if not metadata["was_over_tts_word_budget"] or tts_length_policy == "warn":
        return [metadata]
    if tts_length_policy == "skip":
        return []

    rechunk_budget = derive_rechunk_word_budget(label_chunk, tts_chunk, max_tts_words)
    subchunks = chunk_document(
        document=label_chunk,
        words_per_30s=rechunk_budget,
        language_code=language_code,
        chunking_mode="sentence-divide",
    )

    segments = []
    for subchunk in subchunks:
        sub_tts_chunk = process_text_structure(client=client, model_name=model_name, text=subchunk)
        segments.append(
            build_tts_length_metadata(
                label_chunk=subchunk,
                tts_chunk=sub_tts_chunk,
                max_tts_words=max_tts_words,
                was_rechunked_after_tts_transform=True,
            )
        )
    return segments
