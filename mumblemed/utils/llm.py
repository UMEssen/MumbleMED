import random
from pathlib import Path

import pandas as pd

# Built-in coding system names and their default CSV filenames
DEFAULT_CODING_SYSTEM_FILES = {
    "ICD": "icd10gm.csv",
    "OPS": "ops.csv",
    "RADLEX": "radlex.csv",
}


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
