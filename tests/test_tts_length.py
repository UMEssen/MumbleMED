from mumblemed.utils.llm import (
    build_synthetic_text_prompt,
    build_text_structure_prompt,
    build_tts_length_metadata,
    derive_rechunk_word_budget,
    prepare_tts_segments,
    tts_expansion_ratio,
)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, model, temperature, messages):
        prompt = messages[-1]["content"]
        marker = "Here is the text:" if "Here is the text:" in prompt else "Hier ist der Text:"
        text = prompt.split(marker)[-1]
        text = " ".join(text.split())
        expanded = text.replace(".", " Punkt").replace(",", " Komma")
        return _FakeResponse(expanded)


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_synthetic_text_prompt_can_be_german():
    prompt = build_synthetic_text_prompt(
        displays={"ICD": ["Diabetes mellitus Typ 2"]},
        prompt_language="de",
    )

    assert "Denk Dir eine klinische Geschichte" in prompt
    assert "Beispieltext" in prompt
    assert "Diabetes mellitus Typ 2" in prompt


def test_synthetic_text_prompt_can_be_english():
    prompt = build_synthetic_text_prompt(
        displays={"ICD": ["type 2 diabetes mellitus"]},
        prompt_language="en",
    )

    assert "Create a clinical story" in prompt
    assert "Example text" in prompt
    assert "type 2 diabetes mellitus" in prompt
    assert "Denk Dir" not in prompt


def test_text_structure_prompt_uses_english_spoken_form_instructions():
    prompt = build_text_structure_prompt(
        text="No acute finding.",
        language_code="en",
    )

    assert "clear spoken form" in prompt
    assert "period" in prompt
    assert "Punkt" not in prompt


def test_text_structure_prompt_forbids_periods_from_english_line_breaks():
    prompt = build_text_structure_prompt(
        text="Diagnoses\nCOPD GOLD II",
        language_code="en",
    )

    assert "line breaks, headings, and section boundaries" in prompt
    assert "Do not insert or spell out \"period\"" in prompt
    assert "If a line has no period in the original text" in prompt


def test_text_structure_prompt_forbids_punkt_from_german_line_breaks():
    prompt = build_text_structure_prompt(
        text="Diagnosen\nCOPD GOLD II",
        language_code="de",
    )

    assert "Zeilenumbrüche, Überschriften und Abschnittswechsel" in prompt
    assert "Füge kein „Punkt“ ein" in prompt
    assert "Wenn im Originaltext am Zeilenende kein Punkt steht" in prompt


def test_tts_expansion_ratio_counts_transformed_text():
    ratio = tts_expansion_ratio("A. B.", "A Punkt B Punkt")
    assert ratio == 2.0


def test_derive_rechunk_word_budget_uses_observed_expansion():
    budget = derive_rechunk_word_budget(
        label_text="one two three four",
        tts_text="one two three four five six eight nine",
        max_tts_words=4,
    )
    assert budget == 2


def test_build_tts_length_metadata_reports_budget_state():
    metadata = build_tts_length_metadata(
        label_chunk="A. B.",
        tts_chunk="A Punkt B Punkt",
        max_tts_words=3,
    )

    assert metadata["label_word_count"] == 2
    assert metadata["tts_word_count"] == 4
    assert metadata["tts_expansion_ratio"] == 2.0
    assert metadata["was_over_tts_word_budget"] is True


def test_prepare_tts_segments_rechunks_once_after_expansion():
    segments = prepare_tts_segments(
        client=_FakeClient(),
        model_name="fake-model",
        label_chunk="one. two. three. four.",
        language_code="en",
        max_tts_words=4,
        tts_length_policy="rechunk",
    )

    assert len(segments) == 2
    assert all(segment["was_rechunked_after_tts_transform"] for segment in segments)
    assert all(segment["tts_word_count"] <= 4 for segment in segments)


def test_prepare_tts_segments_can_skip_overlong_transformed_text():
    segments = prepare_tts_segments(
        client=_FakeClient(),
        model_name="fake-model",
        label_chunk="one. two.",
        language_code="en",
        max_tts_words=2,
        tts_length_policy="skip",
    )

    assert segments == []


def test_prepare_tts_segments_does_not_create_punkt_from_report_line_breaks():
    segments = prepare_tts_segments(
        client=_FakeClient(),
        model_name="fake-model",
        label_chunk="Diagnosen\nCOPD GOLD II\nAufnahme\nDyspnoe seit drei Tagen",
        language_code="de",
        max_tts_words=20,
        tts_length_policy="warn",
    )

    assert len(segments) == 1
    assert "Punkt" not in segments[0]["tts_chunk"]
