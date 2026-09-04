from mumblemed.utils.llm import (
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
        text = messages[-1]["content"].split("Hier ist der Text:")[-1]
        text = " ".join(text.split())
        expanded = text.replace(".", " Punkt").replace(",", " Komma")
        return _FakeResponse(expanded)


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


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
