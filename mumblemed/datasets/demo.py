import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf


LOGGER = logging.getLogger(__name__)


DEMO_DOCUMENT = (
    "A 62-year-old patient presents with progressive dyspnea and fatigue. "
    "The history includes arterial hypertension, type 2 diabetes mellitus, and prior appendectomy. "
    "Computed tomography of the chest shows a right lower lobe consolidation without pleural effusion. "
    "Laboratory testing demonstrates elevated C-reactive protein and mild leukocytosis. "
    "The working diagnosis is community-acquired pneumonia with relevant cardiometabolic comorbidity. "
    "Intravenous antibiotic therapy is started after blood cultures are obtained. "
    "Oxygen saturation remains stable under low-flow oxygen. "
    "Follow-up radiography is recommended after clinical improvement. "
    "The patient is advised to continue antihypertensive medication and monitor blood glucose closely. "
    "Discharge planning includes outpatient pulmonary assessment and primary care follow-up."
)


DEMO_SENTENCES = [
    sentence.strip()
    for sentence in DEMO_DOCUMENT.split(". ")
    if sentence.strip()
]


@dataclass
class DemoDatasetConfig:
    """Configuration for the bundled public demo dataset."""

    dataset_path: pathlib.Path
    csv_path: pathlib.Path
    num_samples: int = 10
    sample_rate: int = 16_000
    duration_seconds: float = 1.0
    audio_mode: Literal["tone", "tts"] = "tone"
    tts_language: str = "en"


def validate_demo_config(config: DemoDatasetConfig) -> None:
    """Validate demo sizing, audio mode, and output writability."""
    if config.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if config.duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if config.audio_mode not in {"tone", "tts"}:
        raise ValueError("audio_mode must be 'tone' or 'tts'")

    for path in [config.dataset_path, config.csv_path]:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()


def generate_demo_dataset(config: DemoDatasetConfig) -> None:
    """Create a tiny deterministic dataset from one bundled synthetic document.

    By default, audio files are quiet synthetic tones for a no-download smoke
    test. With audio_mode="tts", the demo uses the TTS model's default voice
    without a speaker reference clip.
    """
    validate_demo_config(config)
    config.dataset_path.mkdir(parents=True, exist_ok=True)
    config.csv_path.mkdir(parents=True, exist_ok=True)

    rows = []
    LOGGER.info("Creating %s demo samples in %s", config.num_samples, config.dataset_path)
    tts_model = _init_demo_tts(config) if config.audio_mode == "tts" else None
    for idx in range(config.num_samples):
        sentence = DEMO_SENTENCES[idx % len(DEMO_SENTENCES)]
        speaker_id = "default" if tts_model else "synthetic_tone"
        audio_path = config.dataset_path / f"demo_document_0_sentence_{idx:02d}_speaker_{speaker_id}.wav"
        if tts_model:
            duration = _write_demo_tts_audio(tts_model, sentence, audio_path, config.tts_language)
        else:
            duration = _write_demo_tone_audio(
                audio_path=audio_path,
                sample_rate=config.sample_rate,
                duration_seconds=config.duration_seconds,
                frequency_hz=220 + (idx * 20),
            )
        rows.append(
            {
                "document_id": 0,
                "sentence_id": idx,
                "tts_chunk": sentence,
                "label_chunk": sentence,
                "speaker_id": speaker_id,
                "duration_in_seconds": duration,
                "audio_path": str(audio_path),
                "source": "bundled synthetic demo document",
            }
        )

    df = pd.DataFrame(rows)
    _split_and_write(df, config.csv_path)
    LOGGER.info("Demo dataset ready: audio=%s csv=%s", config.dataset_path, config.csv_path)


def _init_demo_tts(config: DemoDatasetConfig):
    """Load the TTS model lazily only for the optional TTS demo mode."""
    from mumblemed.utils.speech import init_tts

    return init_tts(use_german_patch=(config.tts_language == "de"))


def _write_demo_tts_audio(tts_model, sentence: str, audio_path: pathlib.Path, tts_language: str) -> float:
    """Write one default-voice TTS demo sample and return its measured duration."""
    from mumblemed.utils.chunking import get_audio_duration
    from mumblemed.utils.speech import generate_tts_audio

    generate_tts_audio(
        tts_model=tts_model,
        text=sentence,
        output_path=str(audio_path),
        language_code=tts_language,
        reference_voice_path=None,
    )
    return float(get_audio_duration(audio_path))


def _write_demo_tone_audio(
    audio_path: pathlib.Path,
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float,
) -> float:
    """Write one quiet tone placeholder and return its configured duration."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    fade = np.minimum(np.linspace(0, 1, len(t)), np.linspace(1, 0, len(t)))
    waveform = 0.03 * np.sin(2 * np.pi * frequency_hz * t) * np.minimum(fade * 20, 1)
    sf.write(audio_path, waveform.astype(np.float32), sample_rate)
    return duration_seconds


def _split_and_write(df: pd.DataFrame, csv_path: pathlib.Path) -> None:
    """Write deterministic 80/10/10-ish splits for the demo metadata."""
    total = len(df)
    train_end = max(1, int(total * 0.8))
    val_end = min(total, train_end + max(1, int(total * 0.1)))

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }
    if splits["test"].empty and total > 1:
        splits["test"] = splits["val"].tail(1)
        splits["val"] = splits["val"].head(max(0, len(splits["val"]) - 1))

    stats = {
        "total_documents": int(df.document_id.nunique()),
        "total_samples": int(total),
        "splits": {},
    }
    for name, split_df in splits.items():
        out_file = csv_path / f"{name}.csv"
        split_df.to_csv(out_file, index=False, sep=";")
        stats["splits"][name] = {
            "samples": int(len(split_df)),
            "duration_seconds": float(split_df["duration_in_seconds"].sum()) if not split_df.empty else 0.0,
        }
        LOGGER.info("Saved %s split with %s samples to %s", name, len(split_df), out_file)

    with (csv_path / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
