import json
import logging
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool, current_process
from urllib.parse import urlparse

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from mumblemed.config import get_env, load_env, resolve_path
from mumblemed.utils.chunking import chunk_document, get_audio_duration
from mumblemed.utils.llm import process_text_structure
from mumblemed.utils.speech import (
    generate_tts_audio,
    init_tts,
    random_speaker_selection,
    speaker_id_from_path,
)
from mumblemed.utils.splitting import split_dataframe_by_group


LOGGER = logging.getLogger(__name__)


@dataclass
class RealDatasetConfig:
    """Configuration for synthesizing speech from an existing report-text CSV."""

    model_name: str
    llm_endpoint: str
    llm_api_key: str | None
    dataset_path: pathlib.Path
    input_csv: pathlib.Path
    name: str
    num_docs: int
    num_workers: int
    gpu_id: str | None = None
    verbose: bool = False
    whisper_mode: bool = False
    tts_language: str = "de"
    use_default_voice: bool = False
    local_llm: bool = False
    words_per_minute: int = 80
    seed: int | None = None


tts_model = None


def clean_text(text: str) -> str:
    """Normalize report text while preserving meaningful line boundaries."""
    text = re.sub(r"(\\r\\n)+", "\n", text)
    text = re.sub(r"\\n+", "\n", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ").replace("\xa0", " ")
    lines = [re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _init_worker(tts_language: str = "de"):
    """Load the TTS model once per worker process."""
    global tts_model
    tts_model = init_tts(use_german_patch=(tts_language == "de"))


def _process_document(item):
    """Normalize and synthesize one input document row."""
    global tts_model
    (
        document_id,
        patient_id,
        document_text,
        dataset_path,
        model_name,
        llm_endpoint,
        llm_api_key,
        verbose,
        tts_language,
        use_default_voice,
        words_per_minute,
    ) = item

    try:
        client = OpenAI(base_url=llm_endpoint, api_key=llm_api_key or "-", timeout=600.0)

        document_text = clean_text(document_text)
        chunks = chunk_document(
            document=document_text,
            words_per_30s=_words_per_30s(words_per_minute),
            language_code=tts_language,
        )
        results = []

        for sentence_id, chunk in enumerate(chunks):
            try:
                tts_chunk = process_text_structure(client=client, model_name=model_name, text=chunk)
                label_chunk = chunk

                speaker_file = random_speaker_selection(use_default_voice=use_default_voice)
                speaker_id = speaker_id_from_path(speaker_file)

                audio_path = pathlib.Path(dataset_path) / f"{document_id}_sentence_{sentence_id}_{speaker_id}.wav"
                audio_file = generate_tts_audio(
                    tts_model=tts_model,
                    text=tts_chunk,
                    output_path=str(audio_path),
                    language_code=tts_language,
                    reference_voice_path=str(speaker_file) if speaker_file else None,
                )

                audio_duration = get_audio_duration(audio_path)

                results.append(
                    {
                        "document_id": document_id,
                        "patient_id": patient_id,
                        "tts_chunk": tts_chunk,
                        "label_chunk": label_chunk,
                        "speaker_id": speaker_id,
                        "duration_in_seconds": audio_duration,
                        "audio_path": audio_file,
                    }
                )

                if verbose:
                    LOGGER.info(
                        "[%s] Document %s Chunk %s | duration=%.2f",
                        current_process().name,
                        document_id,
                        sentence_id,
                        audio_duration,
                    )

            except Exception as exc:
                LOGGER.exception("Error processing chunk in document %s: %s", document_id, exc)

        return results
    except Exception as exc:
        LOGGER.exception("Failed to process document %s: %s", document_id, exc)
        return None


def _split_and_write(
    df: pd.DataFrame,
    csv_path: pathlib.Path,
    suffix: str,
    whisper_mode: bool,
    seed: int | None = None,
) -> None:
    """Write split CSVs and stats for a real-text dataset run."""
    split_group = "patient_id" if "patient_id" in df.columns else "document_id"
    splits = split_dataframe_by_group(df=df, group_column=split_group, seed=seed)

    stats_report = {
        "split_group": split_group,
        "total_documents": int(df.document_id.nunique()),
        "total_samples": int(len(df)),
        "splits": {},
    }

    if "patient_id" in df.columns:
        stats_report["total_patients"] = int(df.patient_id.nunique())

    for mode, df_tmp in splits.items():
        if whisper_mode and mode in {"train", "val"} and "duration_in_seconds" in df_tmp.columns:
            df_tmp = df_tmp[df_tmp.duration_in_seconds <= 30]
        out_file = csv_path / f"{mode}_{suffix}.csv"
        df_tmp.to_csv(out_file, index=False, sep=";")
        LOGGER.info("Saved %s split to %s", mode, out_file)

        stats_report["splits"][mode] = _compute_split_stats(df_tmp)

    stats_path = csv_path / f"stats_{suffix}.json"
    _write_stats(stats_path, stats_report)


def _compute_split_stats(df_split: pd.DataFrame) -> dict:
    """Summarize sample counts, duration, speakers, and transcript length."""
    stats = {
        "samples": int(len(df_split)),
        "documents": int(df_split.document_id.nunique()) if "document_id" in df_split.columns else 0,
    }

    if "duration_in_seconds" in df_split.columns and not df_split.empty:
        durations = df_split["duration_in_seconds"].astype(float)
        stats["duration_seconds"] = {
            "min": float(durations.min()),
            "max": float(durations.max()),
            "mean": float(durations.mean()),
            "p50": float(durations.quantile(0.5)),
            "p90": float(durations.quantile(0.9)),
        }

    if "speaker_id" in df_split.columns:
        stats["speaker_counts"] = df_split["speaker_id"].value_counts().to_dict()

    if "patient_id" in df_split.columns:
        stats["patients"] = int(df_split.patient_id.nunique())

    if "label_chunk" in df_split.columns and not df_split.empty:
        word_counts = df_split["label_chunk"].astype(str).apply(lambda x: len(x.split()))
        stats["text_words"] = {
            "total": int(word_counts.sum()),
            "mean": float(word_counts.mean()),
            "max": int(word_counts.max()),
        }

    return stats


def _write_stats(path: pathlib.Path, stats: dict) -> None:
    """Persist split statistics as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    LOGGER.info("Saved stats report to %s", path)


def validate_real_config(config: RealDatasetConfig) -> None:
    """Validate paths, LLM settings, input CSV schema, and voice-source configuration."""
    if config.num_docs <= 0:
        raise ValueError("num_docs must be > 0")
    if config.num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    if config.words_per_minute <= 0:
        raise ValueError("words_per_minute must be > 0")
    if not config.model_name:
        raise ValueError("model_name is required")
    if not config.llm_endpoint:
        raise ValueError("llm_endpoint is required")

    parsed = urlparse(config.llm_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"llm_endpoint must be a valid URL. Got: {config.llm_endpoint}")

    if not config.local_llm and not config.llm_api_key:
        raise ValueError("llm_api_key is required unless --local-llm is set.")

    config.dataset_path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=config.dataset_path, delete=True):
            pass
    except Exception as exc:
        raise PermissionError(f"Cannot write to {config.dataset_path}: {exc}") from exc

    input_csv = config.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, nrows=1)
    if "report_clear" not in df.columns:
        raise KeyError(f"Expected 'report_clear' column in {input_csv}")

    if "patient_id" in df.columns:
        patient_ids = pd.read_csv(input_csv, usecols=["patient_id"])["patient_id"]
        if patient_ids.isna().any() or patient_ids.astype(str).str.strip().eq("").any():
            raise ValueError("patient_id column must not contain empty values.")

    if not config.use_default_voice:
        voices_path = pathlib.Path(os.getenv("SPEAKER_VOICES_PATH") or (pathlib.Path(__file__).resolve().parents[2] / "speaker-voices"))
        if not voices_path.exists():
            raise FileNotFoundError(f"Speaker voices directory not found: {voices_path}")
        if not any(x.is_file() for x in voices_path.iterdir()):
            raise RuntimeError(f"No speaker voice files found in {voices_path}")


def generate_real_dataset(config: RealDatasetConfig) -> None:
    """Run the real text -> text normalization -> TTS dataset pipeline."""
    validate_real_config(config)
    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    dataset_path = config.dataset_path / f"real-{config.name}"
    dataset_path.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_path
    input_csv = config.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "report_clear" not in df.columns:
        raise KeyError(f"Expected 'report_clear' column in {input_csv}")
    df = df.head(config.num_docs)
    docs = df.report_clear.values
    indices = df.index.values
    patient_ids = df.patient_id.values if "patient_id" in df.columns else indices

    LOGGER.info("Found %s documents. Starting TTS generation.", len(docs))

    items = [
        (
            idx,
            patient_id,
            doc,
            str(dataset_path),
            config.model_name,
            config.llm_endpoint,
            config.llm_api_key,
            config.verbose,
            config.tts_language,
            config.use_default_voice,
            config.words_per_minute,
        )
        for idx, patient_id, doc in zip(indices, patient_ids, docs)
    ]

    results = []
    with Pool(
        processes=config.num_workers,
        initializer=_init_worker,
        initargs=(config.tts_language,),
    ) as pool:
        for result in tqdm(pool.imap(_process_document, items), total=len(items)):
            if result is not None:
                results.append(result)

    if not results:
        raise RuntimeError("No documents generated. Check logs for errors.")

    df = pd.concat([pd.DataFrame(x) for x in results if x is not None], axis=0)
    _split_and_write(df, csv_path, config.name, whisper_mode=config.whisper_mode, seed=config.seed)


def build_real_config_from_env(
    name: str,
    input_csv: str,
    num_docs: int | None = None,
    num_workers: int | None = None,
    dataset_path: str | None = None,
    model_name: str | None = None,
    llm_endpoint: str | None = None,
    llm_api_key: str | None = None,
    gpu_id: str | None = None,
    seed: int | None = None,
    verbose: bool = False,
    whisper_mode: bool = False,
    tts_language: str | None = None,
    use_default_voice: bool | str | None = None,
    local_llm: bool | str | None = None,
    words_per_minute: int | str | None = None,
) -> RealDatasetConfig:
    """Build a real-text dataset config from CLI overrides, config files, and .env."""
    load_env()
    return RealDatasetConfig(
        model_name=model_name or get_env("LLM_NAME", required=True),
        llm_endpoint=llm_endpoint or get_env("LLM_ENDPOINT", required=True),
        llm_api_key=llm_api_key or get_env("LLM_API_KEY", None),
        dataset_path=resolve_path(dataset_path or get_env("REAL_DATASET_PATH", required=True)),
        input_csv=resolve_path(input_csv),
        name=name,
        num_docs=num_docs or int(get_env("NUM_REAL_DOCUMENTS", 50)),
        num_workers=num_workers or int(get_env("NUM_WORKERS", 4)),
        gpu_id=gpu_id or get_env("GPU_ID", None),
        seed=seed,
        verbose=verbose,
        whisper_mode=whisper_mode,
        tts_language=tts_language or get_env("TTS_LANGUAGE", "de"),
        use_default_voice=_parse_bool(
            use_default_voice if use_default_voice is not None else get_env("USE_DEFAULT_TTS_VOICE", None)
        ),
        local_llm=_parse_bool(local_llm if local_llm is not None else get_env("LOCAL_LLM", None)),
        words_per_minute=int(
            words_per_minute if words_per_minute is not None else get_env("WORDS_PER_MINUTE", 80)
        ),
    )


def _parse_bool(value: bool | str | None) -> bool:
    """Parse common env/CLI truthy values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _words_per_30s(words_per_minute: int) -> int:
    """Convert a speaking-rate estimate into the chunker's 30-second budget."""
    return max(1, round(words_per_minute / 2))
