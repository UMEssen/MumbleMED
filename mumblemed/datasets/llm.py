import json
import logging
import os
import pathlib
import random
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool, current_process
from urllib.parse import urlparse

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from mumblemed.config import get_env, load_env, resolve_path
from mumblemed.paths import PROJECT_ROOT
from mumblemed.utils.chunking import (
    CHUNKING_MODES,
    DEFAULT_CHUNKING_MODE,
    chunk_document,
    get_audio_duration,
)
from mumblemed.utils.llm import (
    DEFAULT_TTS_LENGTH_POLICY,
    TTS_LENGTH_POLICIES,
    generate_random_displays,
    generate_synthetic_medical_text,
    load_coding_tables,
    prepare_tts_segments,
)
from mumblemed.utils.speech import (
    generate_tts_audio,
    init_tts,
    random_speaker_selection,
    speaker_id_from_path,
)
from mumblemed.utils.splitting import split_dataframe_by_group


LOGGER = logging.getLogger(__name__)


@dataclass
class LlmDatasetConfig:
    """Configuration for synthetic text generation followed by TTS synthesis."""

    model_name: str
    llm_endpoint: str
    llm_api_key: str | None
    dataset_path: pathlib.Path
    csv_path: pathlib.Path
    num_docs: int
    num_workers: int
    gpu_id: str | None = None
    seed: int | None = None
    verbose: bool = False
    coding_systems_path: pathlib.Path | None = None
    coding_systems: list[str] | None = None  # e.g. ["ICD", "OPS", "RADLEX"] or ["ICD", "OPS"]
    coding_system_files: dict[str, str] | None = None  # optional name -> filename for custom systems
    whisper_mode: bool = False
    tts_language: str = "de"
    use_default_voice: bool = False
    local_llm: bool = False
    words_per_minute: int = 80
    chunking_mode: str = DEFAULT_CHUNKING_MODE
    max_tts_words: int | None = None
    tts_length_policy: str = DEFAULT_TTS_LENGTH_POLICY
    max_audio_duration_seconds: float = 30.0
    split_seed: int | None = None


tts_model = None
coding_tables = None


def _init_worker(
    data_dir: pathlib.Path,
    tts_language: str = "de",
    coding_systems: list[str] | None = None,
    coding_system_files: dict[str, str] | None = None,
):
    """Load the TTS model and coding tables once per worker process."""
    global tts_model, coding_tables
    tts_model = init_tts(use_german_patch=(tts_language == "de"))
    coding_tables = load_coding_tables(
        data_dir,
        system_names=coding_systems,
        system_files=coding_system_files,
    )


def _process_document(
    idx: int,
    model_name: str,
    llm_endpoint: str,
    llm_api_key: str | None,
    dataset_path: pathlib.Path,
    verbose: bool,
    tts_language: str = "de",
    use_default_voice: bool = False,
    words_per_minute: int = 80,
    chunking_mode: str = DEFAULT_CHUNKING_MODE,
    max_tts_words: int | None = None,
    tts_length_policy: str = DEFAULT_TTS_LENGTH_POLICY,
):
    """Generate one synthetic document, synthesize its chunks, and return metadata rows."""
    global tts_model, coding_tables

    try:
        client = OpenAI(base_url=llm_endpoint, api_key=llm_api_key or "-", timeout=600.0)

        random_displays, random_codes = generate_random_displays(coding_tables)
        synthetic_text = generate_synthetic_medical_text(
            client=client,
            model_name=model_name,
            displays=random_displays,
        )

        chunks = chunk_document(
            document=synthetic_text,
            words_per_30s=_words_per_30s(words_per_minute),
            language_code=tts_language,
            chunking_mode=chunking_mode,
        )
        results = []
        max_tts_word_budget = _max_tts_words(max_tts_words, words_per_minute)
        output_sentence_id = 0

        for chunk in chunks:
            try:
                speaker_file = random_speaker_selection(use_default_voice=use_default_voice)
                speaker_id = speaker_id_from_path(speaker_file)
                tts_segments = prepare_tts_segments(
                    client=client,
                    model_name=model_name,
                    label_chunk=chunk,
                    language_code=tts_language,
                    max_tts_words=max_tts_word_budget,
                    tts_length_policy=tts_length_policy,
                )

                for segment in tts_segments:
                    tts_chunk = segment["tts_chunk"]
                    label_chunk = segment["label_chunk"]
                    sentence_id = output_sentence_id
                    output_sentence_id += 1

                    audio_path = dataset_path / f"{idx}_sentence_{sentence_id}_{speaker_id}.wav"
                    audio_file = generate_tts_audio(
                        tts_model=tts_model,
                        text=tts_chunk,
                        output_path=str(audio_path),
                        language_code=tts_language,
                        reference_voice_path=str(speaker_file) if speaker_file else None,
                    )

                    audio_duration = get_audio_duration(audio_path)

                    if verbose:
                        LOGGER.info(
                            "[%s] Document %s Chunk %s | duration=%.2f | tts_words=%s",
                            current_process().name,
                            idx,
                            sentence_id,
                            audio_duration,
                            segment["tts_word_count"],
                        )

                    results.append(
                        {
                            "document_id": idx,
                            "patient_id": f"synthetic_patient_{idx}",
                            "tts_chunk": tts_chunk,
                            "label_chunk": label_chunk,
                            "speaker_id": speaker_id,
                            "duration_in_seconds": audio_duration,
                            "audio_path": audio_file,
                            "label_word_count": segment["label_word_count"],
                            "tts_word_count": segment["tts_word_count"],
                            "tts_expansion_ratio": segment["tts_expansion_ratio"],
                            "was_rechunked_after_tts_transform": segment[
                                "was_rechunked_after_tts_transform"
                            ],
                            "was_over_tts_word_budget": segment["was_over_tts_word_budget"],
                            "document_icd_codes": random_codes.get("ICD", []),
                            "document_ops_codes": random_codes.get("OPS", []),
                            "document_radlex_codes": random_codes.get("RADLEX", []),
                        }
                    )
            except Exception as exc:
                LOGGER.exception("Error processing chunk for document %s: %s", idx, exc)

        return results
    except Exception as exc:
        LOGGER.exception("Failed to process document %s: %s", idx, exc)
        return None


def _process_document_wrapper(args):
    (
        idx,
        model_name,
        llm_endpoint,
        llm_api_key,
        dataset_path,
        verbose,
        tts_language,
        use_default_voice,
        words_per_minute,
        chunking_mode,
        max_tts_words,
        tts_length_policy,
    ) = args
    return _process_document(
        idx,
        model_name,
        llm_endpoint,
        llm_api_key,
        dataset_path,
        verbose,
        tts_language,
        use_default_voice,
        words_per_minute,
        chunking_mode,
        max_tts_words,
        tts_length_policy,
    )


def _split_and_write(
    df: pd.DataFrame,
    csv_path: pathlib.Path,
    whisper_mode: bool,
    max_audio_duration_seconds: float,
    seed: int | None = None,
) -> None:
    """Write train/validation/test CSVs and a compact stats report."""
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
        df_before_filter = df_tmp
        duration_filter_seconds = max_audio_duration_seconds if whisper_mode and mode in {"train", "val"} else None
        if whisper_mode and mode in {"train", "val"} and "duration_in_seconds" in df_tmp.columns:
            df_tmp = df_tmp[df_tmp.duration_in_seconds <= max_audio_duration_seconds]
        out_file = csv_path / f"{mode}.csv"
        df_tmp.to_csv(out_file, index=False, sep=";")
        LOGGER.info("Saved %s split to %s", mode, out_file)

        stats_report["splits"][mode] = _compute_split_stats(
            df_tmp,
            df_before_filter=df_before_filter,
            duration_filter_seconds=duration_filter_seconds,
        )

    stats_path = csv_path / "stats.json"
    _write_stats(stats_path, stats_report)


def _compute_split_stats(
    df_split: pd.DataFrame,
    df_before_filter: pd.DataFrame | None = None,
    duration_filter_seconds: float | None = None,
) -> dict:
    """Summarize sample counts, durations, speakers, and transcript length."""
    df_before_filter = df_split if df_before_filter is None else df_before_filter
    stats = {
        "samples": int(len(df_split)),
        "samples_before_duration_filter": int(len(df_before_filter)),
        "samples_after_duration_filter": int(len(df_split)),
        "samples_removed_by_duration_filter": int(len(df_before_filter) - len(df_split)),
        "duration_filter_seconds": duration_filter_seconds,
        "documents": int(df_split.document_id.nunique()) if "document_id" in df_split.columns else 0,
    }

    if "duration_in_seconds" in df_before_filter.columns:
        stats["duration_seconds_before_filter"] = _duration_stats(df_before_filter)

    if "duration_in_seconds" in df_split.columns:
        stats["duration_seconds_after_filter"] = _duration_stats(df_split)
        stats["duration_seconds"] = stats["duration_seconds_after_filter"]

    if "speaker_id" in df_split.columns:
        stats["speaker_counts"] = df_split["speaker_id"].value_counts().to_dict()

    if "patient_id" in df_split.columns:
        stats["patients"] = int(df_split.patient_id.nunique())

    if "label_chunk" in df_split.columns and not df_split.empty:
        word_counts = (
            df_split["label_word_count"]
            if "label_word_count" in df_split.columns
            else df_split["label_chunk"].astype(str).apply(lambda x: len(x.split()))
        )
        stats["text_words"] = {
            "total": int(word_counts.sum()),
            "mean": float(word_counts.mean()),
            "max": int(word_counts.max()),
        }
        stats["label_words"] = _word_count_stats(word_counts)

    if "tts_chunk" in df_split.columns and not df_split.empty:
        tts_word_counts = (
            df_split["tts_word_count"]
            if "tts_word_count" in df_split.columns
            else df_split["tts_chunk"].astype(str).apply(lambda x: len(x.split()))
        )
        stats["tts_words"] = _word_count_stats(tts_word_counts)

    if "tts_expansion_ratio" in df_split.columns and not df_split.empty:
        ratios = df_split["tts_expansion_ratio"].astype(float)
        stats["tts_expansion_ratio"] = {
            "mean": float(ratios.mean()),
            "max": float(ratios.max()),
            "p50": float(ratios.quantile(0.5)),
            "p90": float(ratios.quantile(0.9)),
        }

    if "was_rechunked_after_tts_transform" in df_split.columns:
        stats["rechunked_after_tts_transform"] = int(df_split["was_rechunked_after_tts_transform"].sum())

    if "was_over_tts_word_budget" in df_split.columns:
        stats["over_tts_word_budget"] = int(df_split["was_over_tts_word_budget"].sum())

    return stats


def _duration_stats(df_split: pd.DataFrame) -> dict:
    """Summarize duration values for a split."""
    if df_split.empty:
        return {}
    durations = df_split["duration_in_seconds"].astype(float)
    return {
        "min": float(durations.min()),
        "max": float(durations.max()),
        "mean": float(durations.mean()),
        "p50": float(durations.quantile(0.5)),
        "p90": float(durations.quantile(0.9)),
    }


def _word_count_stats(word_counts: pd.Series) -> dict:
    """Summarize a word-count series."""
    word_counts = word_counts.astype(float)
    return {
        "min": int(word_counts.min()),
        "mean": float(word_counts.mean()),
        "max": int(word_counts.max()),
        "p50": float(word_counts.quantile(0.5)),
        "p90": float(word_counts.quantile(0.9)),
    }


def _write_stats(path: pathlib.Path, stats: dict) -> None:
    """Persist split statistics as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    LOGGER.info("Saved stats report to %s", path)


def validate_llm_config(config: LlmDatasetConfig) -> None:
    """Validate paths, LLM settings, coding tables, and voice-source configuration."""
    if config.num_docs <= 0:
        raise ValueError("num_docs must be > 0")
    if config.num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    if config.words_per_minute <= 0:
        raise ValueError("words_per_minute must be > 0")
    if config.chunking_mode not in CHUNKING_MODES:
        raise ValueError(f"chunking_mode must be one of {sorted(CHUNKING_MODES)}")
    if config.max_tts_words is not None and config.max_tts_words <= 0:
        raise ValueError("max_tts_words must be > 0")
    if config.tts_length_policy not in TTS_LENGTH_POLICIES:
        raise ValueError(f"tts_length_policy must be one of {sorted(TTS_LENGTH_POLICIES)}")
    if config.max_audio_duration_seconds <= 0:
        raise ValueError("max_audio_duration_seconds must be > 0")
    if not config.model_name:
        raise ValueError("model_name is required")
    if not config.llm_endpoint:
        raise ValueError("llm_endpoint is required")

    parsed = urlparse(config.llm_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"llm_endpoint must be a valid URL. Got: {config.llm_endpoint}")

    for path in [config.dataset_path, config.csv_path]:
        path.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=path, delete=True):
                pass
        except Exception as exc:
            raise PermissionError(f"Cannot write to {path}: {exc}") from exc

    parsed = urlparse(config.llm_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"llm_endpoint must be a valid URL. Got: {config.llm_endpoint}")

    if not config.local_llm and not config.llm_api_key:
        raise ValueError("llm_api_key is required unless --local-llm is set.")

    data_dir = config.coding_systems_path or (PROJECT_ROOT / "data" / "coding-systems")
    load_coding_tables(
        data_dir,
        system_names=config.coding_systems,
        system_files=config.coding_system_files,
    )

    if not config.use_default_voice:
        voices_path = pathlib.Path(os.getenv("SPEAKER_VOICES_PATH") or (PROJECT_ROOT / "speaker-voices"))
        if not voices_path.exists():
            raise FileNotFoundError(f"Speaker voices directory not found: {voices_path}")
        if not any(x.is_file() for x in voices_path.iterdir()):
            raise RuntimeError(f"No speaker voice files found in {voices_path}")


def generate_llm_dataset(config: LlmDatasetConfig) -> None:
    """Run the full LLM -> text normalization -> TTS dataset pipeline."""
    validate_llm_config(config)
    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    if config.seed is not None:
        random.seed(config.seed)

    config.dataset_path.mkdir(parents=True, exist_ok=True)
    config.csv_path.mkdir(parents=True, exist_ok=True)

    data_dir = config.coding_systems_path or (PROJECT_ROOT / "data" / "coding-systems")
    load_coding_tables(
        data_dir,
        system_names=config.coding_systems,
        system_files=config.coding_system_files,
    )

    results = []
    with Pool(
        processes=config.num_workers,
        initializer=_init_worker,
        initargs=(
            data_dir,
            config.tts_language,
            config.coding_systems,
            config.coding_system_files,
        ),
    ) as pool:
        args_iter = (
            (
                i,
                config.model_name,
                config.llm_endpoint,
                config.llm_api_key,
                config.dataset_path,
                config.verbose,
                config.tts_language,
                config.use_default_voice,
                config.words_per_minute,
                config.chunking_mode,
                config.max_tts_words,
                config.tts_length_policy,
            )
            for i in range(config.num_docs)
        )
        results_iter = pool.imap(_process_document_wrapper, args_iter)
        for result in tqdm(results_iter, total=config.num_docs):
            if result is not None:
                results.append(result)

    if not results:
        raise RuntimeError("No documents generated. Check logs for errors.")

    df = pd.concat([pd.DataFrame(x) for x in results if x is not None], axis=0)
    _split_and_write(
        df,
        config.csv_path,
        whisper_mode=config.whisper_mode,
        max_audio_duration_seconds=config.max_audio_duration_seconds,
        seed=config.split_seed if config.split_seed is not None else config.seed,
    )


def _parse_coding_systems(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _parse_coding_system_files(value: str | dict[str, str] | None) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _parse_bool(value: bool | str | None) -> bool:
    """Parse common env/CLI truthy values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_int(value: int | str | None) -> int | None:
    """Parse optional integer values from CLI/env/config input."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _words_per_30s(words_per_minute: int) -> int:
    """Convert a speaking-rate estimate into the chunker's 30-second budget."""
    return max(1, round(words_per_minute / 2))


def _max_tts_words(max_tts_words: int | None, words_per_minute: int) -> int:
    """Resolve the transformed-text word budget used before TTS synthesis."""
    return max_tts_words or _words_per_30s(words_per_minute)


def build_llm_config_from_env(
    num_docs: int | None = None,
    num_workers: int | None = None,
    dataset_path: str | None = None,
    csv_path: str | None = None,
    model_name: str | None = None,
    llm_endpoint: str | None = None,
    llm_api_key: str | None = None,
    gpu_id: str | None = None,
    seed: int | None = None,
    verbose: bool = False,
    coding_systems_path: str | None = None,
    coding_systems: str | list[str] | None = None,
    coding_system_files: str | dict[str, str] | None = None,
    whisper_mode: bool = False,
    tts_language: str | None = None,
    use_default_voice: bool | str | None = None,
    local_llm: bool | str | None = None,
    words_per_minute: int | str | None = None,
    chunking_mode: str | None = None,
    max_tts_words: int | str | None = None,
    tts_length_policy: str | None = None,
    max_audio_duration_seconds: float | str | None = None,
    split_seed: int | None = None,
) -> LlmDatasetConfig:
    """Build an LLM dataset config from CLI overrides, config files, and .env."""
    load_env()
    return LlmDatasetConfig(
        model_name=model_name or get_env("LLM_NAME", required=True),
        llm_endpoint=llm_endpoint or get_env("LLM_ENDPOINT", required=True),
        llm_api_key=llm_api_key or get_env("LLM_API_KEY", None),
        dataset_path=resolve_path(dataset_path or get_env("LLM_DATASET_PATH", required=True)),
        csv_path=resolve_path(csv_path or get_env("LLM_CSV_PATH", required=True)),
        num_docs=num_docs or int(get_env("NUM_SYNTHETIC_DOCS", 5000)),
        num_workers=num_workers or int(get_env("NUM_WORKERS", 4)),
        gpu_id=gpu_id or get_env("GPU_ID", None),
        seed=seed,
        verbose=verbose,
        coding_systems_path=resolve_path(coding_systems_path, None) if coding_systems_path else None,
        coding_systems=_parse_coding_systems(coding_systems or get_env("CODING_SYSTEMS", None)),
        coding_system_files=_parse_coding_system_files(
            coding_system_files or get_env("CODING_SYSTEM_FILES", None)
        ),
        whisper_mode=whisper_mode,
        tts_language=tts_language or get_env("TTS_LANGUAGE", "de"),
        use_default_voice=_parse_bool(
            use_default_voice if use_default_voice is not None else get_env("USE_DEFAULT_TTS_VOICE", None)
        ),
        local_llm=_parse_bool(local_llm if local_llm is not None else get_env("LOCAL_LLM", None)),
        words_per_minute=int(
            words_per_minute if words_per_minute is not None else get_env("WORDS_PER_MINUTE", 80)
        ),
        chunking_mode=chunking_mode or get_env("CHUNKING_MODE", DEFAULT_CHUNKING_MODE),
        max_tts_words=_parse_optional_int(
            max_tts_words if max_tts_words is not None else get_env("MAX_TTS_WORDS", None)
        ),
        tts_length_policy=tts_length_policy or get_env("TTS_LENGTH_POLICY", DEFAULT_TTS_LENGTH_POLICY),
        max_audio_duration_seconds=float(
            max_audio_duration_seconds
            if max_audio_duration_seconds is not None
            else get_env("MAX_AUDIO_DURATION_SECONDS", 30.0)
        ),
        split_seed=split_seed,
    )
