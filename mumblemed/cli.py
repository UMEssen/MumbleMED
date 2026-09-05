import argparse
import logging
import pathlib
from typing import Any

from mumblemed.config import load_config
from mumblemed.datasets.demo import DemoDatasetConfig, generate_demo_dataset, validate_demo_config


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    colors = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    reset = "\033[0m"

    class ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            levelname = record.levelname
            record.levelname = f"{levelname:<8}"
            base = super().format(record)
            color = colors.get(levelname, "")
            return f"{color}{base}{reset}" if color else base

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


def _load_config_dict(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return load_config(path)


def _merge_config(args, config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    if "llm-api-key" in merged and "llm_api_key" not in merged:
        merged["llm_api_key"] = merged["llm-api-key"]
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mumblemed",
        description="MumbleMED dataset generation pipeline (LLM + TTS).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--config", type=str, help="Path to a YAML or JSON config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Create a minimal local 10-sample dataset from a bundled synthetic document.",
    )
    demo_parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of demo samples to create (default: 10).",
    )
    demo_parser.add_argument(
        "--dataset-path",
        type=str,
        default="./examples/demo-output/audio",
        help="Output path for demo WAV files.",
    )
    demo_parser.add_argument(
        "--csv-path",
        type=str,
        default="./examples/demo-output/csv",
        help="Output path for demo train/val/test CSVs and stats.",
    )
    demo_parser.add_argument(
        "--audio-mode",
        choices=["tone", "tts"],
        default="tone",
        help="Demo audio backend: 'tone' for no-download placeholders or 'tts' for the TTS model default voice.",
    )
    demo_parser.add_argument(
        "--tts-language",
        type=str,
        default="en",
        help="TTS language for --audio-mode tts. Use 'de' for Kartoffelbox/German patch.",
    )
    demo_parser.add_argument("--dry-run", action="store_true", help="Validate demo paths without writing samples.")

    llm_parser = subparsers.add_parser("llm", help="Generate synthetic dataset using LLM + TTS.")
    llm_parser.add_argument("--num-docs", type=int, help="Number of synthetic documents to generate.")
    llm_parser.add_argument("--num-workers", type=int, help="Number of parallel workers.")
    llm_parser.add_argument("--dataset-path", type=str, help="Output path for generated audio.")
    llm_parser.add_argument("--csv-path", type=str, help="Output path for CSV splits.")
    llm_parser.add_argument("--model-name", type=str, help="LLM model name.")
    llm_parser.add_argument("--llm-endpoint", type=str, help="LLM endpoint base URL.")
    llm_parser.add_argument("--llm-api-key", type=str, help="API key for hosted LLM providers.")
    llm_parser.add_argument(
        "--local-llm",
        action="store_true",
        default=None,
        help="Mark the LLM endpoint as locally/self hosted, allowing an empty API key.",
    )
    llm_parser.add_argument("--gpu-id", type=str, help="CUDA device id.")
    llm_parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    llm_parser.add_argument("--coding-systems-path", type=str, help="Path to directory containing coding system CSVs.")
    llm_parser.add_argument(
        "--coding-systems",
        type=str,
        help="Comma-separated list of coding systems to use for generation (e.g. ICD,OPS,RADLEX or ICD,OPS). Default: all built-in.",
    )
    llm_parser.add_argument(
        "--coding-system-files",
        type=str,
        help='Optional JSON mapping name -> filename for custom systems (e.g. {"CUSTOM":"custom.csv"}).',
    )
    llm_parser.add_argument(
        "--tts-language",
        type=str,
        default="de",
        help="TTS language code: 'de' for German (Kartoffelbox patch), e.g. 'en' for default Chatterbox (English).",
    )
    llm_parser.add_argument(
        "--words-per-minute",
        type=int,
        help="Estimated speaking rate used for chunk sizing before TTS (default: 80, or WORDS_PER_MINUTE).",
    )
    llm_parser.add_argument(
        "--chunking-mode",
        choices=["sentence-strict", "sentence-divide", "word-divide"],
        help="Chunking strategy before TTS (default: sentence-divide, or CHUNKING_MODE).",
    )
    llm_parser.add_argument(
        "--max-tts-words",
        type=int,
        help="Maximum words allowed after TTS text normalization (default: derived from --words-per-minute).",
    )
    llm_parser.add_argument(
        "--tts-length-policy",
        choices=["warn", "rechunk", "skip"],
        help="How to handle chunks whose TTS-normalized text exceeds --max-tts-words (default: rechunk).",
    )
    llm_parser.add_argument(
        "--max-audio-duration-seconds",
        type=float,
        help="Maximum generated audio duration retained by --whisper filtering (default: 30.0).",
    )
    llm_parser.add_argument(
        "--whisper",
        action="store_true",
        help="Filter train/val samples to the configured max duration for Whisper fine-tuning.",
    )
    llm_parser.add_argument(
        "--use-default-voice",
        action="store_true",
        default=None,
        help="Use the TTS model's default voice instead of SPEAKER_VOICES_PATH reference clips.",
    )
    llm_parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs without generating data.")

    real_parser = subparsers.add_parser(
        "real",
        help="Generate dataset from real documents (CSV): TTS turns report text into audio, then train/val/test splits.",
    )
    real_parser.add_argument("--name", type=str, required=True, help="Name for this dataset run (used in output filenames).")
    real_parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to CSV of real document texts (must have report_clear column). Audio is TTS-generated from this text.",
    )
    real_parser.add_argument("--num-docs", type=int, help="Number of real documents (rows) to process.")
    real_parser.add_argument("--num-workers", type=int, help="Number of parallel workers.")
    real_parser.add_argument("--dataset-path", type=str, help="Output path for TTS-generated audio (from real document text).")
    real_parser.add_argument("--model-name", type=str, help="LLM model name.")
    real_parser.add_argument("--llm-endpoint", type=str, help="LLM endpoint base URL.")
    real_parser.add_argument("--llm-api-key", type=str, help="API key for hosted LLM providers.")
    real_parser.add_argument(
        "--local-llm",
        action="store_true",
        default=None,
        help="Mark the LLM endpoint as locally/self hosted, allowing an empty API key.",
    )
    real_parser.add_argument("--gpu-id", type=str, help="CUDA device id.")
    real_parser.add_argument("--seed", type=int, help="Random seed for reproducible patient/document splits.")
    real_parser.add_argument(
        "--tts-language",
        type=str,
        default="de",
        help="TTS language code: 'de' for German (Kartoffelbox patch), e.g. 'en' for default Chatterbox (English).",
    )
    real_parser.add_argument(
        "--words-per-minute",
        type=int,
        help="Estimated speaking rate used for chunk sizing before TTS (default: 80, or WORDS_PER_MINUTE).",
    )
    real_parser.add_argument(
        "--chunking-mode",
        choices=["sentence-strict", "sentence-divide", "word-divide"],
        help="Chunking strategy before TTS (default: sentence-divide, or CHUNKING_MODE).",
    )
    real_parser.add_argument(
        "--max-tts-words",
        type=int,
        help="Maximum words allowed after TTS text normalization (default: derived from --words-per-minute).",
    )
    real_parser.add_argument(
        "--tts-length-policy",
        choices=["warn", "rechunk", "skip"],
        help="How to handle chunks whose TTS-normalized text exceeds --max-tts-words (default: rechunk).",
    )
    real_parser.add_argument(
        "--max-audio-duration-seconds",
        type=float,
        help="Maximum generated audio duration retained by --whisper filtering (default: 30.0).",
    )
    real_parser.add_argument(
        "--whisper",
        action="store_true",
        help="Filter train/val samples to the configured max duration for Whisper fine-tuning.",
    )
    real_parser.add_argument(
        "--use-default-voice",
        action="store_true",
        default=None,
        help="Use the TTS model's default voice instead of SPEAKER_VOICES_PATH reference clips.",
    )
    real_parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs without generating data.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    config_dict = _load_config_dict(args.config)

    if args.command == "demo":
        merged = _merge_config(args, config_dict)
        config = DemoDatasetConfig(
            dataset_path=pathlib.Path(merged.get("dataset_path")).expanduser().resolve(),
            csv_path=pathlib.Path(merged.get("csv_path")).expanduser().resolve(),
            num_samples=merged.get("num_samples", 10),
            audio_mode=merged.get("audio_mode", "tone"),
            tts_language=merged.get("tts_language", "en"),
        )
        if args.dry_run:
            validate_demo_config(config)
            logging.info("Demo dry-run passed. Output paths are writable.")
            return
        generate_demo_dataset(config)
        return

    if args.command == "llm":
        from mumblemed.datasets.llm import (
            build_llm_config_from_env,
            generate_llm_dataset,
            validate_llm_config,
        )

        merged = _merge_config(args, config_dict)
        config = build_llm_config_from_env(
            num_docs=merged.get("num_docs"),
            num_workers=merged.get("num_workers"),
            dataset_path=merged.get("dataset_path"),
            csv_path=merged.get("csv_path"),
            model_name=merged.get("model_name"),
            llm_endpoint=merged.get("llm_endpoint"),
            llm_api_key=merged.get("llm_api_key"),
            local_llm=merged.get("local_llm", False),
            gpu_id=merged.get("gpu_id"),
            seed=merged.get("seed"),
            verbose=merged.get("verbose", False),
            coding_systems_path=merged.get("coding_systems_path"),
            coding_systems=merged.get("coding_systems"),
            coding_system_files=merged.get("coding_system_files"),
            whisper_mode=merged.get("whisper", False),
            tts_language=merged.get("tts_language"),
            use_default_voice=merged.get("use_default_voice", False),
            words_per_minute=merged.get("words_per_minute"),
            chunking_mode=merged.get("chunking_mode"),
            max_tts_words=merged.get("max_tts_words"),
            tts_length_policy=merged.get("tts_length_policy"),
            max_audio_duration_seconds=merged.get("max_audio_duration_seconds"),
        )
        if args.dry_run:
            validate_llm_config(config)
            logging.warning("LLM dry-run passed. Configuration and inputs look valid.")
            return
        generate_llm_dataset(config)
        return

    if args.command == "real":
        from mumblemed.datasets.real import (
            build_real_config_from_env,
            generate_real_dataset,
            validate_real_config,
        )

        merged = _merge_config(args, config_dict)
        config = build_real_config_from_env(
            name=merged.get("name"),
            input_csv=merged.get("input_csv"),
            num_docs=merged.get("num_docs"),
            num_workers=merged.get("num_workers"),
            dataset_path=merged.get("dataset_path"),
            model_name=merged.get("model_name"),
            llm_endpoint=merged.get("llm_endpoint"),
            llm_api_key=merged.get("llm_api_key"),
            local_llm=merged.get("local_llm", False),
            gpu_id=merged.get("gpu_id"),
            seed=merged.get("seed"),
            verbose=merged.get("verbose", False),
            whisper_mode=merged.get("whisper", False),
            tts_language=merged.get("tts_language"),
            use_default_voice=merged.get("use_default_voice", False),
            words_per_minute=merged.get("words_per_minute"),
            chunking_mode=merged.get("chunking_mode"),
            max_tts_words=merged.get("max_tts_words"),
            tts_length_policy=merged.get("tts_length_policy"),
            max_audio_duration_seconds=merged.get("max_audio_duration_seconds"),
        )
        if args.dry_run:
            validate_real_config(config)
            logging.warning("Real-data dry-run passed. Configuration and inputs look valid.")
            return
        generate_real_dataset(config)
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
