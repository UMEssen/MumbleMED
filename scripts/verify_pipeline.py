#!/usr/bin/env python3
import argparse
import logging

from mumblemed.cli import _setup_logging
from mumblemed.datasets.llm import build_llm_config_from_env, generate_llm_dataset, validate_llm_config
from mumblemed.datasets.real import build_real_config_from_env, generate_real_dataset, validate_real_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MumbleMED pipeline configuration.")
    parser.add_argument("--mode", choices=["llm", "real"], required=True)
    parser.add_argument("--name", help="Required for --mode real.")
    parser.add_argument("--input-csv", help="Required for --mode real.")
    parser.add_argument("--whisper", action="store_true", help="Filter train/val samples to max 30s.")
    parser.add_argument("--use-default-voice", action="store_true", help="Validate/run without reference voice clips.")
    parser.add_argument("--local-llm", action="store_true", help="Allow an empty API key for a local/self-hosted LLM.")
    parser.add_argument("--words-per-minute", type=int, help="Estimated speaking rate used for chunk sizing.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible splitting.")
    parser.add_argument("--run", action="store_true", help="Run a minimal 1-doc generation (requires full setup).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.mode == "llm":
        config = build_llm_config_from_env(
            num_docs=1,
            num_workers=1,
            verbose=args.verbose,
            whisper_mode=args.whisper,
            use_default_voice=args.use_default_voice,
            local_llm=args.local_llm,
            words_per_minute=args.words_per_minute,
            split_seed=args.seed,
        )
        validate_llm_config(config)
        logging.warning("LLM config check passed.")
        if args.run:
            generate_llm_dataset(config)
        return

    if args.mode == "real":
        if not args.name or not args.input_csv:
            parser.error("--name and --input-csv are required for --mode real")
        config = build_real_config_from_env(
            name=args.name,
            input_csv=args.input_csv,
            num_docs=1,
            num_workers=1,
            verbose=args.verbose,
            whisper_mode=args.whisper,
            use_default_voice=args.use_default_voice,
            local_llm=args.local_llm,
            words_per_minute=args.words_per_minute,
            seed=args.seed,
        )
        validate_real_config(config)
        logging.warning("Real-data config check passed.")
        if args.run:
            generate_real_dataset(config)
        return


if __name__ == "__main__":
    main()
