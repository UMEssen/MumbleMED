import argparse

from mumblemed.datasets.real import build_real_config_from_env, generate_real_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS data from real CSV reports.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-csv", required=True)
    args = parser.parse_args()

    config = build_real_config_from_env(name=args.name, input_csv=args.input_csv)
    generate_real_dataset(config)
