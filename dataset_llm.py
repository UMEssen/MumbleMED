from mumblemed.datasets.llm import build_llm_config_from_env, generate_llm_dataset


if __name__ == "__main__":
    config = build_llm_config_from_env()
    generate_llm_dataset(config)
