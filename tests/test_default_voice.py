from pathlib import Path

import pytest

from mumblemed.datasets.llm import LlmDatasetConfig, build_llm_config_from_env, validate_llm_config
from mumblemed.datasets.real import RealDatasetConfig, validate_real_config


def test_llm_validation_allows_default_voice_without_speaker_dir(
    tmp_path: Path,
    example_coding_dir: Path,
    monkeypatch,
):
    monkeypatch.delenv("SPEAKER_VOICES_PATH", raising=False)

    validate_llm_config(
        LlmDatasetConfig(
            model_name="demo-model",
            llm_endpoint="http://127.0.0.1:8000/v1",
            llm_api_key=None,
            dataset_path=tmp_path / "audio",
            csv_path=tmp_path / "csv",
            num_docs=1,
            num_workers=1,
            coding_systems_path=example_coding_dir,
            use_default_voice=True,
            local_llm=True,
        )
    )


def test_llm_validation_requires_local_flag_when_api_key_is_empty(
    tmp_path: Path,
    example_coding_dir: Path,
):
    with pytest.raises(ValueError, match="--local-llm"):
        validate_llm_config(
            LlmDatasetConfig(
                model_name="demo-model",
                llm_endpoint="http://10.99.0.230/v1",
                llm_api_key=None,
                dataset_path=tmp_path / "audio",
                csv_path=tmp_path / "csv",
                num_docs=1,
                num_workers=1,
                coding_systems_path=example_coding_dir,
                use_default_voice=True,
            )
        )


def test_llm_validation_allows_hosted_endpoint_with_api_key(
    tmp_path: Path,
    example_coding_dir: Path,
):
    validate_llm_config(
        LlmDatasetConfig(
            model_name="gpt-4.1-mini",
            llm_endpoint="https://api.openai.com/v1",
            llm_api_key="test-token",
            dataset_path=tmp_path / "audio",
            csv_path=tmp_path / "csv",
            num_docs=1,
            num_workers=1,
            coding_systems_path=example_coding_dir,
            use_default_voice=True,
            local_llm=False,
        )
    )


def test_real_validation_allows_default_voice_without_speaker_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPEAKER_VOICES_PATH", raising=False)
    input_csv = tmp_path / "reports.csv"
    input_csv.write_text("report_clear\nNo acute cardiopulmonary abnormality.\n", encoding="utf-8")

    validate_real_config(
        RealDatasetConfig(
            model_name="demo-model",
            llm_endpoint="http://127.0.0.1:8000/v1",
            llm_api_key=None,
            dataset_path=tmp_path / "real-audio",
            input_csv=input_csv,
            name="demo",
            num_docs=1,
            num_workers=1,
            use_default_voice=True,
            local_llm=True,
        )
    )


def test_real_validation_rejects_empty_patient_id(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPEAKER_VOICES_PATH", raising=False)
    input_csv = tmp_path / "reports.csv"
    input_csv.write_text(
        "patient_id,report_clear\np1,No acute cardiopulmonary abnormality.\n,Follow-up report.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="patient_id"):
        validate_real_config(
            RealDatasetConfig(
                model_name="demo-model",
                llm_endpoint="http://127.0.0.1:8000/v1",
                llm_api_key=None,
                dataset_path=tmp_path / "real-audio",
                input_csv=input_csv,
                name="demo",
                num_docs=2,
                num_workers=1,
                use_default_voice=True,
                local_llm=True,
            )
        )


def test_llm_validation_rejects_invalid_words_per_minute(
    tmp_path: Path,
    example_coding_dir: Path,
):
    with pytest.raises(ValueError, match="words_per_minute"):
        validate_llm_config(
            LlmDatasetConfig(
                model_name="demo-model",
                llm_endpoint="http://127.0.0.1:8000/v1",
                llm_api_key=None,
                dataset_path=tmp_path / "audio",
                csv_path=tmp_path / "csv",
                num_docs=1,
                num_workers=1,
                coding_systems_path=example_coding_dir,
                use_default_voice=True,
                local_llm=True,
                words_per_minute=0,
            )
        )


def test_llm_validation_rejects_invalid_chunking_mode(
    tmp_path: Path,
    example_coding_dir: Path,
):
    with pytest.raises(ValueError, match="chunking_mode"):
        validate_llm_config(
            LlmDatasetConfig(
                model_name="demo-model",
                llm_endpoint="http://127.0.0.1:8000/v1",
                llm_api_key=None,
                dataset_path=tmp_path / "audio",
                csv_path=tmp_path / "csv",
                num_docs=1,
                num_workers=1,
                coding_systems_path=example_coding_dir,
                use_default_voice=True,
                local_llm=True,
                chunking_mode="creative-chaos",
            )
        )


def test_llm_config_reads_words_per_minute_from_env(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("LLM_NAME", "demo-model")
    monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_DATASET_PATH", str(tmp_path / "audio"))
    monkeypatch.setenv("LLM_CSV_PATH", str(tmp_path / "csv"))
    monkeypatch.setenv("WORDS_PER_MINUTE", "120")

    config = build_llm_config_from_env(
        num_docs=1,
        num_workers=1,
        use_default_voice=True,
        local_llm=True,
        coding_systems_path=str(tmp_path),
    )

    assert config.words_per_minute == 120


def test_llm_config_reads_chunking_mode_from_env(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("LLM_NAME", "demo-model")
    monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_DATASET_PATH", str(tmp_path / "audio"))
    monkeypatch.setenv("LLM_CSV_PATH", str(tmp_path / "csv"))
    monkeypatch.setenv("CHUNKING_MODE", "word-divide")

    config = build_llm_config_from_env(
        num_docs=1,
        num_workers=1,
        use_default_voice=True,
        local_llm=True,
        coding_systems_path=str(tmp_path),
    )

    assert config.chunking_mode == "word-divide"


def test_llm_config_preserves_invalid_words_per_minute_override(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("LLM_NAME", "demo-model")
    monkeypatch.setenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_DATASET_PATH", str(tmp_path / "audio"))
    monkeypatch.setenv("LLM_CSV_PATH", str(tmp_path / "csv"))

    config = build_llm_config_from_env(
        num_docs=1,
        num_workers=1,
        use_default_voice=True,
        local_llm=True,
        words_per_minute=0,
    )

    assert config.words_per_minute == 0
