import logging
import os
from pathlib import Path
import random


import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv

from mumblemed.paths import PROJECT_ROOT

load_dotenv()

LOGGER = logging.getLogger(__name__)

MODEL_REPO = os.getenv('MODEL_REPO')
T3_CHECKPOINT_FILE = os.getenv('T3_CHECKPOINT') 


class _NoOpWatermarker:
    """Fallback for environments where Chatterbox's optional watermarker is unavailable."""

    def apply_watermark(self, wav, sample_rate: int):
        return wav


def _patch_missing_watermarker() -> None:
    """Let Chatterbox initialize when the optional perth watermarker is missing."""
    try:
        import chatterbox.tts as chatterbox_tts
    except Exception:
        return

    watermarker = getattr(chatterbox_tts.perth, "PerthImplicitWatermarker", None)
    if watermarker is not None:
        return

    LOGGER.warning(
        "Chatterbox watermarking backend is unavailable; generated audio will be written without watermarking."
    )
    chatterbox_tts.perth.PerthImplicitWatermarker = _NoOpWatermarker


def init_tts(use_gpu: bool = True, use_german_patch: bool = True):
    """Initialize Chatterbox TTS and optionally apply the German checkpoint patch."""
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    LOGGER.info("Using device: %s", device)

    _patch_missing_watermarker()
    model = ChatterboxTTS.from_pretrained(device=device)

    if use_german_patch:
        if not MODEL_REPO or not T3_CHECKPOINT_FILE:
            raise ValueError(
                "MODEL_REPO and T3_CHECKPOINT must be set in the environment for German TTS. "
                "For non-German (e.g. English), use --tts-language en to use the default Chatterbox model."
            )
        LOGGER.info("Downloading and applying German patch.")
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPO, filename=T3_CHECKPOINT_FILE, token=os.getenv("HF_TOKEN")
        )
        t3_state = load_file(checkpoint_path, device="cpu")
        model.t3.load_state_dict(t3_state)
        LOGGER.info("German TTS patch applied successfully.")
    else:
        LOGGER.info("Using default Chatterbox TTS (no German patch).")

    return model


def generate_tts_audio(
    tts_model,
    text: str,
    output_path: str,
    language_code: str = "de",
    reference_voice_path: str | None = None,
) -> str:
    """Synthesize one text chunk with either a reference voice or the default model voice."""
    LOGGER.info("Generating speech for language: %s", language_code)
    with torch.inference_mode():

        if reference_voice_path is None:
            wav = tts_model.generate(
                text,
                exaggeration=0.3,
                temperature=0.3,
                cfg_weight=0.3,
            )
        else:
            wav = tts_model.generate(
                text,
                audio_prompt_path=reference_voice_path,
                exaggeration=0.3,
                temperature=0.3,
                cfg_weight=0.3,
            )

    sf.write(output_path, wav.squeeze().cpu().numpy(), tts_model.sr)
    LOGGER.info("Audio saved to %s", output_path)
    return output_path


def random_speaker_selection(use_default_voice: bool = False):
    """Return a random reference clip path, or None for default-voice synthesis."""
    if use_default_voice:
        return None

    voices_path = Path(os.getenv("SPEAKER_VOICES_PATH") or (PROJECT_ROOT / "speaker-voices"))
    if not voices_path.exists():
        raise FileNotFoundError(
            f"Speaker voices directory not found: {voices_path}. "
            "Set SPEAKER_VOICES_PATH or place voices under speaker-voices/."
        )
    speaker_list = [x for x in voices_path.iterdir() if x.is_file()]
    if not speaker_list:
        raise RuntimeError(f"No speaker voice files found in {voices_path}.")
    return random.choice(speaker_list)


def speaker_id_from_path(speaker_file: Path | None) -> str:
    """Create the CSV speaker id from a reference filename or the default voice marker."""
    if speaker_file is None:
        return "default"
    return speaker_file.parts[-1].split('.')[0].split('_')[0]
