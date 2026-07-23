import chatterbox.tts as chatterbox_tts
from mumblemed.utils import speech


def test_patch_missing_watermarker(monkeypatch):
    monkeypatch.setattr(chatterbox_tts.perth, "PerthImplicitWatermarker", None)

    speech._patch_missing_watermarker()

    watermarker = chatterbox_tts.perth.PerthImplicitWatermarker()
    assert watermarker.apply_watermark("audio", sample_rate=16_000) == "audio"
