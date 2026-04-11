from pathlib import Path

from src.omni.gemma_provider import GemmaProvider


class _FakeTorch:
    bfloat16 = "bf16"


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_gemma_quantized_load_kwargs_enable_cpu_offload(tmp_path: Path):
    provider = GemmaProvider(
        cpu_offload=True,
        offload_dir=str(tmp_path / "gemma-offload"),
    )

    load_kwargs = provider._build_quantized_load_kwargs(
        _FakeTorch,
        bitsandbytes_config_cls=_FakeBitsAndBytesConfig,
    )

    quant_cfg = load_kwargs["quantization_config"]
    assert quant_cfg.kwargs["load_in_4bit"] is True
    assert quant_cfg.kwargs["llm_int8_enable_fp32_cpu_offload"] is True
    assert load_kwargs["device_map"] == "auto"
    assert load_kwargs["offload_folder"] == str(tmp_path / "gemma-offload")
    assert (tmp_path / "gemma-offload").exists()


def test_gemma_quantized_load_kwargs_skip_offload_folder_when_disabled(tmp_path: Path):
    provider = GemmaProvider(
        cpu_offload=False,
        offload_dir=str(tmp_path / "unused-offload"),
    )

    load_kwargs = provider._build_quantized_load_kwargs(
        _FakeTorch,
        bitsandbytes_config_cls=_FakeBitsAndBytesConfig,
    )

    quant_cfg = load_kwargs["quantization_config"]
    assert quant_cfg.kwargs["llm_int8_enable_fp32_cpu_offload"] is False
    assert "offload_folder" not in load_kwargs
