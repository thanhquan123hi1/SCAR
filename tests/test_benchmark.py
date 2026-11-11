"""Validate benchmark arithmetic coverage and portable CLI behavior."""
import json

import pytest
import torch

from training.benchmark import counted_flops, main
from training.models.modules.m2_plus import SpatialAwareCrossAttention
from training.models.modules.sspanet_formula import PaperFormulaSSPANet


def test_counter_includes_functional_strip_projections_and_attention_core():
    c, h, w = 4, 3, 5
    block = PaperFormulaSSPANet(c).eval()
    result = counted_flops(block, (torch.rand(1, c, h, w),))
    # channel 2C->C, two C->1 strip projections, style 2->1 and spatial 1->1.
    assert result["macs"] == 2 * c * c + c * (h + w) + 3 * h * w
    fusion = SpatialAwareCrossAttention(in_channels=32, latent_dim=16, out_channels=16, num_heads=4)
    n = 6
    result = counted_flops(fusion, tuple(torch.rand(1, 32, 2, 3) for _ in range(3)))
    conv_macs = 3 * 32 * 16 * n + 3 * 16 * 16 * n
    mha_macs = 2 * (4 * n * 16**2 + 2 * n**2 * 16)
    assert result["macs"] == conv_macs + mha_macs
    assert result["flops_by_op"]["aten::bmm"] > 0


def test_cpu_smoke_benchmarks_both_scopes_and_training_without_data(tmp_path):
    output = tmp_path / "benchmark.json"
    argv = ["--config", "testing", "--variants", "paper_formula_v1", "identity",
            "--device", "cpu", "--warmup", "1", "--iterations", "1", "--repeats", "1",
            "--train-batch-size", "1", "--output", str(output)]
    report = main(argv)
    assert len(report["results"]) == 4
    assert report["status"] == "complete"
    assert report["precision"] == "float32" and not report["tf32"] and not report["amp"]
    assert report["source"]["kind"] == "random_initialization"
    for row in report["results"]:
        assert row["inference"]["mean_ms"] > 0
        assert row["training_step"]["mean_ms"] > 0
        assert row["training_step"]["peak_allocated_mib"] is None
        assert row["model_config"]["sspanet_gate"] == row["variant"]
        assert row["arithmetic"]["output_shape"][0] == 1
    assert json.loads(output.read_text())["protocol"] == report["protocol"]
    with pytest.raises(FileExistsError):
        main(argv)


def test_a100_guard_rejects_cpu_before_measurement(tmp_path):
    with pytest.raises(RuntimeError, match="A100"):
        main(["--device", "cpu", "--require-a100", "--output", str(tmp_path / "unused.json")])


@pytest.mark.parametrize("flag,value", [("--img-size", "0"), ("--iterations", "0"),
                                       ("--warmup", "0"), ("--train-batch-size", "0")])
def test_invalid_benchmark_dimensions_are_rejected(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        main(["--device", "cpu", "--output", str(tmp_path / "unused.json"), flag, value])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_cuda_cli_resolves_default_device_and_times_optimizer_steps(tmp_path):
    report = main(["--config", "testing", "--scope", "module", "--variants", "paper_formula_v1", "identity",
                   "--device", "cuda", "--warmup", "1", "--iterations", "1", "--repeats", "1",
                   "--train-batch-size", "1", "--output", str(tmp_path / "cuda.json")])
    assert report["status"] == "complete" and report["device"].startswith("cuda:")
    assert report["timing_method"] == "cuda_events"
    assert report["gpu_properties"]["total_memory_bytes"] > 0
    for row in report["results"]:
        assert row["training_step"]["mean_ms"] > 0
        assert row["training_step"]["peak_allocated_mib"] > 0


def test_checkpoint_source_preserves_saved_architecture_and_rejects_gate_change(tmp_path):
    from training.dataset.data_contract import CLASS_NAMES
    from training.models import M2PlusNet, get_testing
    config = get_testing()
    config.sspanet_gate = "paper_formula_v1"
    model = M2PlusNet(config, img_size=32)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(format_version=1, class_names=CLASS_NAMES, model_config=model.config.to_dict(),
                    model=model.state_dict(), args={"img_size": 32}), checkpoint)
    argv = ["--checkpoint", str(checkpoint), "--scope", "module", "--device", "cpu",
            "--skip-training", "--warmup", "1", "--iterations", "1", "--repeats", "1",
            "--output", str(tmp_path / "saved.json")]
    report = main(argv)
    assert report["source"]["kind"] == "checkpoint"
    assert report["results"][0]["model_config"] == model.config.to_dict()
    assert report["results"][0]["params_total"] == sum(p.numel() for p in model.sspanet_cine.parameters())
    argv[-1] = str(tmp_path / "invalid.json")
    with pytest.raises(SystemExit):
        main(argv + ["--variants", "legacy_v1"])
