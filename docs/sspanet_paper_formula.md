# M2-Plus with SSPANet paper-formula attention

`paper_formula_v1` is SCAR's explicit interpretation of equations (17)-(23),
section 3.5, pages 6-7 of Hasan et al., *Enhancing brain tumor classification
with a novel attention based explainable deep learning framework*, BSPC 112
(2026), 108636, [DOI](https://doi.org/10.1016/j.bspc.2025.108636).

The paper's sequential equations disagree with its parallel computational graph
and [author demo](https://github.com/HelloJahid/SSPANet/blob/fb77a27adc9a089e2d147ed178378cbd641fdb88/sspanet_demo.py).
This version follows the equations. It is not claimed to reproduce the author's
training implementation or to improve Dice without experiments.

## Exact interpretation

For `X: B,C,H,W`:

1. Global mean and max over H,W are concatenated as `B,2C,1,1`.
2. One dense Conv1x1 `2C -> C` (no bias), BatchNorm, then one sigmoid gives
   `Mc: B,C,1,1`. Define `Xp = Mc * X`.
3. Pool `Xp` over H and W separately, producing `Sh: B,C,1,W` and
   `Sv: B,C,H,1`. Equation (21) is interpreted as a Conv1x1 `2C -> 1` on
   their concatenated, broadcast maps. The implementation projects each strip
   first with the corresponding half of the same kernel, then broadcasts/adds.
   The bias is included exactly once. This is algebraically equivalent and
   avoids materializing `B,2C,H,W`.
4. Equation (22) uses mean and population standard deviation over **C at each
   pixel of Xp**, giving `B,2,H,W`. A Conv1x1 `2 -> 1` with bias produces style
   logits. This is not RMS, nor per-channel statistics pooled over H,W.
5. Add strip/style logits; Conv1x1 `1 -> 1` with bias and one sigmoid give
   `Ms: B,1,H,W`. Return `Y = Ms * Xp`.

The choices of output widths for the projections, use of Xp for style, descriptor
concatenation and a single sigmoid follow the displayed equations where prose
is incomplete or contradictory. No residual, tanh, bottleneck, depthwise conv,
extra activation or extra normalization is added. `sspanet_reduction` does not
apply to this version. Epsilon is fixed to `1e-6`; squares/variance and pooling
statistics accumulate in FP32 under FP16/BF16 autocast. Normal FP64 is retained.
For a pooled training batch of size one, channel BatchNorm uses stored running
statistics without updating them. Batch sizes >=2 use ordinary BatchNorm.

All three modality bottlenecks receive independent blocks. Encoders, dropout,
cross-attention, skip fusion, decoder, labels, loss and inference are unchanged.

For the production config at input 128, the implemented counts are:

| Variant | One SSPANet block params | Full M2-Plus params | Full counted MACs |
|---|---:|---:|---:|
| `centered_tanh_v2` / SCAR `legacy_v1` | 7,345,252 | 63,748,336 | 6.886 G |
| `paper_formula_v1` | 2,101,254 | 48,016,342 | 6.691 G |
| `identity` | 0 | 41,712,580 | 6.685 G |

The paper block has `2*C*C + 4*C + 6` parameters. Its convolution MAC count for
batch one is `2*C*C + C*(H+W) + 3*H*W`. Full-model parameters decrease by about
24.7%, but the counted MACs decrease by only about 2.8%; measured A100 latency
may behave differently. These counts use the coverage defined below, not total
arithmetic including normalization/pooling. No A100 speedup is claimed here.

## Configurations and checkpoint identity

| Config (`--config`) | `model.sspanet_gate` | Role |
|---|---|---|
| `m2_plus` | `centered_tanh_v2` | Existing default, unchanged |
| `m2_plus_legacy_sspanet` | `legacy_v1` | Author-code formula control |
| `m2_plus_paper_formula` | `paper_formula_v1` | Sequential paper-equation interpretation |
| `m2_plus_no_sspanet` | `identity` | Unmodified bottleneck features |

These are full M2-Plus models, not the shared-encoder Lightweight architecture.
Lightweight continues to require `optimized` attention.

The version is recorded in `model_config.sspanet_gate`, already locked by resume.
Paper weights use distinct `sspanet_*.paper_formula.*` state keys. The loader
rejects missing/conflicting paper metadata; strict loading rejects mismatched
structures. Old checkpoints retain their original gate fallback. Start a new run
for a different variant, using the same intended encoder initialization. Do not
reinterpret an existing trained checkpoint by changing its gate. This patch also
changes source fingerprints, so existing locked no-val runs should be completed
with their original checkout rather than resumed across the source change.

## Train on A100

Run from the repository root. Replace dataset paths with locations on the A100
machine. The YAML inherits the existing 300-epoch, batch-16, accumulation-1,
0.2 CE + 0.8 batch-Dice recipe, including its existing AMP setting.

```bash
python train.py --config m2_plus_paper_formula --data-root /data/MyoPS380/Processed_data --label-order legacy --run-id myops380_paper_formula_seed1234 --seed 1234
python train.py --config m2_plus_paper_formula --dataset myopspp_unet2d_roi128_1mm --data-root /data/myopspp_roi/cache --list-dir /data/myopspp_roi/cache/lists --run-id myopspp_paper_formula_seed1234 --seed 1234
```

Use the same data, manifests, physical batch, initialization, precision, loss,
optimizer, schedule and seeds for controls. `--pretrained /path/R50-ViT-B_16.npz`
is optional and must be applied consistently. Initialization is random otherwise.
Use separate run IDs for each variant and seed. For MyoPS380 development,
`--with-val` selects the existing patient-level validation workflow. MyoPS++'s
current fixed ROI profile remains no-val; internal validation is a separate
workflow change, not implemented by these model configs. Freeze choices before
final test reporting. Evaluate each dataset separately using the existing evaluator.

The final paper-model checkpoint is `m2pluspaperformula_seed1234.pt` for seed 1234.
Resume with the same model YAML and original settings. For the strict FP32 training
protocol in `BENCHMARK_EFFICIENCY_PROTOCOL.txt`, use `--amp none --num-workers 0
--pin-memory` for every variant and explicitly disable both TF32 flags in the
training process. For example, this launcher passes remaining options to train:

```bash
python -c "import torch; from training.train import main; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; main()" --config m2_plus_paper_formula --data-root /data/MyoPS380/Processed_data --amp none --num-workers 0 --pin-memory --run-id paper_formula_fp32_seed1234
```

## Synthetic A100 benchmark

The benchmark requires no images or long training. Compare the same module and
full network, at the same precision and batch sizes, before investing in 300
epochs. The default is FP32, AMP off, TF32 off, deterministic cuDNN settings,
30 warmups, 100 iterations per repeat and three repeats. All scopes/variants run
sequentially. Use one idle A100 and record whether it is PCIe/SXM, its memory
capacity, and whether MIG is enabled alongside the output. CUDA events measure
GPU operation latency; the JSON records versions, architecture, shapes, source
hashes and individual repeat means. CPU mode exists only for software checks.

```bash
python -m training.benchmark --config m2_plus_paper_formula --variants centered_tanh_v2 legacy_v1 paper_formula_v1 identity --scope both --device cuda --require-a100 --batch-size 1 --train-batch-size 16 --warmup 30 --iterations 100 --repeats 3 --output outputs/benchmarks/a100_sspanet/efficiency_metrics.json
```

`scope=module` times the CINE block with `B,1024,8,8` features at input size 128.
`scope=model` times the full network with **three** `B,1,128,128` tensors. The
isolated module's training loss is output mean-square, including gradients to
the input (also for identity). Full-network training uses the selected recipe's
SegmentationLoss and uniform synthetic four-class targets. Training timing
includes forward/loss/backward/AdamW at constant LR, after optimizer state warmup;
it excludes scheduler, data loading, augmentation, logging and metrics. This is
not a time-per-epoch, total-300-epoch or accuracy result. The existing trainer's
CSV/JSONL contains real train seconds and peak memory for those measurements.

Inference memory is measured before allocating the training batch/optimizer.
Allocated and reserved peak memory use MiB; timing uses milliseconds. Reserved
memory reflects the caching allocator and is reported separately. An OOM fails
the benchmark; do not silently change batch sizes between models.

Arithmetic counts include Conv2d and matrix multiplications, including attention
projections, QK and AV. One MAC is two FLOPs. Pooling, mean/std, normalization,
softmax and elementwise work are **excluded**, and this coverage is recorded in
JSON. SDPA uses the math backend only during the counting pass so fused attention
does not disappear from the count; timed passes use normal backend dispatch.
These counts are not a latency prediction or an exhaustive operation count.

For an existing checkpoint, benchmark its saved architecture/weights exactly:

```bash
python -m training.benchmark --checkpoint /path/to/model.pt --device cuda --require-a100 --output outputs/benchmarks/checkpoint/efficiency_metrics.json
```

The script refuses to overwrite output unless `--overwrite` is explicit and
rejects changing the SSPANet variant when a checkpoint is supplied. Random
benchmarks do not load optional initialization paths from training YAML. Results
are labeled synthetic and contain no Dice or claimed speedup. The JSON is saved
after each completed scope; `status=complete` means all requested measurements
finished. A measurement failure is recorded as `status=failed` and re-raised.

## Validation

```bash
python -m pytest -q tests/test_sspanet_formula.py tests/test_benchmark.py
python -m pytest -q
```

Tests cover the explicit sequential equations, strip projection equivalence and
gradients, style-axis semantics, singleton/constant inputs, autocast, full-model
gradients, checkpoint rejection/roundtrip, arithmetic counting including SDPA,
and the existing ROI train-resume-evaluate integration for the new variant.
