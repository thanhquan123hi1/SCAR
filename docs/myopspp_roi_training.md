# Train the MyoPS++ ROI datasets

The repository supports `myops380`, native `myopspp_bc80`, and two independent exported ROI profiles. Model architecture files and MyoPS380 defaults are unchanged.

| Profile | Patients (train / val / test) | Export folder | Cache | Evaluation XY spacing |
|---|---|---|---|---|
| `myopspp_roi128_76` | 48 / 13 / 15 | `roi128mm_76cases` | `data/myopspp_roi128_76/cache` | 1.0 mm |
| `myopspp_roi160_80` | 51 / 13 / 16 | `roi160mm_to128_80cases` | `data/myopspp_roi160_80/cache` | 1.25 mm |

Both caches contain 128×128 slices; Z spacing is retained per patient. The 76-case cohort excludes Case2013, Case2017, Case2018 and Case2031 and filters the original fixed split without moving retained patients. A model's random seed never regenerates these splits.

## Prepare a cache on another computer

Extract the existing dataset archive first. The adapter requires the export manifest, splits, per-case metadata and `checksums.sha256`. It validates checksums, geometry, cohort and split before publishing a new cache. It refuses any existing destination. MRI values are copied exactly as float32; no second normalization, crop or resampling occurs. Raw labels are mapped once: 0/500/600→background, 200→normal myocardium, 1220→edema, 2221→scar.

Run from the repository root, substituting the export path on that computer:

```powershell
python -m preprocessing.myopspp_roi --dataset myopspp_roi128_76 --src-path D:/NCKH/MyoPSpp_preprocessed_2026-10-03/roi128mm_76cases --dst-path data/myopspp_roi128_76/cache
python -m preprocessing.myopspp_roi --dataset myopspp_roi160_80 --src-path D:/NCKH/MyoPSpp_preprocessed_2026-10-03/roi160mm_to128_80cases --dst-path data/myopspp_roi160_80/cache
```

The caches have already been created on this computer. Do not run those commands against the existing destinations. Data/cache files and training outputs are ignored by Git; pulling the repository on another machine does not download the datasets.

## Train and evaluate

`run_all.py` reuses a cache with `--skip-cache`, verifies it, trains with validation-based checkpoint selection, and evaluates the best checkpoint on held-out patient volumes:

```powershell
python run_all.py --dataset myopspp_roi128_76 --config training/config/models/cmspa_net.yaml --skip-cache --batch-size 2 --accum-steps 8 --run-id roi128_m3_seed1234
python run_all.py --dataset myopspp_roi160_80 --config training/config/models/cmspa_net.yaml --skip-cache --batch-size 2 --accum-steps 8 --run-id roi160_m3_seed1234
```

Use an existing model YAML instead of `cmspa_net.yaml` to select another model. The default input size is 128×128. `testing.yaml` intentionally uses a smaller network and 32×32 model input for software smoke tests and should not be used for reported research experiments. Batch size can be lowered with `--batch-size`; use `--accum-steps` to preserve the desired effective batch size. GPU memory needs depend on the selected model.

For example, train V7 using its existing model configuration:

```powershell
python run_all.py --dataset myopspp_roi160_80 --config training/config/models/m2_max_v7.yaml --skip-cache --batch-size 2 --accum-steps 8 --run-id roi160_v7_seed1234
```

Auxiliary loss weights continue to come from the chosen model YAML or train CLI; selecting an ROI profile does not enable auxiliary losses. For V6/V7, wall/inclusion loss may be enabled explicitly. The new ROI profiles default to `loss.pathology_reduction: sample`: conditional AAR/scar Dice contributes zero for negative slices and is averaged over all samples, so sample-weighted gradient accumulation preserves the objective. Existing MyoPS380 and native MyoPS++ runs keep `positive` reduction by default. BatchNorm and stochastic operations can still make microbatch training differ from a large physical batch. This change only fixes the loss denominator.

To change that choice explicitly, set `loss.pathology_reduction` in a custom model YAML, or pass `--pathology-reduction` to `training/train.py`. Resume locks that setting; a checkpoint cannot silently switch modes. `--init-weights` transfers compatible canonical model weights into a new experiment without carrying optimizer state, old splits or scores. Hierarchical base classifiers transfer to and from ordinary classifiers; additional anatomy heads remain architecture-specific.

Resume with the same dataset, model configuration and loss settings:

```powershell
python training/train.py --dataset myopspp_roi160_80 --config training/config/models/cmspa_net.yaml --batch-size 2 --accum-steps 8 --resume outputs/myopspp_roi160_80/runs/roi160_m3_seed1234/checkpoints/last.pth --output-dir outputs/myopspp_roi160_80/runs/roi160_m3_seed1234
python training/evaluate.py --checkpoint outputs/myopspp_roi160_80/runs/roi160_m3_seed1234/checkpoints/best.pth --data-root data/myopspp_roi160_80/cache
```

Dataset identity, fixed split IDs, cache-byte fingerprint, class semantics and metric protocol are stored in checkpoints and checked on resume/evaluation. Do not edit caches in place between runs.

## Evaluation protocol

These datasets use **GT-centered oracle ROIs**, including validation/test. Predictions are evaluated against the exported ROI GT with equal patient weight; Dice is dimensionless and HD95/ASD use the exported NIfTI spacing in mm (including native Z spacing). Reports record `evaluation_grid: preprocessed_roi` and `localization: oracle_gt_bbox`. Prediction NIfTI files use the ROI affine. They are not restored to the original whole-image native grid.

This measures segmentation given a cardiac ROI. It does not measure automatic heart localization or performance on uncropped MRI. `training/predict.py` rejects these checkpoints on its raw-NIfTI path to prevent silently using the wrong preprocessing. Use the ROI cache evaluator for this protocol. Native restoration and a deployable localizer are separate work.

MyoPS380 retains its existing voxel-distance benchmark, label-order conversion, split and default loss behavior. Its commands remain unchanged, or choose it explicitly with `--dataset myops380`. Neither ROI cohort is the published 95-patient MyoPS++ cohort, and their results should be named separately in comparisons.
