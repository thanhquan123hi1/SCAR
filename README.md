# SCAR — M2-Plus

Phân đoạn cơ tim đa phương thức từ bSSFP/CINE, LGE/PSIR và T2w đã căn chỉnh. Pipeline hỗ trợ **M2-Plus** và **Lightweight M2-Plus** cho train, resume và đánh giá volume.

## Kiến trúc và nhãn

M2-Plus dùng ba encoder ResNetV2 độc lập, SSPANet trên từng modality, hai nhánh cross-attention riêng cho scar/edema, ba tầng skip fusion concat/project và decoder bốn cấp. CINE cung cấp queries; LGE và T2w cung cấp keys/values cho hai attention stream. Hai stream được kết hợp với đặc trưng CINE trước decoder.

Mã chính: [m2_plus_net.py](training/models/m2_plus_net.py), [m2_plus.py](training/models/modules/m2_plus.py). Input mặc định 128×128; model trả raw logits `(B,4,H,W)` trong cả train và eval. Không có anatomy head hay skip gate của các model nghiên cứu trước đây. File pretrained `R50-ViT-B_16.npz` chỉ khởi tạo phần ResNet tương thích; model không dùng ViT hoặc text encoder.

SSPANet mặc định dùng `model.sspanet_gate: centered_tanh_v2`: `y = x * (1 + 0.5 * tanh(g_channel(x) + g_spatial(x)))`. Hai nhánh cung cấp logits trước sigmoid và trước phép nhân với `x`; hệ số trong khoảng 0.5–1.5 cho phép giảm hoặc tăng đặc trưng, bằng 1 khi tổng logits bằng 0. Đây là biến thể SCAR của SSPANet; ảnh hưởng tới Dice cần được xác nhận bằng ablation.

Biến thể mới `--config m2_plus_paper_formula` dùng `paper_formula_v1`: channel attention rồi strip/style spatial attention theo công thức (17)–(23), `Y = Ms * (Mc * X)`. Đây là cách diễn giải được đặc tả rõ của SCAR vì công thức và code tác giả chưa nhất quán. Các config `m2_plus_legacy_sspanet` và `m2_plus_no_sspanet` cung cấp đối chứng. Xem [đặc tả, checkpoint và lệnh A100](docs/sspanet_paper_formula.md). Benchmark bằng `python -m training.benchmark`; số đo tổng hợp không thay thế Dice hoặc thời gian train trên dataset.

M2-Plus mặc định có khoảng 63.75M tham số. Bản Lightweight có khoảng 14.67M: shared trunk với FiLM, OptimizedSSPANet và skip gates. Chọn bằng `--config training/config/models/lightweight_m2_plus.yaml`; cấu hình này dùng chung pipeline train/resume/evaluate. Hai tên chỉ rõ biến thể; mức giảm tham số không tự chứng minh cải thiện độ chính xác.

| ID | Nhãn canonical |
|---:|---|
| 0 | background |
| 1 | normal myocardium |
| 2 | edema |
| 3 | scar |

## Cài đặt

Python 3.11 hoặc 3.12, PyTorch ≥2.3. Chạy lệnh từ root repo:

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Chuẩn bị dataset và pretrained weights riêng; Git không chứa chúng. Huấn luyện bằng `train.py` hoặc chạy pipeline bằng `run_all.py`.

## Dataset và split

**Mặc định MyoPS380: 304 train / 76 test, không validation, dùng `m2plus_seed1234.pt`.**
`base.yaml` đặt `train.no_val: true` và dùng manifest `preprocessing/splits/myops380_no_val`.
Xem [MYOPS380_NO_VAL.md](MYOPS380_NO_VAL.md); cấu hình riêng
`training/config/experiments/myops380_no_val.yaml` vẫn dùng được.
Evaluator từ chối checkpoint chưa hoàn thành số epoch đã định trước.
Muốn dùng protocol phát triển 243/61/76, thêm `--with-val`; MyoPS++ dự đoán ROI dùng 64 train / 16 test, không validation.

| Profile | Train / val / test (bệnh nhân) | Khoảng cách metric |
|---|---|---|
| `myops380` — mặc định | 304 / 0 / 76 | voxel |
| `myops380 --with-val` | 243 / 61 / 76 với bộ chia phát triển | voxel |
| `myopspp_unet2d_roi128_1mm` | 64 / 0 / 16 | mm, predicted ROI grid |

MyoPS++ dùng split 64/16 cố định từ export U-Net2D, đã version-control; nguồn `D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm`. MyoPS380 khóa 76 ca test trong [test_vol.txt](preprocessing/splits/test_vol.txt). Không chia lát cắt của cùng bệnh nhân sang nhiều tập.

Chỉ hai dataset ID được hỗ trợ: `myops380` và `myopspp_unet2d_roi128_1mm`. Bốn profile MyoPS++ native/oracle cũ đã được gỡ khỏi CLI và cấu hình.

| Dữ liệu | MyoPS380 | MyoPS++ ROI128 1 mm |
|---|---|---|
| Lát train / volume test | 1.972 / 76 | 296 / 16 |
| Cache khi train | Ba file `.npz` mỗi lát, theo thứ tự bSSFP/LGE/T2w | Cùng định dạng; C0/LGE/T2 nguồn tương ứng bSSFP/LGE/T2w |
| Cache khi test | Ba file `.npy.h5` mỗi bệnh nhân, mảng `(H,W,D)` | Cùng định dạng; kèm spacing và affine ROI |
| Nhãn trong cache đang dùng | Legacy, `2=scar`, `3=edema`; loader đổi sang canonical | Canonical, `2=edema`, `3=scar`; đã chuyển khi đóng gói |
| Chuẩn hóa ảnh | Cache đã chuẩn hóa về [0,1] | Giữ nguyên p1/p99 [0,1] của export |

Cả hai dùng chung `MyopsDataset` và augmentation. Loader kiểm tra ảnh/mask khớp giữa ba modality và khoảng giá trị ảnh trước augmentation. Model nhận ba tensor ảnh `(B,1,H,W)` và một mask `(B,H,W)` với nhãn canonical. Cache MyoPS380 chính thức thiếu spacing/affine; metric khoảng cách dùng voxel. MyoPS++ giữ spacing XY 1 mm, Z gốc và affine của ROI.

Đường dẫn MyoPS380 mặc định trong YAML là `E:/STUDY/DATASET/MyoPS380/Processed_data`. Bản dữ liệu đã kiểm tra trên máy này nằm tại `D:/NCKH/Dataset/MyoPS380/Processed_data`; truyền `--data-root D:/NCKH/Dataset/MyoPS380/Processed_data` nếu dùng bản này. Đường dẫn trong các lệnh bên dưới cần khớp nơi lưu cache thực tế.

Chạy nhiều seed với **cùng manifest**: mặc định no-val chỉ dùng `train.txt`, `test_vol.txt`; `--with-val` cần thêm `val.txt`, `val_vol.txt`. `--seed` thay đổi khởi tạo, augmentation và thứ tự batch. Với validation, tạo manifest đầy đủ trước thí nghiệm nhiều seed để khóa cùng bệnh nhân.

Tạo/kiểm tra manifest từ cache MyoPS380 đã có:

```powershell
python preprocessing/build_splits.py --no-val --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir preprocessing/splits/myops380_no_val --test-list preprocessing/splits/test_vol.txt --seed 1234
python preprocessing/verify.py --no-val --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir preprocessing/splits/myops380_no_val --label-order legacy
```

Manifest đầy đủ được dùng lại, không ghi đè khi đổi seed. Cache MyoPS380 chính thức thiếu metadata dùng `legacy` (`2=scar, 3=edema`); cache SCAR mới dùng `canonical`. Pipeline chuyển nhãn về canonical và kiểm tra metadata; `auto` không đoán khi thiếu thông tin.

Packaging NIfTI đã căn chỉnh dùng `preprocessing/process_and_save.py`; script từ chối ghi đè cache có dữ liệu và không thực hiện đăng ký ảnh. Khi đóng gói NIfTI MyoPS380 bằng mặc định `unit255`, ảnh được clip(0,255)/255. MyoPS++ ROI giữ nguyên ảnh float32 [0,1] đã chuẩn hóa p1/p99 trong export; adapter không crop, resize hay chuẩn hóa lần hai. Nhãn nguồn MyoPS++ được đổi một lần: `0/500/600 → 0`, `200 → 1`, `1220 → 2`, `2221 → 3`. Xem [MyoPS++](preprocessing/MYOPSPP.md) và [ROI](docs/myopspp_roi_training.md) cho lệnh chuẩn bị dữ liệu và giới hạn của profile.

## Huấn luyện

M2-Plus là mặc định của `train.py` và `run_all.py`. Config chính: [m2_plus.yaml](training/config/models/m2_plus.yaml); thông số chung: [base.yaml](training/config/base.yaml). Ưu tiên base YAML → dataset profile → model YAML → CLI.

Augmentation train dùng đúng implementation **I-MMSeg-Fork**: cùng thứ tự RNG, xoay/flip/noise theo `if/elif`, dịch `np.roll`, brightness/gamma chung ba modality, noise σ=0.2 dùng chung, random crop và cubic resize. Giữ nguyên công thức gamma theo 255 và không clip bổ sung sau resize. Xem [protocol augmentation](docs/im_mseg_augmentation.md) và kiểm thử đối chiếu tensor/RNG. Thay đổi này áp dụng cho các profile/model dùng chung `RandomGenerator`; bắt đầu run mới để dùng nhất quán augmentation I-MMSeg từ đầu.

```powershell
python run_all.py --skip-cache --label-order legacy --run-id m2_plus_seed1234
```

Lệnh trên tạo/kiểm tra manifest, verify cache, train và evaluate `m2plus_seed1234.pt` sau đủ ngân sách train. Thêm `--skip-evaluate` để chỉ train. Train riêng:

```powershell
python train.py --config training/config/models/m2_plus.yaml --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --run-id m2_plus_seed1234 --seed 1234
```

Pretrained encoder là tùy chọn: thêm `--pretrained training/pretrained/R50-ViT-B_16.npz`. Mặc định batch vật lý 16, accumulation 1. GPU ít VRAM có thể giảm `--batch-size` hoặc bật `--gradient-checkpointing`; batch Dice yêu cầu accumulation 1. Nếu cần accumulation, chọn rõ `--negative-slice-handling none` để dùng per-image Dice; đây là mục tiêu khác recipe I-MMSeg. Batch hiệu dụng giống nhau không bảo đảm batch Dice/BatchNorm tương đương.

MyoPS++ predicted ROI dùng cùng M2-Plus (hoặc model YAML Lightweight). Tạo cache một lần khi chưa có:

```powershell
python -m preprocessing.myopspp_roi
```

Train riêng hoặc chạy cả train và test trên cache có sẵn, dùng batch 16 và accumulation 1 mặc định:

```powershell
python train.py --dataset myopspp_unet2d_roi128_1mm --run-id myopspp_m2_plus_seed1234
python run_all.py --dataset myopspp_unet2d_roi128_1mm --skip-cache --run-id myopspp_m2_plus_full
```

Hai lệnh trên là hai cách khởi chạy run riêng. Với GPU ít VRAM, có thể thêm `--batch-size 1 --accum-steps 1 --gradient-checkpointing`; batch 1 làm thay đổi batch Dice và thống kê BatchNorm so với batch 16. Adapter từ chối thư mục cache đã tồn tại; khi cache đã có, dùng verify hoặc train trực tiếp.

Recipe train mặc định khớp phần tối ưu hóa chung của **I-MMSeg**: LR `0.0003333`, AdamW một nhóm tham số, weight decay `0.0001` cho cả kernel/bias/norm, không gradient clipping (`clip_grad: 0`). Scheduler polynomial mũ 0.9 cập nhật sau optimizer: lần cập nhật đầu vẫn giữ base LR, hai optimizer updates đầu dùng cùng base LR; LR sau update cuối là `base_lr * (1 / total_updates)^0.9`. AMP tự động và 4 workers được giữ nguyên.

Loss mặc định: `0.2 CE + 0.8 Dice`. Dice gộp batch, mẫu số bình phương, smooth `1e-5`, có background, trọng số `[0.5,1,1,1]` và tổng loss lớp **chia 4** (`loss.dice_normalization: classes`), đúng I-MMSeg. `weights` giữ cách chia tổng trọng số cho thí nghiệm riêng. Hỗ trợ `--negative-slice-handling none|mask|batch`, `--no-include-background` và class weights; loss phụ AAR/scar mặc định bằng 0. AAR là edema-inclusive `{2,3}`. Khi bật loss phụ và accumulation >1, chọn `pathology_reduction: sample`. Loss protocol 4 khóa cả normalization khi resume. Optimizer/scheduler được khóa bằng `optimization_protocol`; checkpoint recipe cũ cần run mới hoặc `--init-weights`, không chuyển recipe giữa chừng bằng resume.

| Cấu hình loss hiệu lực mặc định | MyoPS380 | MyoPS++ ROI128 1 mm |
|---|---|---|
| `pathology_reduction` | `positive` trong base YAML | `positive` trong dataset YAML |
| `aar_weight` / `scar_weight` | 0 / 0 | 0 / 0 |
| Loss được tối ưu | `0.2 CE + 0.8 Dice` | `0.2 CE + 0.8 Dice` |

Hai dataset hiện cùng dùng `positive`. `pathology_reduction` chỉ điều khiển loss phụ AAR/scar. `positive` lấy trung bình trên các lát có vùng tương ứng; `sample` chia tổng loss của các lát đó cho toàn bộ số lát trong batch, các lát âm đóng góp 0 vào thành phần phụ. Ví dụ batch 16 có 4 lát scar, thành phần phụ scar của `sample` bằng 1/4 của `positive` trên cùng logits/GT. Khi hai trọng số phụ bằng 0, đổi reduction giữ nguyên loss và gradient trên cùng batch. Khi bật loss phụ để so sánh hai dataset, cần đặt cùng reduction và trọng số. Tham số này độc lập với nhãn legacy/canonical. `train.py` nhận `--pathology-reduction`; với `run_all.py`, đặt `loss.pathology_reduction` trong YAML. Resume yêu cầu giữ nguyên reduction của run gốc, kể cả khi trọng số phụ bằng 0.

`--config testing` dùng **M2-Plus thu nhỏ**, input 32×32, để kiểm thử phần mềm; không dùng cho bảng kết quả nghiên cứu.

## Checkpoint và resume

```text
outputs/runs/M2-Plus_seed1234_<timestamp>/
├── checkpoints/m2plus_seed1234.pt      # Có thêm best.pt khi bật validation
├── tensorboard/
└── logs_M2-Plus_seed1234/
    ├── config.json, train.log, metrics.csv, metrics.jsonl, summary.json
    ├── splits/
    └── test_results/
```

Mặc định no-val chỉ lưu `m2plus_seed1234.pt` sau mỗi epoch; không chọn best hoặc early stopping. Với `--with-val`, `best.pt` chọn bằng trung bình patient-mean Dice scar và edema-inclusive trên **validation volumes**. Early stopping dùng cùng điểm; `patience: 0` tắt early stopping. `m2plus_seed1234.pt` giữ optimizer, scheduler, scaler và RNG.

Tên checkpoint cuối theo `<model>_seed<seed>.pt`: `M2-Plus` → `m2plus_seed1234.pt`, `Lightweight-M2-Plus` → `lightweightm2plus_seed1234.pt` (ví dụ seed 1234). File này cũng là điểm khôi phục sau mỗi epoch; metadata xác định run đã hoàn thành ngân sách hay chưa. CLI vẫn nhận tên cũ `last.pt`/`last.pth` theo các điều kiện resume/evaluate hiện có.

```powershell
python train.py --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --run-id m2_plus_seed1234 --resume outputs/runs/m2_plus_seed1234/checkpoints/m2plus_seed1234.pt
```

Truyền lại cùng config, seed, batch size, accumulation, LR, tổng epoch, loss và nhãn của run gốc; `--resume` không tự nạp toàn bộ CLI. `--epochs-per-run` dừng sau một số epoch mà giữ lịch LR đã lên kế hoạch. `--init-weights` nạp chính xác trọng số M2-Plus cho run mới, không mang optimizer/split/score cũ. Ba nguồn `--resume`, `--init-weights`, `--pretrained` loại trừ nhau.

Tên tensor M2-Plus được giữ để nạp checkpoint cũ. Metadata `architecture: cmspa_net` chỉ được chấp nhận khi `ablation` xác định M2-PLUS. Model khác bị từ chối. Checkpoint dùng protocol chọn best cũ vẫn có thể evaluate hoặc khởi tạo run mới nếu metadata hợp lệ; không resume trạng thái best/patience theo metric cũ. Các kiểm tra dữ liệu, loss và split vẫn áp dụng.

Fusion được version hóa riêng: `legacy_v1` giữ attention full-width/BatchNorm và tensor cũ; `spatial_v2` dùng projection latent/GroupNorm/PE. Run mới lưu `fusion_version`, `latent_dim`, `use_pos_enc` và `sspanet_reduction`. Loader nhận diện fusion của checkpoint thiếu metadata từ các tensor đã lưu, không chuyển shape trọng số. Muốn resume fusion cũ, chọn `model.fusion_version: legacy_v1` cùng cấu hình run gốc. `--init-weights` vẫn yêu cầu toàn bộ trọng số tương thích; chuyển sang fusion khác cần run mới với khởi tạo encoder tương thích.

Checkpoint M2-Plus thiếu `sspanet_gate` được evaluate/predict bằng công thức `legacy_v1` ban đầu. Run mới mặc định dùng `centered_tanh_v2`; `--resume` từ chối đổi gate giữa chừng. Để chuyển trọng số cũ sang gate mới, dùng `--init-weights` với run mới và huấn luyện lại trước khi đánh giá. Muốn tiếp tục run cũ với gate cũ, đặt `model.sspanet_gate: legacy_v1` trong YAML của run; các điều kiện resume khác vẫn phải khớp.

## Đánh giá và prediction

```powershell
python test.py --checkpoint outputs/runs/m2_plus_seed1234/checkpoints/m2plus_seed1234.pt --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --split test_vol --batch-size 8 --save-predictions
```

Evaluator xuất `test.log`, `per_case.csv`, `metrics.json` và mask `.npz`. Chỉ xuất NIfTI khi cache có affine hợp lệ. Evaluation không cần TensorBoard. Chọn thư mục mới nếu kết quả đã tồn tại.

Validation, test và prediction dùng chung `predict_volume`: một forward mỗi batch, resize logits về lưới đánh giá trước `argmax`, giữ tổn thương nhỏ và không liên thông. Không TTA/lọc connected components. Báo cáo ghi `single_pass_argmax_v1`, nguồn dữ liệu, hash manifest và protocol.

Prediction NIfTI đã căn chỉnh cho checkpoint thuộc profile native:

```powershell
python training/predict.py --checkpoint outputs/runs/m2_plus_seed1234/checkpoints/m2plus_seed1234.pt --cine /path/case_C0.nii.gz --lge /path/case_LGE.nii.gz --t2w /path/case_T2.nii.gz --normalization unit255 --output /path/prediction.nii.gz
```

Dùng normalization đúng với training. Checkpoint ROI phải dùng evaluator trên ROI cache; raw prediction từ chối chúng vì cần localizer/preprocessing riêng.

## Protocol báo cáo

- Region: normal `{1}`, edema-exclusive `{2}`, scar `{3}`, edema-inclusive `{2,3}`, myocardial ring `{1,2,3}`. Không tự gán edema-inclusive thành cột edema trong paper khác.
- Metric trung bình theo bệnh nhân. Hai mask rỗng: metric không xác định; một mask rỗng: Dice/IoU bằng 0, distance không xác định. Báo số ca defined/undefined và trạng thái empty.
- MyoPS380 dùng HD95/ASD trên voxel grid đơn vị, không suy diễn mm từ affine identity. MyoPS++ dùng mm. ROI lấy tâm từ **dự đoán U-Net2D**, giữ mọi bệnh nhân, kể cả ca crop mất nhãn. Đánh giá trên ROI grid; không khôi phục prediction về whole-image native grid. Đây là metric segmentation trên ROI đã xuất, không phải metric end-to-end toàn ảnh.
- Cache bytes, nhãn, split và protocol được khóa trong checkpoint. Resume/evaluation từ chối khi không khớp. Test không dùng để tune hyperparameter/chọn model.
- Baseline cần cùng bệnh nhân, GT/grid, region, evaluator, ngân sách train và inference. Điểm paper chỉ là tham khảo khi protocol chưa khớp; không suy ra vượt SOTA từ bảng lịch sử.

## Kiểm thử và cấu trúc

```powershell
python -m pytest -q
```

Suite kiểm tra forward/backward M2-Plus, loss/accumulation, checkpoint cũ, rejection model đã bỏ, train → resume → evaluate, patient split, cache fingerprint, native/ROI geometry, metric và logging.

```text
SCAR/
├── preprocessing/          # Packaging, verify, patient manifests
├── training/
│   ├── models/
│   │   ├── m2_plus_net.py
│   │   ├── backbones/resnet_v2.py
│   │   └── modules/        # m2_plus, sspanet, fusion, decoder
│   ├── config/             # base, M2-Plus/testing, dataset profiles
│   ├── dataset/
│   ├── loss/
│   ├── metrics/
│   ├── trainer/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── docs/
├── train.py                # CLI wrapper
├── test.py                 # CLI wrapper
└── run_all.py
```

`data/`, `outputs/`, `reports/` và weights không được đưa vào Git. Kết quả/checkpoint lịch sử được giữ trên máy; implementation model khác đã bỏ khỏi phiên bản repo này.

## File sinh tự động và dữ liệu lịch sử

`__pycache__/`, `*.pyc` và `.pytest_cache/` là cache sinh tự động, đã được Git ignore và có thể xóa khi dọn workspace. Python/pytest sẽ tạo lại khi chạy.

Các thư mục `data/myopspp_bc80`, `data/myopspp_roi128_76`, `data/myopspp_roi128_80`, `data/myopspp_roi160_80` là cache lịch sử của các profile đã gỡ. Hai profile hiện tại dùng các đường dẫn riêng trong YAML; cache MyoPS++ hiện tại là `data/myopspp_unet2d_roi128_1mm/cache`. Cache lịch sử vẫn có thể cần để đối chiếu các run cũ.

`preprocessing/splits/myopspp_bc80_no_val` là manifest lịch sử, ngoài hai profile được hỗ trợ. `preprocessing/myopspp_unet2d.py` và `preprocessing/myopspp_unet2d_roi192.py` phục vụ tái lập localizer/export và được các test tương ứng sử dụng. `data/processed/splits` phục vụ MyoPS380 `--with-val`. Giữ các manifest đã dùng cho checkpoint cần resume; fingerprint nguồn hiện bao gồm cả file `.py`, `.yaml`, `.txt`, `.json` trong `training/` và `preprocessing/`.

`BENCHMARK_EFFICIENCY_PROTOCOL.txt` và `Bang_So_Sanh_Do_Phuc_Tap_Model.xlsx` là tài liệu benchmark riêng trên máy. Quy ước A100/FP32/0 worker trong tài liệu benchmark cần được cấu hình riêng khi đo; CLI train mặc định dùng AMP tự động và 4 worker.
