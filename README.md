# SCAR — M2-Plus

Phân đoạn cơ tim đa phương thức từ bSSFP/CINE, LGE/PSIR và T2w đã căn chỉnh. Pipeline hỗ trợ **M2-Plus** và **Lightweight M2-Plus** cho train, resume và đánh giá volume.

## Kiến trúc và nhãn

M2-Plus dùng ba encoder ResNetV2 độc lập, SSPANet trên từng modality, hai nhánh cross-attention riêng cho scar/edema, ba tầng skip fusion concat/project và decoder bốn cấp. CINE cung cấp queries; LGE và T2w cung cấp keys/values cho hai attention stream. Hai stream được kết hợp với đặc trưng CINE trước decoder.

Mã chính: [m2_plus_net.py](training/models/m2_plus_net.py), [m2_plus.py](training/models/modules/m2_plus.py). Input mặc định 128×128; model trả raw logits `(B,4,H,W)` trong cả train và eval. Không có anatomy head hay skip gate của các model nghiên cứu trước đây. File pretrained `R50-ViT-B_16.npz` chỉ khởi tạo phần ResNet tương thích; model không dùng ViT hoặc text encoder.

SSPANet mặc định dùng `model.sspanet_gate: centered_tanh_v2`: `y = x * (1 + 0.5 * tanh(g_channel(x) + g_spatial(x)))`. Hai nhánh cung cấp logits trước sigmoid và trước phép nhân với `x`; hệ số trong khoảng 0.5–1.5 cho phép giảm hoặc tăng đặc trưng, bằng 1 khi tổng logits bằng 0. Đây là biến thể SCAR của SSPANet; ảnh hưởng tới Dice cần được xác nhận bằng ablation.

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
Muốn dùng protocol phát triển 243/61/76, thêm `--with-val`; MyoPS++/ROI giữ validation theo profile riêng.

| Profile | Train / val / test (bệnh nhân) | Khoảng cách metric |
|---|---|---|
| `myops380` — mặc định | 304 / 0 / 76 | voxel |
| `myops380 --with-val` | 243 / 61 / 76 với bộ chia phát triển | voxel |
| `myopspp_bc80` | 51 / 13 / 16 | mm, native grid |
| `myopspp_roi128_76` | 48 / 13 / 15 | mm, ROI grid |
| `myopspp_roi128_80` | 51 / 13 / 16 | mm, ROI grid |
| `myopspp_roi160_80` | 51 / 13 / 16 | mm, ROI grid |

MyoPS++ và ROI dùng patient split cố định đã version-control. MyoPS380 khóa 76 ca test trong [test_vol.txt](preprocessing/splits/test_vol.txt). Không chia lát cắt của cùng bệnh nhân sang nhiều tập.

Chạy nhiều seed với **cùng manifest**: mặc định no-val chỉ dùng `train.txt`, `test_vol.txt`; `--with-val` cần thêm `val.txt`, `val_vol.txt`. `--seed` thay đổi khởi tạo, augmentation và thứ tự batch. Với validation, tạo manifest đầy đủ trước thí nghiệm nhiều seed để khóa cùng bệnh nhân.

Tạo/kiểm tra manifest từ cache MyoPS380 đã có:

```powershell
python preprocessing/build_splits.py --no-val --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir preprocessing/splits/myops380_no_val --test-list preprocessing/splits/test_vol.txt --seed 1234
python preprocessing/verify.py --no-val --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir preprocessing/splits/myops380_no_val --label-order legacy
```

Manifest đầy đủ được dùng lại, không ghi đè khi đổi seed. Cache MyoPS380 chính thức thiếu metadata dùng `legacy` (`2=scar, 3=edema`); cache SCAR mới dùng `canonical`. Pipeline chuyển nhãn về canonical và kiểm tra metadata; `auto` không đoán khi thiếu thông tin.

Packaging NIfTI đã căn chỉnh dùng `preprocessing/process_and_save.py`; script từ chối ghi đè cache có dữ liệu và không thực hiện đăng ký ảnh. MyoPS380 dùng clip(0,255)/255; MyoPS++ native dùng percentile p1/p99 trên volume. Xem [MyoPS++](preprocessing/MYOPSPP.md) và [ROI](docs/myopspp_roi_training.md) cho lệnh chuẩn bị dữ liệu và giới hạn của từng profile.

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

Các profile khác giữ cùng M2-Plus:

```powershell
python train.py --dataset myopspp_bc80 --run-id m2_plus_seed1234
python run_all.py --dataset myopspp_roi160_80 --skip-cache --batch-size 2 --accum-steps 1 --run-id m2_plus_seed1234
```

Recipe train mặc định khớp phần tối ưu hóa chung của **I-MMSeg**: LR `0.0003333`, AdamW một nhóm tham số, weight decay `0.0001` cho cả kernel/bias/norm, không gradient clipping (`clip_grad: 0`). Scheduler polynomial mũ 0.9 cập nhật sau optimizer: lần cập nhật đầu vẫn giữ base LR, hai optimizer updates đầu dùng cùng base LR; LR sau update cuối là `base_lr * (1 / total_updates)^0.9`. AMP tự động và 4 workers được giữ nguyên.

Loss mặc định: `0.2 CE + 0.8 Dice`. Dice gộp batch, mẫu số bình phương, smooth `1e-5`, có background, trọng số `[0.5,1,1,1]` và tổng loss lớp **chia 4** (`loss.dice_normalization: classes`), đúng I-MMSeg. `weights` giữ cách chia tổng trọng số cho thí nghiệm riêng. Hỗ trợ `--negative-slice-handling none|mask|batch`, `--no-include-background` và class weights; loss phụ AAR/scar mặc định bằng 0. AAR là edema-inclusive `{2,3}`. Khi bật loss phụ và accumulation >1, chọn `pathology_reduction: sample`. Loss protocol 4 khóa cả normalization khi resume. Optimizer/scheduler được khóa bằng `optimization_protocol`; checkpoint recipe cũ cần run mới hoặc `--init-weights`, không chuyển recipe giữa chừng bằng resume.

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
python training/predict.py --checkpoint outputs/runs/m2_plus_seed1234/checkpoints/best.pt --cine /path/case_C0.nii.gz --lge /path/case_LGE.nii.gz --t2w /path/case_T2.nii.gz --normalization unit255 --output /path/prediction.nii.gz
```

Dùng normalization đúng với training. Checkpoint ROI phải dùng evaluator trên ROI cache; raw prediction từ chối chúng vì cần localizer/preprocessing riêng.

## Protocol báo cáo

- Region: normal `{1}`, edema-exclusive `{2}`, scar `{3}`, edema-inclusive `{2,3}`, myocardial ring `{1,2,3}`. Không tự gán edema-inclusive thành cột edema trong paper khác.
- Metric trung bình theo bệnh nhân. Hai mask rỗng: metric không xác định; một mask rỗng: Dice/IoU bằng 0, distance không xác định. Báo số ca defined/undefined và trạng thái empty.
- MyoPS380 dùng HD95/ASD trên voxel grid đơn vị, không suy diễn mm từ affine identity. MyoPS++ dùng mm. ROI là **GT-centered oracle crop**, đánh giá trên ROI grid; không đánh giá localizer hoặc khôi phục whole-image grid.
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
