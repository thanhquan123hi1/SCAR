# SCAR — M2-Plus

Phân đoạn cơ tim đa phương thức từ bSSFP/CINE, LGE/PSIR và T2w đã căn chỉnh. Repo chỉ hỗ trợ **M2-Plus** cho train, resume, đánh giá volume và prediction.

## Kiến trúc và nhãn

M2-Plus dùng ba encoder ResNetV2 độc lập, SSPANet trên từng modality, hai nhánh cross-attention riêng cho scar/edema, ba tầng skip fusion concat/project và decoder bốn cấp. CINE cung cấp queries; LGE và T2w cung cấp keys/values cho hai attention stream. Hai stream được kết hợp với đặc trưng CINE trước decoder.

Mã chính: [m2_plus_net.py](training/models/m2_plus_net.py), [m2_plus.py](training/models/modules/m2_plus.py). Input mặc định 128×128; model trả raw logits `(B,4,H,W)` trong cả train và eval. Không có anatomy head hay skip gate của các model nghiên cứu trước đây. File pretrained `R50-ViT-B_16.npz` chỉ khởi tạo phần ResNet tương thích; model không dùng ViT hoặc text encoder.

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

Notebook Colab: [scar_pipeline.ipynb](scar_pipeline.ipynb). Chuẩn bị dataset và pretrained weights riêng; Git không chứa chúng.

## Dataset và split

| Profile | Train / val / test (bệnh nhân) | Khoảng cách metric |
|---|---|---|
| `myops380` — mặc định | 243 / 61 / 76 với bộ chia hiện có | voxel |
| `myopspp_bc80` | 51 / 13 / 16 | mm, native grid |
| `myopspp_roi128_76` | 48 / 13 / 15 | mm, ROI grid |
| `myopspp_roi128_80` | 51 / 13 / 16 | mm, ROI grid |
| `myopspp_roi160_80` | 51 / 13 / 16 | mm, ROI grid |

MyoPS++ và ROI dùng patient split cố định đã version-control. MyoPS380 khóa 76 ca test trong [test_vol.txt](preprocessing/splits/test_vol.txt). Không chia lát cắt của cùng bệnh nhân sang nhiều tập.

Chạy nhiều seed với **cùng `--list-dir` có đủ** `train.txt`, `val.txt`, `val_vol.txt`, `test_vol.txt`. `--seed` thay đổi khởi tạo, augmentation và thứ tự batch. Nếu chưa có `val.txt`, trainer có thể tạo validation theo seed; hãy tạo manifest hoàn chỉnh trước thí nghiệm nhiều seed.

Tạo/kiểm tra manifest từ cache MyoPS380 đã có:

```powershell
python preprocessing/build_splits.py --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir data/processed/splits --test-list preprocessing/splits/test_vol.txt --seed 1234
python preprocessing/verify.py --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir data/processed/splits --label-order legacy
```

Manifest đầy đủ được dùng lại, không ghi đè khi đổi seed. Cache MyoPS380 chính thức thiếu metadata dùng `legacy` (`2=scar, 3=edema`); cache SCAR mới dùng `canonical`. Pipeline chuyển nhãn về canonical và kiểm tra metadata; `auto` không đoán khi thiếu thông tin.

Packaging NIfTI đã căn chỉnh dùng `preprocessing/process_and_save.py`; script từ chối ghi đè cache có dữ liệu và không thực hiện đăng ký ảnh. MyoPS380 dùng clip(0,255)/255; MyoPS++ native dùng percentile p1/p99 trên volume. Xem [MyoPS++](preprocessing/MYOPSPP.md) và [ROI](docs/myopspp_roi_training.md) cho lệnh chuẩn bị dữ liệu và giới hạn của từng profile.

## Huấn luyện

M2-Plus là mặc định của `train.py` và `run_all.py`. Config chính: [m2_plus.yaml](training/config/models/m2_plus.yaml); thông số chung: [base.yaml](training/config/base.yaml). Ưu tiên base YAML → dataset profile → model YAML → CLI.

```powershell
python run_all.py --skip-cache --label-order legacy --run-id m2_plus_seed1234
```

Lệnh trên tạo/kiểm tra manifest, verify cache, train và evaluate `best.pt`. Train riêng:

```powershell
python train.py --config training/config/models/m2_plus.yaml --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir data/processed/splits --run-id m2_plus_seed1234 --seed 1234
```

Pretrained encoder là tùy chọn: thêm `--pretrained training/pretrained/R50-ViT-B_16.npz`. GPU ít VRAM có thể giảm `--batch-size` và tăng `--accum-steps`; batch hiệu dụng là tích hai tham số này. BatchNorm và phép toán ngẫu nhiên vẫn có thể làm microbatch khác batch vật lý lớn.

Các profile khác giữ cùng M2-Plus:

```powershell
python train.py --dataset myopspp_bc80 --run-id m2_plus_seed1234
python run_all.py --dataset myopspp_roi160_80 --skip-cache --batch-size 2 --accum-steps 8 --run-id m2_plus_seed1234
```

Loss mặc định: `0.5 CE + 0.5 Dice`. Hỗ trợ `--aar-weight`, `--scar-weight`, `--ce-class-weights`, `--dice-class-weights`; loss phụ mặc định bằng 0. AAR là edema-inclusive `{2,3}`. ROI dùng `pathology_reduction: sample`; MyoPS380/native MyoPS++ mặc định `positive`. Loss và reduction được khóa khi resume.

`--config testing` dùng **M2-Plus thu nhỏ**, input 32×32, để kiểm thử phần mềm; không dùng cho bảng kết quả nghiên cứu.

## Checkpoint và resume

```text
outputs/runs/M2-Plus_seed1234_<timestamp>/
├── checkpoints/{best.pt,last.pt}
├── tensorboard/
└── logs_M2-Plus_seed1234/
    ├── config.json, train.log, metrics.csv, metrics.jsonl, summary.json
    ├── splits/
    └── test_results/
```

`best.pt` chọn bằng trung bình patient-mean Dice scar và edema-inclusive trên **validation volumes**. Early stopping dùng cùng điểm; `patience: 0` tắt early stopping. `min_delta` chỉ điều khiển patience, mọi cải thiện điểm tốt nhất vẫn cập nhật `best.pt`. `last.pt` giữ optimizer, scheduler, scaler và RNG. Không lưu checkpoint theo chu kỳ.

```powershell
python train.py --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --list-dir data/processed/splits --run-id m2_plus_seed1234 --resume outputs/runs/m2_plus_seed1234/checkpoints/last.pt
```

Truyền lại cùng config, seed, batch size, accumulation, LR, tổng epoch, loss và nhãn của run gốc; `--resume` không tự nạp toàn bộ CLI. `--epochs-per-run` dừng sau một số epoch mà giữ lịch LR đã lên kế hoạch. `--init-weights` nạp chính xác trọng số M2-Plus cho run mới, không mang optimizer/split/score cũ. Ba nguồn `--resume`, `--init-weights`, `--pretrained` loại trừ nhau.

Tên tensor M2-Plus được giữ để nạp checkpoint cũ. Metadata `architecture: cmspa_net` chỉ được chấp nhận khi `ablation` xác định M2-PLUS. Model khác bị từ chối. Checkpoint dùng protocol chọn best cũ vẫn có thể evaluate hoặc khởi tạo run mới nếu metadata hợp lệ; không resume trạng thái best/patience theo metric cũ. Các kiểm tra dữ liệu, loss và split vẫn áp dụng.

## Đánh giá và prediction

```powershell
python test.py --checkpoint outputs/runs/m2_plus_seed1234/checkpoints/best.pt --data-root E:/STUDY/DATASET/MyoPS380/Processed_data --split test_vol --batch-size 8 --save-predictions
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
├── run_all.py
└── scar_pipeline.ipynb
```

`data/`, `outputs/`, `reports/` và weights không được đưa vào Git. Kết quả/checkpoint lịch sử được giữ trên máy; implementation model khác đã bỏ khỏi phiên bản repo này.
