# MyoPS++ U-Net2D ROI128, 1 mm

Profile duy nhất cho MyoPS++: `myopspp_unet2d_roi128_1mm`.
Nguồn trên máy: `D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm`.
MyoPS380 giữ pipeline, nhãn legacy/canonical, split và metric voxel riêng.

## Dữ liệu, split và nhãn

80 bệnh nhân CenterB/CenterC, cố định **64 train / 16 test, không validation**.
Train gồm 28 B + 36 C; test gồm 7 B + 9 C. Manifest được lưu trong
`preprocessing/splits/myopspp_unet2d_roi128_1mm/patients.json`; đổi seed không chia lại bệnh nhân.
Export hiện có 296 lát cắt train và 16 volume test. Tất cả lát cắt, kể cả lát không có tổn thương, đều được giữ.

| Mã nhãn nguồn | ID trong cache/model | Ý nghĩa |
|---|---:|---|
| 0, 500 (LV), 600 (RV) | 0 | background của bài toán phân đoạn cơ tim |
| 200 | 1 | normal myocardium |
| 1220 | 2 | edema-exclusive |
| 2221 | 3 | scar |

Edema-inclusive/AAR là hợp `{2,3}`. Các nhãn lạ hoặc không nguyên bị từ chối.
Không dùng chuyển nhãn legacy của MyoPS380 cho export raw MyoPS++.

Ảnh C0/LGE/T2 là float32 [0,1], đã chuẩn hóa p1/p99 theo volume/modalities.
Adapter sao chép đúng giá trị, mapping nhãn một lần, giữ shape 128×128×D,
spacing XY 1 mm, Z gốc và affine ROI. Adapter kiểm tra checksum, hình học, split,
provenance và localizer không được train trên chính bệnh nhân đang crop.
Dữ liệu nguồn, các checkpoint localizer và cache lịch sử trên đĩa không bị chỉnh sửa.

Tâm crop là dự đoán U-Net2D, không phải oracle theo GT. Provenance localizer vẫn ghi
các fold/train/validation lịch sử; split 64/16 ở đây là split huấn luyện segmentation
hiện tại. Không train lại localizer hoặc đổi tâm crop khi đóng gói cache.

## Tạo cache và verify

Chạy từ root repo. Trên máy khác, thay `--src-path` bằng đường dẫn export đã giải nén:

```powershell
python -m preprocessing.myopspp_roi --src-path D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm --dst-path data/myopspp_unet2d_roi128_1mm/cache
python preprocessing/verify.py --dataset myopspp_unet2d_roi128_1mm --no-val --label-order canonical --data-root data/myopspp_unet2d_roi128_1mm/cache --list-dir data/myopspp_unet2d_roi128_1mm/cache/lists
```

Adapter từ chối thư mục đích đã tồn tại; cache sẵn có chỉ cần verify và train.
Cache nằm ngoài export, được Git ignore. Kéo repo không tự tải dataset.
Bốn profile native/oracle trước đây đã được gỡ khỏi CLI và cấu hình; không dùng
cache/checkpoint benchmark cũ dưới identity của profile mới.

## Train, resume, evaluate

```powershell
python train.py --dataset myopspp_unet2d_roi128_1mm --batch-size 1 --accum-steps 1 --gradient-checkpointing --run-id myopspp_m2_plus_seed1234
```

Profile tự chọn canonical labels, no-val, cache và output riêng. Budget mặc định 300 epoch;
lưu `m2plus_seed1234.pt` sau mỗi epoch, không chọn `best.pt`, không early stopping.
Test không được dùng để chọn epoch/model. GPU ít VRAM dùng batch nhỏ; batch 1 ở lệnh
trên không tương đương batch 16 của recipe mặc định. Batch Dice yêu cầu accumulation 1.
Để dùng Lightweight M2-Plus, thêm `--config training/config/models/lightweight_m2_plus.yaml`.

```powershell
python train.py --dataset myopspp_unet2d_roi128_1mm --batch-size 1 --accum-steps 1 --gradient-checkpointing --resume outputs/myopspp_unet2d_roi128_1mm/runs/myopspp_m2_plus_seed1234/checkpoints/m2plus_seed1234.pt
python test.py --checkpoint outputs/myopspp_unet2d_roi128_1mm/runs/myopspp_m2_plus_seed1234/checkpoints/m2plus_seed1234.pt --data-root data/myopspp_unet2d_roi128_1mm/cache
```

Resume dùng cùng config, seed, budget, batch, loss và đường dẫn dữ liệu. Evaluator từ chối
checkpoint chưa đủ budget. Split, cache bytes, nhãn và protocol được khóa trong checkpoint.
Khi dùng cache ở đường dẫn khác với `train.py`, truyền cả `--data-root` và `--list-dir`.
Config `testing` có validation trong YAML để kiểm thử MyoPS380; với profile này phải thêm
`--no-val` khi dùng `--config testing` (model nhỏ 32×32, chỉ dùng kiểm thử phần mềm).

Pipeline đầy đủ trên cache đã có:

```powershell
python run_all.py --dataset myopspp_unet2d_roi128_1mm --skip-cache --batch-size 1 --accum-steps 1 --gradient-checkpointing --run-id myopspp_m2_plus_full
```

Bỏ `--skip-cache` để pipeline tạo cache khi chưa có. `--raw-root` chọn export nguồn khác.

## Đánh giá

Dice/IoU/precision/recall tính theo bệnh nhân; HD95/ASD tính bằng mm trên **predicted ROI grid**.
Các vùng canonical giữ nguyên: normal `{1}`, edema-exclusive `{2}`, scar `{3}`,
edema-inclusive `{2,3}`, myocardial ring `{1,2,3}`.
Metadata ghi `localization: predicted_myocardium_unet2d_oof_train_final_val_test` và
`evaluation_grid: preprocessed_roi`. Prediction NIfTI mang affine ROI.

Evaluator này chưa khôi phục probabilities về native whole-image grid và chưa chấm với
`reference_native/`; không dùng điểm ROI như điểm end-to-end trên toàn ảnh gốc.
Raw inference từ chối checkpoint ROI vì cần cùng localizer/preprocessing như export.
MyoPS380 tiếp tục dùng metric voxel và cơ chế validation tùy chọn như trước.
