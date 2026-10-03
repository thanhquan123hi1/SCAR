# Benchmark MyoPS++ B/C, đủ ba modality

Profile: **`myopspp_bc80`**. Đây là cohort riêng gồm 80 ca có sẵn trong release local,
không phải tái lập cohort 95 ca của I-MMSeg/PMC hoặc split challenge CARE2024.
MyoPS380 vẫn là benchmark mặc định; cache, split và protocol voxel của nó được giữ riêng.

## Dữ liệu và protocol

- Nguồn local: `E:/STUDY/DATASET/Myo_train/MyoPS_train`.
- Chỉ CenterB (Case2001–Case2035) và CenterC (Case3001–Case3045).
- Mỗi patient phải có đúng một file cho từng `_C0`, `_LGE`, `_T2`, `_gd` (`.nii` hoặc `.nii.gz`).
- Adapter nhận root `Myo_train` hoặc `MyoPS_train`; bỏ qua A/E/F/G/H và CineMyoPS.
- Thiếu modality, duplicate, cohort khác, mã nhãn lạ, geometry không khớp hoặc đơn vị không phải mm đều bị reject.
- Không loại ca có mask scar rỗng, không bổ sung modality bằng ảnh zero.

Split đã version-control trong [`splits/myopspp_bc80/patients.json`](splits/myopspp_bc80/patients.json):

| Center | Train | Validation | Test |
|---|---:|---:|---:|
| B | 22 | 6 | 7 |
| C | 29 | 7 | 9 |
| Tổng | **51** | **13** | **16** |

Seed tạo split = 1234; `--seed` khi train chỉ thay RNG của mô hình/augmentation, không đổi split.
Tỷ lệ tổng tương ứng 63,75% / 16,25% / 20%. Training dùng slice; selection và test dùng
volume bệnh nhân đầy đủ. Mỗi model/baseline phải dùng cùng patient split.

Mapping raw: `0/500/600 → 0`, `200 → 1` normal myocardium, `1220 → 2` edema,
`2221 → 3` scar. Canonical cache không giữ LV/RV; source raw vẫn nguyên vẹn.
Normalization theo từng modality/volume: p1/p99 của voxel khác zero, clip về [0,1].
Volume zero/constant được chuyển thành zero hữu hạn. Không lấy thống kê từ toàn cohort.

Cache giữ native HWD grid, spacing và affine. Training resize ảnh bilinear, label nearest
đến `img_size` của model (mặc định 128); không crop 192 hoặc ROI từ GT. Inference resize
logits về native grid trước argmax, không TTA/connected-component filtering.
Shape/affine agreement không chứng minh anatomical alignment; dùng aligned release hoặc
đăng ký ảnh upstream nếu overlay cho thấy lệch giải phẫu. Grid có shear bị reject vì
distance transform theo spacing cần các trục voxel trực giao.

Metric profile **`myopspp_bc80_mm_v1`**: patient-mean Dice, IoU, precision, recall;
HD95 và ASD theo **mm**. Báo normal, edema-exclusive `{2}`, scar `{3}`,
edema-inclusive `{2,3}`, myocardial ring `{1,2,3}`. Hai mask rỗng bị loại khỏi mean;
một mask rỗng có Dice/IoU=0 và distance undefined. Báo defined/undefined counts.
Không có metric “official reproduction” dựa trên empty-mask threshold của MyoPS380.
`best.pth` chọn bằng trung bình patient-mean Dice scar và edema-exclusive trên val.
`best_inclusive.pth` là output phụ theo scar/edema-inclusive, không dùng test để chọn model.

## Chuẩn bị cache mới

Chạy từ root repo:

```powershell
python -m preprocessing.myopspp --src-path "E:/STUDY/DATASET/Myo_train" --dst-path data/myopspp_bc80/cache
python preprocessing/verify.py --dataset myopspp_bc80 --data-root data/myopspp_bc80/cache --list-dir data/myopspp_bc80/cache/lists --label-order canonical
```

Cache default tại `data/myopspp_bc80/cache`, manifests tại `cache/lists`. Metadata gồm
source SHA256, native shapes, center, raw mapping, normalization và split. Cache chỉ được
publish khi hoàn thành; lệnh từ chối destination đã tồn tại. Nếu đã có cache, dùng verify
và train; không chạy lại packaging vào cùng destination.

## Train và test

```powershell
python train.py --dataset myopspp_bc80 --config training/config/models/cmspa_net.yaml --run-id myopspp_m3_seed1234
python test.py --checkpoint outputs/myopspp_bc80/runs/myopspp_m3_seed1234/checkpoints/best.pth --data-root data/myopspp_bc80/cache
```

Đổi `--config` sang model YAML khác để chạy các ablation trong SCAR trên cùng benchmark.
`--dataset` tự nạp [`../training/config/datasets/myopspp_bc80.yaml`](../training/config/datasets/myopspp_bc80.yaml)
giữa base config và model config. Có thể khai báo `data.dataset_id: myopspp_bc80` trong
model YAML thay cho flag. Dataset identity theo CLI > model YAML > base YAML > MyoPS380.
Explicit CLI có ưu tiên cao nhất. Evaluation tự nhận
dataset từ checkpoint; `--dataset myopspp_bc80` là identity check tùy chọn.

Resume với cùng config, seed, batch size, epoch budget, data-root và model:

```powershell
python train.py --dataset myopspp_bc80 --config training/config/models/cmspa_net.yaml --resume outputs/myopspp_bc80/runs/myopspp_m3_seed1234/checkpoints/last.pth
```

Resume giữ điểm tốt nhất của cả checkpoint chính và inclusive; checkpoint cũ chưa lưu
điểm inclusive được phục hồi từ `best_inclusive.pth` hiện có. Nếu run đã early-stop,
resume với cùng patience không chạy thêm optimizer updates. Muốn tiếp tục, phải chủ động
đổi chính sách patience, ví dụ `--patience 0` để tắt early stopping.

Output theo run: `checkpoints/best.pth`, `last.pth`, `logs_<model>_seed<seed>/config.json`,
metrics CSV/JSONL và `logs_<model>_seed<seed>/test_results/{metrics.json,per_case.csv,*_pred.nii.gz}`.
Cache bytes, label convention, split hashes và metric profile được lưu trong checkpoint;
train/resume/test reject khi không khớp. Checkpoint MyoPS380 không dùng được để benchmark
cache MyoPS++ và ngược lại.

Chạy toàn pipeline trên cache đã có:

```powershell
python run_all.py --dataset myopspp_bc80 --config training/config/models/cmspa_net.yaml --skip-cache --run-id myopspp_m3_full
```

Bỏ `--skip-cache` nếu muốn tự tạo cache khi chưa có; `--raw-root` đổi nguồn,
`--data-root` đổi cache (list mặc định đi cùng cache). Khi chạy train riêng với cache
custom, truyền cả `--data-root` và `--list-dir`. Model seed khác vẫn giữ patient split.
Trong model YAML, thay riêng `data.data_root` sẽ chuyển list mặc định về cache mới `/lists`;
`data.list_dir` được khai báo rõ sẽ được giữ. `run_all` hỗ trợ `outputs.dir` hoặc alias
`outputs.output_dir`; `--run-id` chỉ định trực tiếp sẽ chọn run đó dưới `--run-root`.

MyoPS380 tiếp tục dùng các lệnh cũ không có `--dataset`, các split 243/61/76 và protocol
`myops380_voxel_v1`. Không so sánh trực tiếp HD95 voxel của MyoPS380 với HD95 mm ở đây.
Không gọi score cohort80 là reproduction điểm paper cohort95/challenge.

## Kiểm chứng

`python -m pytest -q` chạy suite SCAR; `pytest.ini` giới hạn discovery vào `tests`,
không quét test trong source clones dưới `tmp`. Tests bao gồm raw mapping, full cohort,
split isolation, cache fingerprint, native mm metrics, rejection geometry, real train →
resume → evaluation và lệnh pipeline trên synthetic NIfTI.

Run `outputs/myopspp_bc80/smoke/integration_smoke` sử dụng cấu hình `testing`, input 32 và
1 epoch để kiểm tra plumbing trên dữ liệu local; không dùng score của run này làm kết quả
benchmark nghiên cứu. Full model vẫn cần train theo budget đã chọn trước khi so sánh.
