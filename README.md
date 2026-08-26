# SCAR — LGE Cardiac MRI Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thanhquan123hi1/SCAR/blob/main/scar_pipeline.ipynb)

Repo NCKH phục vụ segmentation LGE (Late Gadolinium Enhancement) MRI tim đa góc nhìn (SAX 3D, 2CH, 4CH, RAS 2D).
Mục tiêu: segmentation scar/fibrosis và các cấu trúc tim từ ảnh LGE NIfTI và định lượng sẹo cơ tim lâm sàng (Scar volume mL, Scar mass g).

## Cấu trúc thư mục

```text
SCAR/
├── data/
│   ├── raw/CMR-MULTI/LGE_MULTI/   ← Đặt NIfTI gốc vào đây (read-only)
│   └── processed/splits/          ← CSV train/val/test (tự sinh)
├── preprocessing/
│   ├── preprocessing.py            ← Core: resize, normalize, crop/pad
│   ├── config.yaml                 ← Params: shape, spacing, percentiles
│   ├── build_splits.py             ← Sinh CSV splits từ raw data
│   └── verify.py                   ← Sanity check preprocessing
├── training/
│   ├── config/
│   │   ├── base.yaml               ← Config chung (lr, epochs, loss...)
│   │   └── models/
│   │       ├── unet_3d.yaml        ← Config riêng cho 3D U-Net
│   │       └── unet_2d.yaml        ← Config riêng cho 2D U-Net
│   ├── models/                     ← Định nghĩa kiến trúc mạng
│   ├── dataset/lge_dataset.py      ← PyTorch Dataset wrapping preprocessing
│   ├── loss/                       ← Dice, CE+Dice...
│   ├── metrics/                    ← Dice score, IoU...
│   ├── trainer/trainer.py          ← Training loop + early stopping
│   ├── train.py                    ← Entry point training
│   └── predict.py                  ← Entry point inference
├── analysis/                       ← So sánh experiments, vẽ biểu đồ
├── notebooks/                      ← Khám phá dữ liệu
├── outputs/runs/                   ← Checkpoints + metrics (gitignore)
├── figures/                        ← Hình cho paper
├── train.sh / predict.sh           ← Chạy nhanh
└── requirements.txt
```

## Label maps (LGE SAX, 2CH, 4CH)

| Label | Cấu trúc | Ghi chú |
|---|---|---|
| 0 | Background | Nền |
| 1 | LV Cavity | Khoang thất trái |
| 2 | LV Myo | Cơ tim thất trái |
| 3 | **Scar** | Sẹo cơ tim (Mục tiêu chính) |
| 4 | RV Cavity | Khoang thất phải (cho SAX, 4CH) |

## Kỹ thuật SOTA tích hợp trong Standard Baseline

1. **One-vs-Rest Compound Loss:** Kết hợp BCE (với `pos_weight=6.5` riêng cho sẹo), Focal Modulation (`gamma=2.0`) và Binary SoftDice Loss để triệt tiêu hiện tượng sụp đổ gradient của sẹo.
2. **Weighted Rare-Class Sampler (`rare_boost=5.0`):** Ưu tiên lấy mẫu các lát cắt chứa sẹo gấp 5 lần, giải quyết triệt để mất cân bằng dữ liệu cực đoan (<1% sẹo).
3. **2.5D Context / 3-Channel Input (`in_channels=3`):** Cung cấp ngữ cảnh 3 lát cắt $[s-1, s, s+1]$ cho chuỗi volume đa lát cắt (như SAX) hoặc biểu diễn 3-kênh đồng nhất cho các mặt cắt đơn 2D (2CH/4CH/RAS).
4. **Kiến trúc SOTA ResUNet++:** Tích hợp Squeeze-and-Excitation (SE-Block), Attention Gate chuẩn (lọc nhiễu trên Skip Connection) và Atrous Spatial Pyramid Pooling (ASPP Bridge).
5. **Hậu xử lý Ràng buộc Giải phẫu (`Anatomical Constraints`):** Ép sẹo chỉ được xuất hiện trong vùng cơ tim ($\text{Scar} \subseteq \text{Myocardium}$) và lọc bỏ nhiễu giả ngoài tim.

## Chạy 1-Click Toàn Diện (Khuyên Dùng)

```bash
# Huấn luyện mô hình ResUNet++ 2.5D chuẩn SOTA trên góc nhìn 2CH:
python run_all.py --config training/config/models/resunet_plus_plus_2d.yaml --run-id resunet_2ch_sota

# Huấn luyện mô hình 3D U-Net trên góc nhìn SAX:
python run_all.py --config training/config/models/unet_3d.yaml --run-id unet3d_sax_sota
```

## Chạy trên Google Colab

1. Nhấp vào nút **Open in Colab** ở đầu trang hoặc mở file `scar_pipeline.ipynb`.
2. Chọn Runtime GPU (**Runtime > Change runtime type > T4 GPU**).
3. Chạy tuần tự các cell: Mount Google Drive -> Clone/Pull Repo mới nhất -> Chạy pipeline và xem đồ thị `dice_scar`.

## Workflow Chi Tiết Từng Bước

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Sinh CSV splits
python preprocessing/build_splits.py \
    --data-root data/LGE_MULTI \
    --output data/processed/splits

# 3. Tiền xử lý & Caching siêu tốc
python preprocessing/process_and_save.py \
    --data-root data/LGE_MULTI \
    --splits-dir data/processed/splits \
    --output-dir data/processed/cache

# 4. Huấn luyện mô hình ResUNet++ 2.5D SOTA
python training/train.py \
    --config training/config/models/resunet_plus_plus_2d.yaml \
    --run-id resunet_2ch_sota

# 5. Đánh giá chi tiết & Định lượng sẹo lâm sàng
python training/evaluate.py \
    --config outputs/runs/resunet_2ch_sota/config_snapshot.yaml \
    --checkpoint outputs/runs/resunet_2ch_sota/checkpoints/best.pt \
    --split validation
```

## Thêm model mới

1. Tạo `training/models/ten_model.py`
2. Đăng ký vào `training/models/__init__.py`
3. Tạo `training/config/models/ten_model.yaml`
4. Chạy `python training/train.py --config training/config/models/ten_model.yaml --run-id ...`

## Nguồn preprocessing

Preprocessing LGE được tham khảo và điều chỉnh từ:
[cmr-multi-cinema](https://github.com/sinaamirrajab/cmr-multi-cinema) — `src/cmr_multi/data/preprocessing.py`
