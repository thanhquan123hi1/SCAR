# SCAR — LGE Cardiac MRI Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thanhquan123hi1/SCAR/blob/main/3gay.ipynb)

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

## Label maps (LGE SAX)

| Label | Cấu trúc |
|-------|----------|
| 0 | background |
| 1 | lv_cavity |
| 2 | lv_myo |
| 3 | **scar** |
| 4 | rv_cavity |

## Chạy 1-Click Toàn Diện (Khuyên Dùng)

```bash
# Tự động thực hiện 4 bước: Sinh Splits CSV -> Caching .npz -> Train -> Đánh giá & Định lượng sẹo
python run_all.py --config training/config/models/unet_3d.yaml --run-id unet3d_lge_sax_run01
```

## Chạy trên Google Colab

1. Nhấp vào nút **Open in Colab** ở đầu trang hoặc mở file [`3gay.ipynb`](file:///d:/NCKH/SCAR/3gay.ipynb).
2. Chọn Runtime GPU (**Runtime > Change runtime type > T4 GPU**).
3. Chạy tuần tự các cell: Mount Google Drive -> Clone Repo -> Cài dependencies -> Chạy pipeline và xem đồ thị trực quan.

## Workflow Chi Tiết Từng Bước

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Đặt dữ liệu LGE vào:
#    data/LGE_MULTI/SAX_TR/image/*.nii.gz
#                          /anno/*.nii.gz

# 3. Sinh CSV splits
python preprocessing/build_splits.py \
    --data-root data/LGE_MULTI \
    --output data/processed/splits

# 4. Tiền xử lý & Caching siêu tốc
python preprocessing/process_and_save.py \
    --data-root data/LGE_MULTI \
    --splits-dir data/processed/splits \
    --output-dir data/processed/cache

# 5. Huấn luyện mô hình (3D SAX hoặc 2D LAX)
python training/train.py \
    --config training/config/models/unet_3d.yaml \
    --run-id unet3d_lge_sax_exp01

# 6. Đánh giá chi tiết & Định lượng sẹo
python training/evaluate.py \
    --config outputs/runs/unet3d_lge_sax_exp01/config_snapshot.yaml \
    --checkpoint outputs/runs/unet3d_lge_sax_exp01/checkpoints/best.pt \
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
