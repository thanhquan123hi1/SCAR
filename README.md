# SCAR — Multi-View LGE Cardiac MRI Scar & Anatomy Segmentation

<p align="center">
  <a href="https://colab.research.google.com/github/thanhquan123hi1/SCAR/blob/main/scar_pipeline.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Tests-146%20Passed%20(100%25)-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-red.svg" alt="Task">
</p>

---

## 📌 Giới thiệu dự án (Overview)

**SCAR** là pipeline học sâu (Deep Learning) hoàn chỉnh và tối ưu hóa chuẩn y khoa phục vụ phân đoạn (segmentation) **sẹo cơ tim (Myocardial Scar/Fibrosis)** và các cấu trúc giải phẫu tim từ chuỗi ảnh cộng hưởng từ tim **LGE (Late Gadolinium Enhancement) MRI** đa góc nhìn (**SAX 3D**, **2CH**, **4CH**, **RAS 2D**).

Mục tiêu chính:
1. **Phân đoạn chính xác cao**: Phát hiện và phân đoạn các tổn thương sẹo cơ tim có diện tích siêu nhỏ (<1% thể tích) mà không làm mất hình thái tổn thương.
2. **Định lượng lâm sàng chuẩn y khoa**: Tự động tính toán và báo cáo các chỉ số thể tích sẹo (**Scar Volume - mL**), khối lượng sẹo (**Scar Mass - g**), tỷ lệ phần trăm cơ tim bị sẹo (**Scar %**), và khoảng cách Hausdorff 95% (**HD95**).
3. **Thiết kế Modular & Reproducible**: Hỗ trợ huấn luyện 1-click trên máy trạm cục bộ (Windows / Linux) và Google Colab.

---

## 🌟 Kỹ thuật SOTA tích hợp trong SCAR

| # | Kỹ thuật / Module | Chi tiết giải pháp | Hiệu quả kỹ thuật & lâm sàng |
|---|---|---|---|
| 1 | **One-Hot Argmax Continuous Resampling** | Nội suy không gian `order=1` trên từng kênh one-hot độc lập rồi giải mã `argmax` | Triệt tiêu hoàn toàn hiện tượng giãn nở nhân tạo (dilation artifact) và giữ nguyên vi thể tổn thương sẹo nhỏ |
| 2 | **2.5D Context with Boundary Clamping** | Trích xuất 3 lát cắt liên tiếp $[s-1, s, s+1]$ với cơ chế edge clamping `[s0, s0, s1]` tại biên | Cung cấp ngữ cảnh qua các lát cắt (through-plane context) cho mô hình 2D mà không gây mất gradient ở lát cắt biên |
| 3 | **Weighted Rare-Class Sampler** | Lấy mẫu ưu tiên các lát cắt chứa sẹo (`rare_boost=2.0..5.0`, `foreground_boost=1.3`) | Giải quyết triệt để mất cân bằng dữ liệu cực đoan khi sẹo chỉ chiếm <1% thể tích |
| 4 | **SOTA ResUNet++ Architecture** | 4-stage skip connections với Attention Gate, SE-Block ($r=8$) và ASPP Bridge ($\text{rates}=[1, 2, 4, 8]$) | Tăng cường trường tiếp nhận (receptive field) và lọc nhiễu nền trên các skip connection |
| 5 | **Anisotropic 3D U-Net** | Khối chập `ConvBlock3D` với kernel $(1, 3, 3)$ và padding $(0, 1, 1)$ | Xử lý hoàn hảo chuỗi thể tích 3D SAX có khoảng cách lát cắt dày (thick-slice anisotropy) |
| 6 | **One-vs-Rest Compound Loss** | Kết hợp Focal Modulation ($\gamma=2.0$), Weighted BCE (`pos_weight=6.5` cho sẹo) và SoftDice Loss | Ổn định số học tuyệt đối ($<0.05$ trên empty slice $y=0$), triệt tiêu hiện tượng sụp đổ gradient |
| 7 | **Dynamic FOV-Calibrated HD95** | Tính khoảng cách phạt dựa trên đường chéo trường nhìn (patient FOV diagonal) | Đánh giá chính xác khoảng cách biên, hỗ trợ Median + IQR chống nhiễu ngoại lai |
| 8 | **3D Anatomical Constraint Postprocessing** | Hậu xử lý morphology 3D ép ràng buộc sẹo nằm trong cơ tim ($\text{Scar} \subseteq \text{Myo}$) | Loại bỏ 100% các điểm dự đoán dương tính giả ngoài cơ tim mà vẫn bảo toàn sẹo vùng mỏm tim (apical) |

---

## 🏷️ Label Maps (Bản đồ nhãn)

| Label | Cấu trúc giải phẫu | Tên tiếng Anh | Vai trò lâm sàng |
|:---:|---|---|---|
| **0** | Nền | Background | Vùng không thuộc cơ quan quan tâm |
| **1** | Khoang thất trái | LV Cavity | Đánh giá thể tích cuối tâm thu/tâm trương |
| **2** | Cơ tim thất trái | LV Myocardium | Cấu trúc giải phẫu cơ sở chứa sẹo |
| **3** | **Sẹo cơ tim** | **Myocardial Scar / Fibrosis** | **Mục tiêu chẩn đoán chính của LGE** |
| **4** | Khoang thất phải | RV Cavity | Quan sát trên các lát cắt SAX và 4CH |

---

## 📁 Cấu trúc thư mục repository

```text
SCAR/
├── data/                               ← Thư mục dữ liệu (được .gitignore)
│   ├── raw/CMR-MULTI/LGE_MULTI/        ← Đặt file NIfTI gốc vào đây
│   ├── processed/splits/               ← CSV phân chia train/val/test
│   └── processed/cache/                ← Cache .npz tiền xử lý siêu tốc
├── preprocessing/                      ← Module tiền xử lý dữ liệu chuẩn y khoa
│   ├── __init__.py
│   ├── preprocessing.py                 ← One-hot resampling, spatial transforms, invert
│   ├── build_splits.py                  ← Phân chia bệnh nhân, chống rò rỉ dữ liệu (0% leak)
│   ├── process_and_save.py              ← Batch preprocessing & caching
│   ├── verify.py                        ← Sanity check kiểm tra tiền xử lý
│   └── config.yaml                      ← Cấu hình target shape & spacing các view
├── training/                           ← Module huấn luyện & đánh giá mô hình
│   ├── config/                          ← File YAML cấu hình
│   │   ├── base.yaml                    ← Tham số huấn luyện chung (lr, epochs, loss)
│   │   └── models/                      ← Cấu hình riêng từng model & view (2D, 2.5D, 3D)
│   ├── dataset/                         ← PyTorch Dataset & Weighted Sampler
│   │   ├── lge_dataset.py               ← 2D/2.5D/3D Dataset loader với boundary clamping
│   │   └── sampler.py                   ← Rare-class balanced sampler
│   ├── models/                          ← Kiến trúc mạng nơ-ron
│   │   ├── __init__.py                  ← Model registry factory
│   │   ├── modules.py                   ← Attention Gate, SE-Block, ASPP, Stem Block
│   │   ├── resunet_plus_plus.py         ← SOTA ResUNet++ 2D/2.5D
│   │   ├── unet_2d.py                   ← Standard 2D U-Net
│   │   └── unet_3d.py                   ← Anisotropic 3D U-Net
│   ├── loss/                            ← Loss functions (One-vs-Rest, SoftDice, Focal BCE)
│   ├── metrics/                         ← Metrics (Multi-class Dice, Dynamic FOV HD95)
│   ├── postprocess/                     ← Ràng buộc giải phẫu 3D (Anatomical Rules)
│   ├── trainer/                         ← PyTorch Training Loop, AMP, Early Stopping
│   ├── train.py                         ← Entrypoint huấn luyện mô hình
│   ├── evaluate.py                      ← Entrypoint đánh giá & định lượng sẹo lâm sàng
│   └── predict.py                       ← Entrypoint suy luận trên file mới
├── tests/                              ← Bộ kiểm thử tự động toàn diện (146 tests)
│   ├── test_refactor_baseline.py        ← Master baseline verification test suite
│   └── test_*_stress.py / test_*.py     ← Adversarial & Stress testing suites
├── outputs/runs/                        ← Checkpoints, log metrics, hình ảnh (được .gitignore)
├── analysis/                            ← Thư mục lưu trữ biểu đồ và so sánh mô hình
├── figures/                             ← Thư mục lưu trữ hình ảnh bài báo / báo cáo
├── run_all.py                           ← 1-Click Pipeline Runner hoàn chỉnh
├── scar_pipeline.ipynb                  ← Jupyter Notebook chạy trên Google Colab
├── install.sh / train.sh / predict.sh   ← Script chạy nhanh trên Linux/macOS
├── train.ps1                            ← Script chạy nhanh trên Windows PowerShell
├── requirements.txt                     ← Danh sách dependencies
└── README.md                            ← Tài liệu hướng dẫn chính thức
```

---

## 🚀 Hướng dẫn cài đặt (Installation)

### 1. Yêu cầu môi trường
- Python >= 3.10 (khuyến nghị 3.11 hoặc 3.12)
- PyTorch >= 2.0 (hỗ trợ CUDA nếu có GPU)

### 2. Cài đặt dependencies
```bash
git clone https://github.com/thanhquan123hi1/SCAR.git
cd SCAR

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## ⚡ Hướng dẫn chạy 1-Click (Quickstart)

### Cách 1: Chạy 1-Click qua `run_all.py` (Khuyên dùng)
Lệnh này tự động thực hiện từ đầu đến cuối: Kiểm tra dữ liệu -> Tạo splits -> Huấn luyện mô hình -> Đánh giá & Định lượng sẹo:

```bash
# Huấn luyện ResUNet++ 2.5D trên góc nhìn 2CH:
python run_all.py --config training/config/models/resunet_plus_plus_2d.yaml --run-id resunet_2ch_sota

# Huấn luyện 3D U-Net trên góc nhìn SAX:
python run_all.py --config training/config/models/unet_3d.yaml --run-id unet3d_sax_sota
```

### Cách 2: Chạy trên Windows PowerShell
```powershell
.\train.ps1 -Config training/config/models/resunet_plus_plus_2d.yaml -RunId resunet_2ch_sota
```

### Cách 3: Chạy trên Linux / macOS
```bash
bash train.sh training/config/models/resunet_plus_plus_2d.yaml resunet_2ch_sota
```

### Cách 4: Chạy trên Google Colab
1. Mở file [`scar_pipeline.ipynb`](scar_pipeline.ipynb) hoặc nhấp vào badge **Open in Colab** ở đầu trang.
2. Chọn GPU Runtime (**Runtime > Change runtime type > T4 GPU**).
3. Chạy tuần tự các cells để mount Drive, load dữ liệu, huấn luyện và trực quan hóa kết quả.

---

## 🛠️ Quy trình chạy chi tiết từng bước (Step-by-Step Workflow)

### Bước 1: Phân chia tập dữ liệu (Build Patient Splits)
Tạo phân chia Train / Validation / Test đảm bảo **0% rò rỉ bệnh nhân (zero patient leakage)** giữa các góc nhìn:
```bash
python preprocessing/build_splits.py \
    --data-root data/LGE_MULTI \
    --output data/processed/splits \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### Bước 2: Tiền xử lý & Caching siêu tốc (Preprocess & Cache)
Chuyển đổi ảnh NIfTI về chuẩn không gian, nội suy one-hot cho mask và lưu cache `.npz`:
```bash
python preprocessing/process_and_save.py \
    --data-root data/LGE_MULTI \
    --splits-dir data/processed/splits \
    --output-dir data/processed/cache
```

### Bước 3: Huấn luyện mô hình (Training)
```bash
python training/train.py \
    --config training/config/models/resunet_plus_plus_2d.yaml \
    --run-id resunet_2ch_sota
```

### Bước 4: Đánh giá & Định lượng sẹo lâm sàng (Evaluation & Clinical Quantification)
Tính toán Dice scores, Dynamic FOV HD95, thể tích sẹo (mL) và khối lượng sẹo (g):
```bash
python training/evaluate.py \
    --config outputs/runs/resunet_2ch_sota/config_snapshot.yaml \
    --checkpoint outputs/runs/resunet_2ch_sota/checkpoints/best.pt \
    --split validation
```

### Bước 5: Suy luận trên dữ liệu mới (Inference)
```bash
python training/predict.py \
    --config outputs/runs/resunet_2ch_sota/config_snapshot.yaml \
    --checkpoint outputs/runs/resunet_2ch_sota/checkpoints/best.pt \
    --split test
```

---

## 🧪 Kiểm thử tự động (Automated Test Suite)

Repository được trang bị bộ kiểm thử tự động toàn diện gồm **146 test cases** kiểm tra tính ổn định toán học, chống rò rỉ dữ liệu, độ chính xác của ResUNet++, 3D UNet anisotropic convolutions, Loss stability, và HD95 calibration:

```bash
# Chạy toàn bộ test suite:
pytest

# Chạy riêng Master Verification Test Suite:
pytest tests/test_refactor_baseline.py -v
```

---

## ➕ Thêm mô hình hoặc cấu hình mới

1. **Định nghĩa mạng**: Thêm file mô hình vào `training/models/your_model.py`.
2. **Đăng ký registry**: Đăng ký tên mô hình vào `MODEL_REGISTRY` trong `training/models/__init__.py`.
3. **Tạo file cấu hình**: Tạo `training/config/models/your_model.yaml`.
4. **Chạy thử nghiệm**:
   ```bash
   python training/train.py --config training/config/models/your_model.yaml --run-id your_experiment
   ```

---

## 📖 Tham khảo & Trích dẫn (References)

- **Dataset**: CMR-MULTI Challenge & LGE Cardiac MRI multi-view dataset.
- **Preprocessing standard**: Điều chỉnh và nâng cấp từ [cmr-multi-cinema](https://github.com/sinaamirrajab/cmr-multi-cinema).
- **Architectures**:
  - *ResUNet++*: Jha et al., "ResUNet++: An Advanced Architecture for Medical Image Segmentation"
  - *Squeeze-and-Excitation Networks*: Hu et al., CVPR 2018
  - *Atrous Spatial Pyramid Pooling (ASPP)*: Chen et al., IEEE TPAMI 2017

