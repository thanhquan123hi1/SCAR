#!/bin/bash
# ==============================================================================
# Script Cài Đặt Môi Trường & Huấn Luyện nnU-Net Chính Thức - SEED 1234
# Bộ dữ liệu: MyoPS-380 (Fixed Split: 243 Train / 61 Val / 76 Test)
# Hợp đồng kỹ thuật: NCKH/ChanhThuc/Promt.txt
# Repository: https://github.com/thanhquan123hi1/SCAR.git (nhánh feat/nnunet-official)
# ==============================================================================

set -e

SEED=1234
MODEL_CONFIG="training/config/models/nnunet.yaml"
ABLATION="NNUNET"
OUT_DIR="/content/SCAR/outputs/runs/nnUNet_Seed_${SEED}"
ZIP_FILE="/content/nnUNet_Seed_${SEED}_Results.zip"

echo "=============================================================================="
echo "🚀 BẮT ĐẦU PIPELINE nnU-Net CHÍNH THỨC (MyoPS-380) - SEED: ${SEED}"
echo "📁 Thư mục lưu trữ trên SSD Colab: ${OUT_DIR}"
echo "=============================================================================="

# ==============================================================================
# [1/7] KIỂM TRA GPU NVIDIA & THIẾT LẬP DRIVER
# ==============================================================================
echo "=== [1/7] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại kết nối GPU!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# ==============================================================================
# [2/7] CLONE / CẬP NHẬT REPOSITORY SCAR (NHÁNH feat/nnunet-official)
# ==============================================================================
echo "=== [2/7] Clone / Cập nhật Repository SCAR (nhánh feat/nnunet-official) ==="
if [ ! -d "/content/SCAR" ]; then
    git clone https://github.com/thanhquan123hi1/SCAR.git /content/SCAR
    cd /content/SCAR
    git checkout feat/nnunet-official 2>/dev/null || git checkout -b feat/nnunet-official origin/feat/nnunet-official 2>/dev/null || true
else
    echo "Thư mục /content/SCAR đã tồn tại. Chuyển sang nhánh feat/nnunet-official..."
    cd /content/SCAR
    git fetch origin
    git checkout feat/nnunet-official 2>/dev/null || true
    git pull origin feat/nnunet-official 2>/dev/null || true
fi
cd /content/SCAR

# ==============================================================================
# [3/7] CÀI ĐẶT DEPENDENCIES CHO SCAR
# ==============================================================================
echo "=== [3/7] Cài đặt Dependencies ==="
apt-get update -qq && apt-get install -y tmux zip unzip
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt gdown

# ==============================================================================
# [4/7] BỎ QUA PRETRAINED ViT CHECKPOINT (nnU-Net train from scratch)
# ==============================================================================
echo "=== [4/7] nnU-Net là Pure CNN: Khởi tạo He-Normal từ đầu, không dùng ViT Checkpoint ==="

# ==============================================================================
# [5/7] CHUẨN BỊ DATASET MyoPS-380 TRÊN Ổ SSD CỤC BỘ COLAB
# ==============================================================================
echo "=== [5/7] Chuẩn bị Dataset MyoPS-380 trên SSD Colab ==="
LOCAL_DATASET="/content/SCAR/MyoPS380_dataset/Processed_data"
ZIP_PATH="/content/MyoPS380_dataset.zip"

if [ -d "$LOCAL_DATASET/bSSFP/train_npz" ] && [ -d "$LOCAL_DATASET/LGE/train_npz" ] && [ -d "$LOCAL_DATASET/T2w/train_npz" ] && [ $(ls "$LOCAL_DATASET/bSSFP/train_npz" 2>/dev/null | wc -l) -gt 1500 ]; then
    echo "⚡ Dataset Processed_data đã sẵn sàng trên SSD: $LOCAL_DATASET"
else
    DATA_DRIVE_CANDIDATES=(
        "/content/drive/MyDrive/google_drive/MyoPS380_dataset"
        "/content/drive/MyDrive/google_drive/MyoPS380_dataset/Processed_data"
        "/content/drive/MyDrive/NCKH_V2/i_mmseg/MyoPS380_dataset"
        "/content/drive/MyDrive/NCKH_V2/i_mmseg/MyoPS380_dataset/Processed_data"
        "/content/MyoPS380_dataset"
    )

    DATA_ZIP_CANDIDATES=(
        "/content/drive/MyDrive/google_drive/MyoPS380_dataset.zip"
        "/content/drive/MyDrive/NCKH_V2/i_mmseg/MyoPS380_dataset.zip"
        "/content/MyoPS380_dataset.zip"
        "/content/MyoPS380.zip"
        "/content/drive/MyDrive/MyoPS380.zip"
    )

    FOUND_DATA_DIR=""
    for d in "${DATA_DRIVE_CANDIDATES[@]}"; do
        if [ -d "$d/bSSFP/train_npz" ] || [ -d "$d/Processed_data/bSSFP/train_npz" ]; then
            FOUND_DATA_DIR="$d"
            break
        fi
    done

    FOUND_ZIP=""
    for z in "${DATA_ZIP_CANDIDATES[@]}"; do
        if [ -s "$z" ]; then
            FOUND_ZIP="$z"
            break
        fi
    done

    if [ -n "$FOUND_DATA_DIR" ]; then
        echo "Đang sao chép Processed_data từ Drive sang SSD Colab ($FOUND_DATA_DIR)..."
        mkdir -p /content/SCAR/MyoPS380_dataset
        if [ -d "$FOUND_DATA_DIR/Processed_data" ]; then
            cp -r "$FOUND_DATA_DIR/Processed_data" /content/SCAR/MyoPS380_dataset/
        else
            cp -r "$FOUND_DATA_DIR" /content/SCAR/MyoPS380_dataset/Processed_data
        fi
        echo "✅ Sao chép Dataset sang SSD hoàn tất!"
    else
        if [ ! -s "$ZIP_PATH" ]; then
            if [ -n "$FOUND_ZIP" ]; then
                echo "Sao chép file zip từ: $FOUND_ZIP..."
                cp "$FOUND_ZIP" "$ZIP_PATH"
            else
                echo "Đang tải Dataset ZIP qua gdown (ID: 1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b)..."
                gdown --id "1UhD3-ncoARjcv_1IqUcMa_G4oaU9pY-b" -O "$ZIP_PATH"
            fi
        fi

        echo "Đang giải nén Dataset vào SSD /content/SCAR/MyoPS380_dataset..."
        mkdir -p /content/SCAR/MyoPS380_dataset
        unzip -q -o "$ZIP_PATH" -d /content/SCAR/MyoPS380_dataset/
        chmod -R u+rwX /content/SCAR/MyoPS380_dataset
        echo "✅ Giải nén Dataset hoàn tất!"
    fi
fi

if [ -d "/content/SCAR/MyoPS380_dataset/Processed_data/bSSFP/train_npz" ]; then
    LOCAL_DATASET="/content/SCAR/MyoPS380_dataset/Processed_data"
elif [ -d "/content/SCAR/MyoPS380_dataset/bSSFP/train_npz" ]; then
    LOCAL_DATASET="/content/SCAR/MyoPS380_dataset"
fi

# ==============================================================================
# [6/7] HUẤN LUYỆN MÔ HÌNH nnU-Net TRÊN SSD COLAB (300 EPOCHS)
# KHÓA SPLIT CHUẨN: 243 Train / 61 Val / 76 Test (KHÔNG CHẠY build_splits.py)
# ==============================================================================
echo "=== [6/7] Bắt đầu huấn luyện mô hình nnU-Net (Seed ${SEED}) ==="
mkdir -p "$OUT_DIR"

SPLITS_DIR="/content/SCAR/preprocessing/splits/myops380_official"

python3 train.py \
    --config "$MODEL_CONFIG" \
    --ablation "$ABLATION" \
    --data-root "$LOCAL_DATASET" \
    --list-dir "$SPLITS_DIR" \
    --output-dir "$OUT_DIR" \
    --seed ${SEED} \
    --batch-size 16 \
    --lr 0.0003 \
    --epochs 300 \
    --patience 0 \
    --amp auto

# ==============================================================================
# [7/7] ĐÁNH GIÁ 3D TRÊN 76 CA TEST & TRÍCH XUẤT SỐ LIỆU CHO BẢNG EXCEL
# ==============================================================================
echo "=== [7/7] Tự động đánh giá 3D trên 76 ca Test Volume ==="
BEST_CKPT="$OUT_DIR/checkpoints/best.pth"
if [ ! -f "$BEST_CKPT" ] && [ -f "$OUT_DIR/checkpoints/best.pt" ]; then
    BEST_CKPT="$OUT_DIR/checkpoints/best.pt"
fi
TEST_RESULTS_DIR="$OUT_DIR/test_results"
mkdir -p "$TEST_RESULTS_DIR"

if [ -f "$BEST_CKPT" ]; then
    python3 training/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --data-root "$LOCAL_DATASET" \
        --list-dir "$SPLITS_DIR" \
        --split test_vol \
        --save-predictions \
        --output-dir "$TEST_RESULTS_DIR"
    echo "✅ ĐÃ XUẤT BÁO CÁO PER_CASE.CSV VÀ TEST PREDICTIONS VÀO: $TEST_RESULTS_DIR"

    echo ""
    echo "=============================================================================="
    echo "📊 KẾT QUẢ ĐIỀN BẢNG EXCEL CHO SEED ${SEED}:"
    echo "==============================================================================="
    python3 -c "
import json
from pathlib import Path

p = Path('$TEST_RESULTS_DIR/metrics.json')
if not p.is_file():
    p = Path('$TEST_RESULTS_DIR/summary.json')
if not p.is_file():
    print('Chưa tìm thấy summary.json')
else:
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    s_dice = d.get('scar', {}).get('mean_dice', 0.0) * 100
    s_hd   = d.get('scar', {}).get('mean_hd95_voxel', 0.0)
    e_dice = d.get('edema', {}).get('mean_dice', 0.0) * 100
    e_hd   = d.get('edema', {}).get('mean_hd95_voxel', 0.0)
    m_dice = d.get('myocardial_ring', {}).get('mean_dice', 0.0) * 100
    m_hd   = d.get('myocardial_ring', {}).get('mean_hd95_voxel', 0.0)
    a_dice = d.get('avg_pathology_dice', 0.0) * 100
    a_hd   = d.get('avg_pathology_hd95_voxel', 0.0)

    print('SEED | Scar (Dice / HD) | Edema (Dice / HD) | Myo (Dice / HD) | Avg (Dice / HD)')
    print(f'${SEED} | {s_dice:.2f}% / {s_hd:.2f} | {e_dice:.2f}% / {e_hd:.2f} | {m_dice:.2f}% / {m_hd:.2f} | {a_dice:.2f}% / {a_hd:.2f}')
    print('')
    print('Dòng copy-paste vào Excel (phân cách bằng dấu Tab):')
    print(f'{s_dice:.2f}\t{s_hd:.2f}\t{e_dice:.2f}\t{e_hd:.2f}\t{m_dice:.2f}\t{m_hd:.2f}\t{a_dice:.2f}\t{a_hd:.2f}')
"
    echo "=============================================================================="
else
    echo "⚠️ CẢNH BÁO: Không tìm thấy checkpoint best tại $BEST_CKPT"
fi

# Đóng gói toàn bộ thư mục kết quả thành file zip duy nhất trên SSD Colab
echo "=== Đóng gói toàn bộ kết quả thành file ZIP trên SSD Colab ==="
(cd "$OUT_DIR" && zip -r -q "$ZIP_FILE" .)

echo "=============================================================================="
echo "🎉 HOÀN TẤT TOÀN DIỆN THỰC NGHIỆM CHO SEED ${SEED} TRÊN SSD COLAB!"
echo "📁 Thư mục kết quả: ${OUT_DIR}"
echo "📦 File ZIP kết quả: ${ZIP_FILE}"
echo ""
echo "👉 LỆNH ĐƠN CHẠY TRÊN POWERSHELL MÁY BẠN ĐỂ TẢI VỀ:"
echo "   scp <HOST>:${ZIP_FILE} D:/NCKH/"
echo "=============================================================================="
