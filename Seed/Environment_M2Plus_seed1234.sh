#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường & Huấn luyện SCAR (M2-Plus) - SEED 1234
# Chuẩn thông tin: NCKH/Seed/Enveronment.txt
# Tham khảo nạp dữ liệu & lưu SSD Colab: NCKH/Code/Chay_nnUNet_Colab_Nguyen.txt
# Repository: https://github.com/thanhquan123hi1/SCAR.git
# ==============================================================================

set -e

SEED=1234
MODEL_CONFIG="training/config/models/m2_plus.yaml"
ABLATION="M2-Plus"
OUT_DIR="/content/SCAR/outputs/runs/M2Plus_Seed_${SEED}"
ZIP_FILE="/content/M2Plus_Seed_${SEED}_Results.zip"

echo "=============================================================================="
echo "🚀 BẮT ĐẦU PIPELINE SCAR (M2-PLUS) - SEED: ${SEED}"
echo "📁 Thư mục lưu trữ trên SSD Colab: ${OUT_DIR}"
echo "=============================================================================="

# ==============================================================================
# [1/8] KIỂM TRA GPU NVIDIA & THIẾT LẬP DRIVER
# ==============================================================================
echo "=== [1/8] Kiểm tra và Kích hoạt GPU NVIDIA ==="
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
# [2/8] CLONE / CẬP NHẬT REPOSITORY SCAR
# ==============================================================================
echo "=== [2/8] Clone / Cập nhật Repository SCAR ==="
if [ ! -d "/content/SCAR" ]; then
    git clone https://github.com/thanhquan123hi1/SCAR.git /content/SCAR
else
    echo "Thư mục /content/SCAR đã tồn tại. Cập nhật nhánh mới nhất..."
    cd /content/SCAR && git pull origin main || true
fi
cd /content/SCAR

# ==============================================================================
# [3/8] CÀI ĐẶT DEPENDENCIES CHO SCAR
# ==============================================================================
echo "=== [3/8] Cài đặt Dependencies ==="
apt-get update -qq && apt-get install -y tmux zip unzip
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt gdown

# ==============================================================================
# [4/8] CHUẨN BỊ PRETRAINED CHECKPOINT (R50-ViT-B_16.npz)
# ==============================================================================
echo "=== [4/8] Chuẩn bị Pretrained Checkpoint (R50-ViT-B_16.npz) ==="
mkdir -p /content/SCAR/model/vit_checkpoint/imagenet21k
CKPT_FILE="/content/SCAR/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz"

CKPT_CANDIDATES=(
    "/content/drive/MyDrive/google_drive/ViT_checkpoint/R50-ViT-B_16.npz"
    "/content/drive/MyDrive/NCKH_V2/i_mmseg/ViT_checkpoint/R50-ViT-B_16.npz"
    "/content/drive/MyDrive/google_drive/R50-ViT-B_16.npz"
)

FOUND_CKPT=""
for c in "${CKPT_CANDIDATES[@]}"; do
    if [ -s "$c" ]; then
        FOUND_CKPT="$c"
        break
    fi
done

if [ -s "$CKPT_FILE" ]; then
    echo "⚡ Checkpoint đã có sẵn tại: $CKPT_FILE"
elif [ -n "$FOUND_CKPT" ]; then
    echo "Sao chép checkpoint từ Google Drive: $FOUND_CKPT..."
    cp "$FOUND_CKPT" "$CKPT_FILE"
    echo "✅ Sao chép checkpoint thành công!"
else
    echo "Đang tải R50-ViT-B_16.npz qua gdown..."
    gdown --id "1qJI7m6sM6deBZsSmcZjltHNWygRYagdD" -O "$CKPT_FILE" || {
        echo "Tải qua gdown thất bại, fallback sang server Google ViT Storage..."
        wget -c -O "$CKPT_FILE" https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
    }
    echo "✅ Tải checkpoint hoàn tất!"
fi

# ==============================================================================
# [5/8] CHUẨN BỊ DATASET MyoPS-380 TRÊN Ổ SSD CỤC BỘ COLAB
# ==============================================================================
echo "=== [5/8] Chuẩn bị Dataset MyoPS-380 trên SSD Colab ==="
LOCAL_DATASET="/content/SCAR/MyoPS380_dataset/Processed_data"
ZIP_PATH="/content/MyoPS380_dataset.zip"

if [ -d "$LOCAL_DATASET/bSSFP/train_npz" ] && [ -d "$LOCAL_DATASET/LGE/train_npz" ] && [ -d "$LOCAL_DATASET/T2w/train_npz" ] && [ $(ls "$LOCAL_DATASET/bSSFP/train_npz" 2>/dev/null | wc -l) -gt 1500 ]; then
    echo "⚡ Dataset Processed_data đã sẵn sàng trên SSD: $LOCAL_DATASET"
else
    # Kiểm tra các đường dẫn Drive khả dĩ theo chuẩn thực tế
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

# ==============================================================================
# [6/8] TẠO SPLIT BỆNH NHÂN (TRAIN/VAL/TEST) CHO SEED 1234
# ==============================================================================
echo "=== [6/8] Tạo split bệnh nhân riêng cho Seed ${SEED} ==="
cd /content/SCAR
SPLITS_DIR="/content/SCAR/data/processed/splits_seed${SEED}"
mkdir -p "$SPLITS_DIR"

python3 preprocessing/build_splits.py \
    --data-root "$LOCAL_DATASET" \
    --list-dir "$SPLITS_DIR" \
    --test-list preprocessing/splits/test_vol.txt \
    --seed ${SEED}

# ==============================================================================
# [7/8] HUẤN LUYỆN MÔ HÌNH M2-PLUS LƯU TRỰC TIẾP TRÊN SSD COLAB
# ==============================================================================
echo "=== [7/8] Bắt đầu huấn luyện mô hình M2-Plus (Seed ${SEED}) ==="
mkdir -p "$OUT_DIR"

python3 train.py \
    --config "$MODEL_CONFIG" \
    --ablation "$ABLATION" \
    --data-root "$LOCAL_DATASET" \
    --list-dir "$SPLITS_DIR" \
    --pretrained model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz \
    --output-dir "$OUT_DIR" \
    --seed ${SEED} \
    --batch-size 16 \
    --lr 0.0003 \
    --epochs 300 \
    --patience 40 \
    --amp auto

# ==============================================================================
# [8/8] ĐÁNH GIÁ 3D TRÊN 76 CA TEST & ĐÓNG GÓI ZIP TRÊN SSD
# ==============================================================================
echo "=== [8/8] Tự động đánh giá 3D trên 76 ca Test Volume ==="
BEST_CKPT="$OUT_DIR/checkpoints/best.pth"
TEST_RESULTS_DIR="$OUT_DIR/test_results"
mkdir -p "$TEST_RESULTS_DIR"

if [ -f "$BEST_CKPT" ]; then
    python3 training/evaluate.py \
        --checkpoint "$BEST_CKPT" \
        --data-root "$LOCAL_DATASET" \
        --split test_vol \
        --output-dir "$TEST_RESULTS_DIR"
    echo "✅ ĐÃ XUẤT BÁO CÁO PER_CASE.CSV VÀO: $TEST_RESULTS_DIR"
else
    echo "⚠️ CẢNH BÁO: Không tìm thấy checkpoint best.pth tại $BEST_CKPT"
fi

# Đóng gói toàn bộ thư mục kết quả thành file zip duy nhất trên SSD Colab
echo "=== Đóng gói toàn bộ kết quả thành file ZIP trên SSD Colab ==="
(cd "$OUT_DIR" && zip -r -q "$ZIP_FILE" .)

echo "=============================================================================="
echo "🎉 HOÀN TẤT TOÀN DIỆN THỰC NGHIỆM CHO SEED ${SEED} TRÊN SSD COLAB!"
echo "📁 Thư mục kết quả: ${OUT_DIR}"
echo "📦 File ZIP kết quả: ${ZIP_FILE}"
echo ""
echo "👉 LỆNH ĐƠN CHẠY TRÊN TERMINAL WSL / POWERSHELL MÁY BẠN ĐỂ TẢI VỀ:"
echo "   scp -P <PORT> root@<HOST>:${ZIP_FILE} ./"
echo "   (Hoặc trong VS Code: click chuột phải vào file ${ZIP_FILE} -> chọn Download)"
echo "=============================================================================="
