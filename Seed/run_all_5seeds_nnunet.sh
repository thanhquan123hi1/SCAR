#!/bin/bash
# ==============================================================================
# MASTER RUNNER: HUẤN LUYỆN & ĐÁNH GIÁ TRỌN GÓI 5 SEED nnU-Net TRÊN 1 SESSION
# Các seed: 1234 | 1098 | 1024 | 2026 | 6868
# Bộ dữ liệu: MyoPS-380 (Fixed Split: 243 Train / 61 Val / 76 Test)
# Hợp đồng kỹ thuật: NCKH/ChanhThuc/Promt.txt
# Repository: https://github.com/thanhquan123hi1/SCAR.git (nhánh feat/nnunet-official)
# ==============================================================================

set -e

SEEDS=(1234 1098 1024 2026 6868)
MODEL_CONFIG="training/config/models/nnunet.yaml"
ABLATION="NNUNET"
MASTER_DIR="/content/All_5Seeds_nnUNet_Results"
MASTER_ZIP="/content/All_5Seeds_nnUNet_Results.zip"

echo "=============================================================================="
echo "🚀 KHỞI ĐỘNG MASTER RUNNER 5 SEED nnU-Net CHÍNH THỨC TRÊN CÙNG 1 SESSION"
echo "🎯 Danh sách 5 Seed: ${SEEDS[*]}"
echo "📁 Thư mục lưu trữ tổng hợp: ${MASTER_DIR}"
echo "📦 File ZIP tổng hợp cuối cùng: ${MASTER_ZIP}"
echo "=============================================================================="

# ==============================================================================
# [1/6] KIỂM TRA GPU NVIDIA & THIẾT LẬP DRIVER
# ==============================================================================
echo "=== [1/6] Kiểm tra và Kích hoạt GPU NVIDIA ==="
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
# [2/6] CLONE / CẬP NHẬT REPOSITORY SCAR (NHÁNH feat/nnunet-official)
# ==============================================================================
echo "=== [2/6] Clone / Cập nhật Repository SCAR (nhánh feat/nnunet-official) ==="
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
# [3/6] CÀI ĐẶT DEPENDENCIES CHO SCAR
# ==============================================================================
echo "=== [3/6] Cài đặt Dependencies ==="
apt-get update -qq && apt-get install -y tmux zip unzip
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt gdown

# ==============================================================================
# [4/6] CHUẨN BỊ DATASET MyoPS-380 TRÊN Ổ SSD CỤC BỘ COLAB (CHỈ 1 LẦN DUY NHẤT)
# ==============================================================================
echo "=== [4/6] Chuẩn bị Dataset MyoPS-380 trên SSD Colab (dùng chung cho cả 5 seed) ==="
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
# [5/6] KIỂM TRA TÍNH TOÀN VẸN CỦA BỘ CHIA DỮ LIỆU CHUẨN 380 CA
# ==============================================================================
echo "=== [5/6] Xác thực bộ chia dữ liệu cố định (Audit Split Provenance) ==="
SPLITS_DIR="/content/SCAR/preprocessing/splits/myops380_official"

python3 -c "
import hashlib
from pathlib import Path

split_dir = Path('$SPLITS_DIR')
required = {'train.txt': 1578, 'val.txt': 394, 'val_vol.txt': 61, 'test_vol.txt': 76}
manifest = {}

for fname, expected_lines in required.items():
    fpath = split_dir / fname
    if not fpath.is_file():
        raise FileNotFoundError(f'Thiếu file split: {fpath}')
    lines = [line.strip() for line in fpath.read_text(encoding='utf-8').splitlines() if line.strip()]
    count = len(lines)
    sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
    if count != expected_lines:
        raise ValueError(f'{fname}: Mong đợi {expected_lines} dòng, thực tế {count} dòng!')
    manifest[fname] = {'lines': count, 'sha256': sha256}
    print(f'✅ {fname:12s}: {count:4d} mục | SHA256: {sha256[:16]}...')

import json
manifest_path = Path('/content/split_provenance_manifest.json')
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('✅ Đã tạo file xác thực nguồn gốc dữ liệu: /content/split_provenance_manifest.json')
"

# Chuẩn bị thư mục tổng hợp
mkdir -p "$MASTER_DIR"
cp /content/split_provenance_manifest.json "$MASTER_DIR/"
cp -r "$SPLITS_DIR" "$MASTER_DIR/splits_reference"

# ==============================================================================
# [6/6] VÒNG LẶP HUẤN LUYỆN & ĐÁNH GIÁ LẦN LƯỢT 5 SEED
# ==============================================================================
TOTAL_SEEDS=${#SEEDS[@]}
CURRENT_IDX=0

for SEED in "${SEEDS[@]}"; do
    CURRENT_IDX=$((CURRENT_IDX + 1))
    echo ""
    echo "=============================================================================="
    echo "▶️ BẮT ĐẦU SEED [$CURRENT_IDX/$TOTAL_SEEDS]: SEED ${SEED}"
    echo "=============================================================================="

    OUT_DIR="/content/SCAR/outputs/runs/nnUNet_Seed_${SEED}"
    SINGLE_ZIP="/content/nnUNet_Seed_${SEED}_Results.zip"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"

    # 1. Huấn luyện nnU-Net
    echo "--- [Seed ${SEED}] Huấn luyện mô hình (300 Epochs) ---"
    python3 train.py         --config "$MODEL_CONFIG"         --ablation "$ABLATION"         --data-root "$LOCAL_DATASET"         --list-dir "$SPLITS_DIR"         --output-dir "$OUT_DIR"         --seed ${SEED}         --batch-size 16         --lr 0.0003         --epochs 300         --patience 0         --amp auto

    # 2. Đánh giá trên 76 ca Test Volume
    echo "--- [Seed ${SEED}] Đánh giá 3D trên 76 ca Test Volume ---"
    BEST_CKPT="$OUT_DIR/checkpoints/best.pth"
    if [ ! -f "$BEST_CKPT" ] && [ -f "$OUT_DIR/checkpoints/best.pt" ]; then
        BEST_CKPT="$OUT_DIR/checkpoints/best.pt"
    fi
    TEST_RESULTS_DIR="$OUT_DIR/test_results"
    mkdir -p "$TEST_RESULTS_DIR"

    if [ -f "$BEST_CKPT" ]; then
        python3 training/evaluate.py             --checkpoint "$BEST_CKPT"             --data-root "$LOCAL_DATASET"             --list-dir "$SPLITS_DIR"             --split test_vol             --save-predictions             --output-dir "$TEST_RESULTS_DIR"
        echo "✅ [Seed ${SEED}] Hoàn thành đánh giá test!"
    else
        echo "⚠️ CẢNH BÁO: Không tìm thấy checkpoint best tại $BEST_CKPT"
    fi

    # 3. Tạo file zip riêng của từng seed (phòng hờ)
    (cd "$OUT_DIR" && zip -r -q "$SINGLE_ZIP" .)
    echo "📦 [Seed ${SEED}] Đã lưu file zip riêng: $SINGLE_ZIP"

    # 4. Sao chép kết quả của seed vào thư mục tổng hợp master
    SEED_TARGET_DIR="$MASTER_DIR/seed_${SEED}"
    mkdir -p "$SEED_TARGET_DIR"
    cp -r "$OUT_DIR/checkpoints" "$SEED_TARGET_DIR/"
    cp -r "$OUT_DIR/test_results" "$SEED_TARGET_DIR/"
    cp -r "$OUT_DIR"/logs_* "$SEED_TARGET_DIR/" 2>/dev/null || cp -r "$OUT_DIR/logs" "$SEED_TARGET_DIR/" 2>/dev/null || true

    echo "✅ [Seed ${SEED}] ĐÃ HOÀN TẤT THÀNH CÔNG!"
done

# ==============================================================================
# TỔNG HỢP VÀ TÍNH TOÁN BẢNG EXCEL CHÍNH THỨC CHO 5 SEED
# ==============================================================================
echo ""
echo "=============================================================================="
echo "📊 BẢNG TỔNG HỢP KẾT QUẢ ĐIỀN EXCEL CHÍNH THỨC CHO 5 SEED (nnU-Net)"
echo "=============================================================================="

python3 -c "
import json
import numpy as np
from pathlib import Path

master_dir = Path('$MASTER_DIR')
seeds = [1234, 1098, 1024, 2026, 6868]

rows = []
for s in seeds:
    p = master_dir / f'seed_{s}' / 'test_results' / 'metrics.json'
    if not p.is_file():
        p = master_dir / f'seed_{s}' / 'test_results' / 'summary.json'
    if not p.is_file():
        print(f'⚠️ Thiếu file summary.json của seed {s}')
        continue
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
    rows.append({
        'seed': s,
        's_dice': s_dice, 's_hd': s_hd,
        'e_dice': e_dice, 'e_hd': e_hd,
        'm_dice': m_dice, 'm_hd': m_hd,
        'a_dice': a_dice, 'a_hd': a_hd,
    })

print(f'| {"nnU-Net":8s} | {"Scar Dice":10s} | {"Scar HD":8s} | {"Edema Dice":10s} | {"Edema HD":8s} | {"Myo Dice":10s} | {"Myo HD":8s} | {"Avg Dice":10s} | {"Avg HD":8s} |')
print('|----------|------------|---------|------------|---------|------------|---------|------------|---------|')

for r in rows:
    print(f'| {r["seed"]:8d} | {r["s_dice"]:9.2f}% | {r["s_hd"]:7.2f} | {r["e_dice"]:9.2f}% | {r["e_hd"]:7.2f} | {r["m_dice"]:9.2f}% | {r["m_hd"]:7.2f} | {r["a_dice"]:9.2f}% | {r["a_hd"]:7.2f} |')

if rows:
    avg_s_dice = np.mean([r['s_dice'] for r in rows])
    avg_s_hd   = np.mean([r['s_hd'] for r in rows])
    avg_e_dice = np.mean([r['e_dice'] for r in rows])
    avg_e_hd   = np.mean([r['e_hd'] for r in rows])
    avg_m_dice = np.mean([r['m_dice'] for r in rows])
    avg_m_hd   = np.mean([r['m_hd'] for r in rows])
    avg_a_dice = np.mean([r['a_dice'] for r in rows])
    avg_a_hd   = np.mean([r['a_hd'] for r in rows])
    print(f'| {"Avg":8s} | {avg_s_dice:9.2f}% | {avg_s_hd:7.2f} | {avg_e_dice:9.2f}% | {avg_e_hd:7.2f} | {avg_m_dice:9.2f}% | {avg_m_hd:7.2f} | {avg_a_dice:9.2f}% | {avg_a_hd:7.2f} |')

print('')
print('📋 DỮ LIỆU ĐỂ COPY THẲNG VÀO EXCEL (TAB-SEPARATED):')
for r in rows:
    print(f'{r["s_dice"]:.2f}\t{r["s_hd"]:.2f}\t{r["e_dice"]:.2f}\t{r["e_hd"]:.2f}\t{r["m_dice"]:.2f}\t{r["m_hd"]:.2f}\t{r["a_dice"]:.2f}\t{r["a_hd"]:.2f}')
if rows:
    print(f'{avg_s_dice:.2f}\t{avg_s_hd:.2f}\t{avg_e_dice:.2f}\t{avg_e_hd:.2f}\t{avg_m_dice:.2f}\t{avg_m_hd:.2f}\t{avg_a_dice:.2f}\t{avg_a_hd:.2f}')

# Lưu ra file CSV mở trực tiếp bằng Excel
csv_path = master_dir / 'benchmark_summary.csv'
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('Seed,Scar_Dice,Scar_HD,Edema_Dice,Edema_HD,Myo_Dice,Myo_HD,Avg_Pathology_Dice,Avg_Pathology_HD\n')
    for r in rows:
        f.write(f"{r['seed']},{r['s_dice']:.2f},{r['s_hd']:.2f},{r['e_dice']:.2f},{r['e_hd']:.2f},{r['m_dice']:.2f},{r['m_hd']:.2f},{r['a_dice']:.2f},{r['a_hd']:.2f}\n")
    if rows:
        f.write(f"Avg,{avg_s_dice:.2f},{avg_s_hd:.2f},{avg_e_dice:.2f},{avg_e_hd:.2f},{avg_m_dice:.2f},{avg_m_hd:.2f},{avg_a_dice:.2f},{avg_a_hd:.2f}\n")
print(f'✅ Đã lưu bảng CSV mở bằng Excel tại: {csv_path}')
"

# Đóng gói toàn bộ thư mục master thành 1 file ZIP duy nhất
echo ""
echo "=== Đóng gói toàn bộ 5 Seed thành file ZIP Master duy nhất ==="
(cd "$MASTER_DIR" && zip -r -q "$MASTER_ZIP" .)

echo "=============================================================================="
echo "🎉 HOÀN TẤT TOÀN DIỆN THỰC NGHIỆM 5 SEED TRÊN SSD COLAB!"
echo "📁 Thư mục kết quả tổng hợp: ${MASTER_DIR}"
echo "📦 File ZIP tổng hợp duy nhất: ${MASTER_ZIP}"
echo ""
echo "👉 LỆNH ĐƠN CHẠY TRÊN POWERSHELL MÁY BẠN ĐỂ TẢI TOÀN BỘ VỀ MÁY:"
echo "   scp s1:${MASTER_ZIP} D:/NCKH/"
echo "=============================================================================="
