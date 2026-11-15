# Research Handoff — Lightweight M2-Plus Optimization on MyoPS++

> Tài liệu bàn giao toàn bộ quá trình thực nghiệm, kết quả định lượng, chẩn đoán lỗi, mã nguồn đã chỉnh sửa, vị trí tài nguyên đã sao lưu và kế hoạch bước tiếp theo để phiên làm việc sau có thể tiếp tục ngay mà không cần train/test lại hay phân tích lại từ đầu.
> **Cập nhật lần cuối:** `2026-11-15 03:35 UTC`

---

## 1. Bối cảnh & Protocol Thực nghiệm Chung

- **Dataset profile:** `myopspp_unet2d_roi128_1mm` *(giữ tách biệt hoàn toàn với `myops380` dùng đơn vị voxel)*
  - **Split cố định (`--no-val`):** `64` bệnh nhân train (`296` lát cắt 2D), `0` val, `16` bệnh nhân `test_vol` (cache tại `data/myopspp_unet2d_roi128_1mm/cache`).
  - **Không gian & nhãn:** ảnh `128x128`, spacing XY `1.0 mm`, nhãn canonical `0=background`, `1=normal_myocardium`, `2=edema`, `3=scar` (`edema_inclusive = {2,3}`, `myocardial_ring = {1,2,3}`). Đánh giá trên predicted ROI grid, single-pass `argmax_logits` (`single_pass_argmax_v1`), HD95/ASD tính theo `mm`.
- **Cấu hình training chuẩn (cố định để so sánh công bằng):**
  - `seed: 1234`, `epochs: 300`, `batch_size: 16`, `accum_steps: 1`
  - Optimizer: `AdamW` (`lr: 0.0003333`, `weight_decay: 0.0001`), scheduler polynomial (`power: 0.9`)
  - Loss gốc: `0.2 * CE + 0.8 * BatchDice(weights=[0.5, 1.0, 1.0, 1.0], normalization='classes', smooth=1e-5)` (`aar_weight=0.0`, `scar_weight=0.0`, `anat_contrast_weight=0.0`)
  - Precision: `bf16` (GPU `NVIDIA A100-SXM4-40GB`), `--pretrained training/pretrained/R50-ViT-B_16.npz`
  - Data augmentation: `RandomGenerator` mặc định (protocol I-MMSeg trong `training/dataset/myops_dataset.py`)

---

## 2. Kết quả So sánh 3 Model Nền (Seed `1234`, `test_vol` `16` ca)

| Model | Params | Scar Dice / HD95 | Edema Excl Dice / HD95 | Edema Incl Dice / HD95 | Normal Myo | Myo Ring | Avg Path Excl Dice / HD95 | Avg Path Incl Dice / HD95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M2-Plus`** (`m2_plus.yaml`) | `63,748,336` | `67.34% / 9.83` | `39.10% / 25.21` | `70.66% / 17.14` | `76.12%` | `87.61%` | `53.22% / 17.52` | `69.00% / 13.49` |
| **`M2-Plus-Paper-Formula`** | `48,016,342` | `66.27% / 8.90` | `43.03% / 20.69` | `70.90% / 16.16` | `75.71%` | `87.39%` | `54.65% / 14.79` | `68.58% / 12.53` |
| **`Lightweight-M2-Plus`** | **`14,670,711`** | **`67.45% / 12.66`** | `42.56% / 20.86` | **`71.61% / 14.74`** | **`76.41%`** | `87.42%` | **`55.01% / 16.76`** | **`69.53% / 13.70`** |

$\Rightarrow$ **`Lightweight-M2-Plus`** nhẹ nhất (`14.67M`, giảm ~77% tham số) và đạt `Avg Excl` / `Avg Incl` cao nhất.

---

## 3. Giai đoạn 1 — Ablation Từng Thành phần trên `Lightweight-M2-Plus`

| Mã | Config (`training/config/models/`) | Thay đổi so với Baseline | Params | MACs (G) | Scar Dice / HD95 | Edema Excl Dice / HD95 | Edema Incl | NormMyo | Ring | Avg Excl Dice / HD95 | Avg Incl Dice / HD95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline`** | `lightweight_m2_plus` | Đủ thành phần | `14,670,711` | `3.9692` | `67.45 / 12.66` | `42.56 / 20.86` | `71.61` | `76.41` | `87.42` | `55.01 / 16.76` | `69.53 / 13.70` |
| **`E1`** | `lightweight_no_sspanet` | Bỏ `OptimizedSSPANet` | `13,286,007` | `3.9615` | `67.51 / 12.33` | `43.17 / 19.53` | `71.88` | `75.28` | `87.31` | `55.34 / 15.93` | `69.70 / 13.03` |
| **`E2`** | `lightweight_no_skip_gates` | Bỏ `SSPA_AttentionGate` | `14,629,172` | `3.9455` | `66.02 / 10.60` | `40.62 / 21.71` | `71.25` | `77.16` | `87.46` | `53.32 / 16.16` | `68.63 / 12.29` |
| **`E3`** | `lightweight_no_pos_enc` | Bỏ 2D Sinusoidal PE | `14,670,711` | `3.9692` | `68.27 / 9.78` | **`43.57` / 21.94** | `71.90` | `76.27` | `87.55` | `55.92 / 15.86` | `70.08 / 12.11` |
| **`E4`** | `lightweight_no_film` | Bỏ `FiLM` adapters | `14,659,959` | `3.9692` | `67.10 / 10.12` | `41.26 / 21.25` | `71.40` | `76.82` | `87.28` | `54.18 / 15.69` | `69.25 / 12.05` |
| **`E5`** | **`lightweight_no_cross_attention`** | **Bỏ `Cross-Attention MHA` (tự bypass PE, giữ `SSPANet`)** | **`14,144,375`** | `3.9315` | **`70.84 / 8.54`** | `43.30 / 21.48` | **`72.46`** | **`77.84`** | **`88.05`** | **`57.07 / 15.01`** | **`71.65 / 11.61`** |
| **`E6`** | `lightweight_resnet346` | Cắt Stage 3 `[3,4,9]→[3,4,6]` | `11,319,159` | `3.3275` | `65.79 / 11.75` | `39.96 / 22.87` | `71.06` | `75.16` | `87.29` | `52.88 / 17.31` | `68.43 / 13.57` |

**Kết luận GĐ1:** `Skip Gates`, `FiLM`, và độ sâu `ResNet [3,4,9]` đều hữu ích (`E2/E4/E6` giảm mạnh). Cắt `Cross-Attention MHA` (`E5`) trên bottleneck `8x8` giúp `Scar +3.39 pp`, `Avg Incl +2.12 pp`, `Avg Excl +2.06 pp`, giảm `Scar HD95` `-4.12 mm`.

---

## 4. Giai đoạn 2 — Đối chiếu Kiến trúc Pruned (`P2-1..P2-4`) & Chốt Nền `E5`

| Mã | Config (`training/config/models/`) | Kiến trúc | Params | MACs (G) | Scar Dice / HD95 | Edema Excl Dice / HD95 | Edema Incl | NormMyo | Ring | Avg Excl Dice / HD95 | Avg Incl Dice / HD95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5`** *(Chốt)* | `lightweight_no_cross_attention` | Giữ `SSPANet`, bỏ `CrossAttn` | `14,144,375` | `3.9315` | `70.84 / 8.54` | `43.30 / 21.48` | `72.46` | **`77.84`** | `88.05` | `57.07 / 15.01` | **`71.65 / 11.61`** |
| **`P2-1`** | `lightweight_p2_clean_pruned` | `E5` bỏ `OptimizedSSPANet` | `12,759,671` | `3.9238` | `68.51 / 9.70` | `44.77 / 21.39` | **`72.67`** | `77.38` | **`88.42`** | `56.64 / 15.55` | `70.59 / 12.25` |
| **`P2-2`** | `lightweight_p2_contrast_fusion` | `P2-1 + LocalModalityContrastFusion` | `13,471,767` | `3.9346` | **`70.94` / 9.85** | `42.29 / 22.24` | `71.70` | `76.61` | `87.77` | `56.61 / 16.05` | `71.32 / 12.81` |
| **`P2-3`** | `lightweight_p2_adaptive_skip` | `P2-1 + TriModalSkipFusion` | `12,894,032` | `3.9288` | `69.80 / 10.87` | `43.06 / 23.76` | `71.80` | `77.58` | `88.00` | `56.43 / 17.32` | `70.80 / 13.32` |
| **`P2-4`** | `lightweight_p2_v2` | `P2-1 + Contrast + Skip + Boundary` | `13,618,096` | `4.1336` | `70.28 / 11.29` | **`44.97` / 24.19** | `72.56` | `76.86` | `87.91` | **`57.62` / 17.74** | `71.42 / 14.01` |

**Kết luận GĐ2:** Giữ `OptimizedSSPANet` trên `E5` (`14,144,375` params) giúp `Scar +2.33 pp`, `Scar HD95` tốt hơn `1.16 mm` và `Avg Incl +1.06 pp`. Chốt **`E5` làm kiến trúc gốc**.

---

## 5. Giai đoạn 3 — Các Thử nghiệm Cải tiến trên nền `E5` (`2026-11-15`)

Tất cả các thử nghiệm giữ nguyên 100% dataset split (`64/0/16`), augmentation, optimizer, scheduler, `seed=1234`, `300 epochs`, `bf16`.

### 5.1 Kết quả Đợt 1 & Đợt 2 (Đã hoàn tất 300 epochs + `test_vol` 16 ca)

#### Bảng tổng hợp Train (ep 300) & Test Mean (*kèm Median trong ngoặc*)

| Đợt | Mã | Config (`training/config/models/`) | Thay đổi trên nền `E5` | Params | Train `mDice` / `loss` | Scar Dice *(med)* / HD95 | Edema Dice *(med)* / HD95 | Edema Incl | NormMyo | Ring Dice / HD95 | Avg Excl Dice *(med)* / HD95 | Avg Incl Dice *(med)* / HD95 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref** | **`E5`** | `lightweight_no_cross_attention` | Baseline đã chốt | `14,144,375` | `0.909 / 0.053` | **`70.84%`** *(74.39)* / **`8.54`** | `43.30%` *(45.13)* / `21.48` *(20.04)* | `72.46%` | `77.84%` / `4.99` | `88.05%` / `2.81` | **`57.07%`** *(58.75)* / **`15.01`** | **`71.65%`** *(75.15)* / **`11.61`** |
| **W1** | **`E5-A1`** | `lightweight_e5_a1_skip` | `skip_fusion: modality_weighted` (`TriModalSkipFusion`) | `14,278,736` | `0.9088 / 0.0536` | `70.38%` *(75.27)* / `10.95` | `42.58%` *(43.75)* / **`21.28`** *(18.64)* | `71.99%` | `77.10%` / `5.00` | `88.06%` / `2.77` | `56.48%` *(54.55)* / `16.11` | `71.19%` *(76.18)* / `12.84` |
| **W1** | **`E5-A2`** | `lightweight_e5_a2_boundary` | `decoder_refinement: boundary` (`BoundaryRefinementBlock` bản cũ) | `14,156,375` | `0.9077 / 0.0539` | `70.04%` *(74.19)* / `11.11` | `41.75%` *(46.41)* / `23.02` *(20.36)* | `71.75%` | **`78.39%`** / `5.02` | **`88.21%`** / `2.73` | `55.89%` *(54.40)* / `17.07` | `70.89%` *(75.99)* / `13.53` |
| **W1** | **`E5-L1`** | `lightweight_e5_l1_hier_loss` | `aar_weight=0.25, scar_weight=0.25` (`positive`) | `14,144,375` | `0.9001 / 0.1035` | `70.41%` *(74.76)* / `10.84` | `43.01%` *(49.87)* / `22.76` *(19.34)* | `72.10%` | `76.44%` / `5.08` | `87.31%` / `3.09` | `56.71%` *(57.80)* / `16.80` | `71.26%` *(75.00)* / `13.65` |
| **W2** | **`E5-B1`** | `lightweight_e5_b1_res_contrast` | `bottleneck_fusion: residual_contrast` (`ResidualContrastBottleneck`, zero-init) | `14,311,415` | `0.9090 / 0.0532` | `70.18%` *(**75.72**)* / `8.99` *(med **5.33**)* | `42.29%` *(44.82)* / `22.27` *(21.06)* | `71.89%` | `77.44%` / **`4.76`** | `88.01%` / `2.78` | `56.23%` *(57.39)* / `15.63` | `71.03%` *(75.09)* / `12.40` |
| **W2** | **`E5-S2`** | `lightweight_e5_s2_spatial_skip` | `skip_fusion: spatial_residual` (`SpatialResidualSkipFusion`, zero-init + `(C+P+T)/3` anchor) | `14,241,757` | `0.9075 / 0.0544` | `70.36%` *(74.08)* / `9.72` *(med 5.61)* | **`43.35%`** *(**47.12**)* / `22.08` *(med **18.75**)* | **`72.51%`** *(med **76.35**)* | `77.37%` / **`4.76`** | `87.79%` / **`2.66`** | `56.86%` *(**59.59**)* / `15.90` | `71.43%` *(75.07)* / `12.80` |
| **W2** | **`E5-L2`** | `lightweight_e5_l2_anat_loss` | `anat_contrast_weight=0.15` (V1 chia `ring_area` — lỗi spike đầu train) | `14,144,375` | `0.7852 / 0.1727` | `65.80%` *(69.98)* / `12.06` | `40.94%` *(39.45)* / `22.89` *(20.02)* | `71.49%` | `74.40%` / `5.48` | `86.13%` / `3.25` | `53.37%` *(52.25)* / `17.47` | `68.64%` *(74.22)* / `14.09` |

#### Chi tiết Precision / Recall trên `test_vol` (W1 & W2)
- `E5`: Scar `P 82.14% / R 65.73%`, Edema `P 44.14% / R 48.70%`, EdemaIncl `P 76.94% / R 70.55%`, Ring `P 89.69% / R 86.80%`
- `E5-A1`: Scar `P 81.03% / R 66.29%`, Edema `P 44.11% / R 48.32%`
- `E5-A2`: Scar `P 81.69% / R 66.61%`, Edema `P 43.67% / R 47.45%`
- `E5-L1`: Scar `P 80.20% / R 67.12%`, Edema `P 42.26% / R 50.96%`
- `E5-B1`: Scar `P 80.79% / R 66.85%`, Edema `P 44.68% / R 47.52%`, EdemaIncl `P 75.57% / R 70.59%`, Ring `P 89.48% / R 86.89%`
- `E5-S2`: Scar `P 81.08% / R 66.12%`, Edema `P 44.43% / R 50.06%`, EdemaIncl `P 75.86% / R 71.51%`, Ring `P 89.43% / R 86.56%`
- `E5-L2`: Scar `P 73.93% / R 65.32%`, Edema `P 39.83% / R 49.38%`, EdemaIncl `P 71.38% / R 74.38%`, Ring `P 85.86% / R 86.85%`

---

### 5.2 Phân tích Chẩn đoán Gốc rễ W1 + W2 (Để tránh lặp lại các sai lầm cũ)

1. **Outlier `Case3003 z=2` (`GT Scar = 0`) chi phối mean `Scar HD95` và `AvgExcl`:**
   - Lát cắt `z=2` của `Case3003` có nhiễu/cường độ bất thường làm 10/11 mô hình lịch sử tạo cụm dương tính giả Scar tách rời; tình cờ chỉ `E5` seed `1234` tránh được.
   - Toàn bộ 6 biến thể W1 + W2 đều tạo FP Scar trên `Case3003 z=2`:
     - `E5-A1`: `150` voxels Scar giả (`Case3003` Scar `65.1% / HD95 36.27 mm`)
     - `E5-A2`: `194` voxels Scar giả (`68.3% / 36.78 mm`)
     - `E5-L1`: `202` voxels Scar giả (`65.5% / 36.67 mm`)
     - `E5-B1`: `126` voxels Scar giả (`68.9% / 31.33 mm`)
     - `E5-S2`: `91` voxels Scar giả (`74.0% / 34.03 mm`)
     - `E5-L2`: `148` voxels Scar giả (`67.2% / 34.01 mm`)
   - Khi nhìn vào **median** (kháng outlier `Case3003 z=2`): **`E5-S2` thắng `E5` ở `AvgExcl` (`59.59%` vs `58.75%`), `Edema Dice` (`47.12%` vs `45.13%`), `Edema HD95` (`18.75` vs `20.04 mm`)**; **`E5-B1` thắng `E5` ở `Scar Dice` (`75.72%` vs `74.39%`) & `Scar HD95` (`5.33` vs `5.52 mm`)**; và cả `B1`/`S2` đều giảm `Normal Myo HD95` (`4.76` vs `4.99 mm`) & `Myo Ring HD95` (`2.78 / 2.66` vs `2.81 mm`).
2. **Tại sao W1 (`A1`, `A2`, `L1`) thoái lui:**
   - `E5-A1` (`TriModalSkipFusion`): hạ tĩnh trọng số nhánh `CINE`, làm mờ biên giải phẫu vùng cơ tim mỏng.
   - `E5-A2` (`BoundaryRefinementBlock` bản W1 không có zero-init `gamma`): làm trơn quá mức vùng `Edema` mảnh (`41.75%`, HD95 `23.02 mm`).
   - `E5-L1` (`aar=0.25, scar=0.25`): tổng trọng số bệnh lý quá nặng, ép `Ring` (`87.31%`) và `NormMyo` (`76.44%`) giảm.
3. **Tại sao `E5-L2` thất bại & phát hiện cần lưu ý:**
   - Trong `training/loss/losses.py` bản W2, phép chia `out_ring_leak` theo `ring_area` không chặn biên gây spike `anat_contrast_loss = 15.7384` ở ep 1 → model sập về dự đoán nền (`mean_dice = 0.000` ở ep 10–20) → ep 300 chỉ bò lại được `train mean_dice = 0.7852` (`grad_norm = 1.224 ± 0.059`).
   - Dù hỏng tổng thể, `E5-L2` lại kéo được ca Edema yếu `Case3007` lên **`35.9%`** (so với `11.6%` ở `B1`, `10.8%` ở `S2`, `13.7%` ở `A1`, `7.5%` ở `A2`, `17.7%` ở `L1`), gợi ý loss tương phản giải phẫu vẫn có ích nếu chặn biên toán học `[0, 1]`.
4. **Tại sao `E5-B1` & `E5-S2` chưa bật rõ trên mean:**
   - Kiểm tra checkpoint ep 300 cho thấy các hệ số residual zero-init học khá bảo thủ:
     - `E5-B1` (`cross_attention.contrast_gamma`): `mean=0.00150`, `absmean=0.04185`, `min=-0.12600`, `max=0.11374`
     - `E5-S2` (`skip_fusions.0/1/2.gamma`): `absmean=0.01702 / 0.01962 / 0.01272` (`[-0.06060, +0.05219]`).
   - Ngoài ra, ở W2 `ResidualContrastBottleneck.__init__` không cô lập RNG khi khởi tạo module phụ, làm lệch chuỗi RNG khởi tạo các lớp sau và thứ tự batch so với `E5` (khác biệt ngẫu nhiên này ảnh hưởng mạnh tới ca nhạy cảm như `Case3003 z=2`).

---

### 5.3 Đợt 3 (Sửa lỗi từ W2 — đang chạy ở epoch ~234–246/300 tại `03:35 UTC`)

Mã nguồn hiện tại đã được cập nhật cho Đợt 3 (`B1R`, `L3`, `Combo`):
- **Cô lập RNG khi thêm module residual (`torch.random.fork_rng`):** Đã thêm vào `ResidualContrastBottleneck` (`training/models/modules/m2_plus.py`), `SpatialResidualSkipFusion` (`training/models/modules/fusion.py`), và `BoundaryRefinementBlock` (`training/models/modules/decoder.py`, đồng thời thêm zero-init `self.gamma` vào `BoundaryRefinementBlock`). Mục tiêu: giữ nguyên hoàn toàn quỹ đạo RNG khởi tạo ban đầu như `E5`.
- **Sửa `anat_contrast_loss` sang dạng chặn cứng `[0, 1]` (`training/loss/losses.py`):** Bỏ chia `ring_area`, thay bằng:
  `0.35 * batch_scar_dice + 0.35 * batch_aar_dice + 0.15 * batch_ring_dice + 0.15 * (p_scar * y_edema).sum() / (y_edema.sum() + 1.0)` (`anat_contrast_weight: 0.10`).

| Mã | Config (`training/config/models/`) | Mô tả | Params | Tiến độ ghi nhận lúc `03:34:58 UTC` |
| :--- | :--- | :--- | :---: | :--- |
| **`E5-B1R`** | `lightweight_e5_b1r_res_contrast` | `ResidualContrastBottleneck` với `fork_rng` khớp seed `E5` | `14,311,415` | `ep 244/300`, `mDice=0.8979`, `loss=0.0602` |
| **`E5-L3`** | `lightweight_e5_l3_bounded_loss` | Kiến trúc `E5` + Bounded `anat_contrast_weight=0.10` | `14,144,375` | `ep 246/300`, `mDice=0.8978`, `loss=0.0692`, `ep1_anat=0.742 -> 0.0878` *(không collapse)* |
| **`E5-Combo`** | `lightweight_e5_combo_arch_loss` | `ResidualContrast` + `SpatialResidualSkip` + `Boundary(gamma)` + `BoundedLoss(0.10)` | `14,420,797` | `ep 234/300`, `mDice=0.8952`, `loss=0.0713`, `ep1_anat=0.741 -> 0.0888` |

*(Cần cập nhật kết quả `test_vol` của Đợt 3 ngay sau khi 3 tiến trình hoàn tất lúc ~`03:42 UTC`.)*

---

## 6. Kiểm kê Tài nguyên, Backup Drive & Hướng dẫn Tiếp tục Phiên sau

### 6.1 Vị trí lưu trữ trên Google Drive (`/content/NCKH_V2` = `gdrive:NCKH_V2`)
- **Trọng số gốc `E5` (`MyoPS++`):** `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_Lightweight_No_CrossAttn/` (`lightweightnocrossattn_seed1234.pt`, logs, masks).
- **Phase 1 & 2 (`E1..E6`, `P2-1..P2-4`):** `Lightweight_Phase1_Ablation/`, `Lightweight_Phase2_Arch/`.
- **Phase 3 logs & metrics (KHÔNG lưu `.pt` vì chưa có biến thể vượt `E5`):**
  `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/Lightweight_Phase3_E5_Optimization/`
  - `configs/`: tất cả YAML `lightweight_e5_*.yaml` + `lightweight_no_cross_attention.yaml`
  - `per_variant_logs/{E5-A1,E5-A2,E5-L1,E5-B1,E5-S2,E5-L2}/`: trọn bộ `config.json`, `summary.json`, `metrics.csv`, `metrics.jsonl`, `train.log`, `splits/`, và `test_results/` (`metrics.json`, `per_case.csv`, `test.log`, 16 cặp mask dự đoán `Case*_pred.{nii.gz,npz}`).
  - File tổng hợp phân tích nhanh: `phase3_summary_metrics.csv`, `phase3_per_case_long.csv`, `phase3_per_case_wide.csv`.
- **Tập `MyoPS380` (tách biệt, đơn vị voxel):** `/content/NCKH_V2/Checkpoints/MyoPS380/E5_Lightweight_No_CrossAttn/` (`Nov 15 00:40 UTC`).

### 6.2 Kiểm tra Code & Trạng thái Git hiện tại
- Bộ unit test mô hình (`PYTHONPATH=. python3 -m pytest -q tests/test_lightweight_ablation.py tests/test_lightweight_phase2.py tests/test_shared_backbone_and_skip_gates.py`): **41 passed**.
- Đã chỉnh sửa sạch các module trong `training/` (`config_utils.py`, `losses.py`, `models/__init__.py`, `m2_plus_net.py`, `decoder.py`, `fusion.py`, `m2_plus.py`, `train.py`, `trainer.py`) và thêm 9 YAML `training/config/models/lightweight_e5_*.yaml`.

### 6.3 Checklist Khởi động Nhanh cho Phiên Làm Việc Sau (Prompt gợi ý)
Khi mở phiên mới, chỉ cần đọc `RESEARCH_HANDOFF.md` này và thực hiện theo thứ tự:
1. **Kiểm tra Đợt 3 (`B1R`, `L3`, `Combo`):**
   - Kiểm tra `outputs/myopspp_unet2d_roi128_1mm/runs/{e5_b1r_res_contrast_seed1234,e5_l3_bounded_loss_seed1234,e5_combo_arch_loss_seed1234}/logs_*/test_results/metrics.json` (hoặc trên Drive nếu đã đồng bộ sau `03:42 UTC`).
   - Nếu có biến thể thắng `E5` (`Avg Path Excl > 57.07%` và không hỏng `Ring/NormMyo`), **giữ checkpoint `.pt` đó và backup lên Drive**.
   - Nếu cả Đợt 3 vẫn hòa/nhỉnh nhẹ theo median nhưng bị dao động bởi outlier `Case3003 z=2`: chạy kiểm chứng **đa seed (`seed 1234, 1024, 1098`)** cho `E5` vs ứng viên tốt nhất (`E5-S2` / `E5-B1R` / `E5-L3` / `E5-Combo`) để loại nhiễu 1 seed, đồng thời khảo sát cơ chế loại FP Scar ngoài phân bố ở lát `z=2` của `Case3003` và kéo Edema trên ca yếu (`Case3007`, `Case2013`, `Case3044`) bằng file dự đoán `.nii.gz` đã lưu sẵn trong `per_variant_logs/*/test_results/`.

### 6.4 Lệnh Tái lập Chuẩn (trên `myopspp_unet2d_roi128_1mm`)
*(Cần có `data/myopspp_unet2d_roi128_1mm/cache` và `training/pretrained/R50-ViT-B_16.npz`.)*

```bash
# Train E5 hoặc bất kỳ biến thể Phase 3 nào (ví dụ lightweight_e5_s2_spatial_skip.yaml)
python3 train.py \
  --config training/config/models/lightweight_no_cross_attention.yaml \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --pretrained training/pretrained/R50-ViT-B_16.npz \
  --seed 1234 --epochs 300 --batch-size 16 --img-size 128 --no-val --patience 0 --min-delta 0 --amp bf16

# Đánh giá trên test_vol (16 bệnh nhân)
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_Lightweight_No_CrossAttn/lightweightnocrossattn_seed1234.pt \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --amp bf16 --save-predictions
```
