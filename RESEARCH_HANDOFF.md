# Research Handoff — Lightweight M2-Plus Optimization on MyoPS++

> Tài liệu bàn giao toàn bộ quá trình thực nghiệm, kết quả định lượng, chẩn đoán lỗi, mã nguồn đã chỉnh sửa, vị trí tài nguyên đã sao lưu và kế hoạch bước tiếp theo để phiên làm việc sau có thể tiếp tục ngay mà không cần train/test lại hay phân tích lại từ đầu.
> **Mốc tốt nhất hiện tại (`seed=1234`, `test_vol` MyoPS++ — cập nhật đến hết Round 7, 23 trial):**
> - **#1 Tổng thể theo tiêu chí Pass & HD95 (`6/8` pass — Round 4):** **`R4_L5_CalibTversky`** ([lightweight_e5_b1r_s3_l5_calib_tversky.yaml](training/config/models/lightweight_e5_b1r_s3_l5_calib_tversky.yaml), `15,726,455` params, `9.4695 G` FLOPs) — **`AvgPathIncl = 72.09% / 10.72 mm`** *(duy nhất vượt ngưỡng `≤ 10.93 mm`)*, **`Scar = 70.42% / 7.65 mm`** *(Best Scar HD95)*, `EdemaIncl = 73.76% / 13.79 mm`, `MyoRing = 88.37% / 2.74 mm`, `FP3003 = 22`.
> - **#2 Tổng thể & #1 Scar Dice / MyoRing HD95 (`5/8` pass — Round 6):** **`R6_L8_MildScaleKarimi`** ([lightweight_e5_b1r_s3_l8_mild_scale_karimi.yaml](training/config/models/lightweight_e5_b1r_s3_l8_mild_scale_karimi.yaml), `15,726,455` params, `9.4695 G` FLOPs) — **`AvgPathIncl = 72.90% / 11.06 mm`**, **`Scar = 72.29% / 7.78 mm`**, **`MyoRing = 88.46% / 2.70 mm`** *(chỉ còn cách ngưỡng `2.67 mm` đúng `0.03 mm`)*, `FP3003 = 26`.
> - **#3 Tổng thể & #1 Kiến trúc thuần (`5/8` pass — Round 4):** **`R4_A3_B2SA_S3AnatMean`** ([lightweight_e5_b2sa_s3_anat_mean.yaml](training/config/models/lightweight_e5_b2sa_s3_anat_mean.yaml), `17,292,842` params, `9.8039 G` FLOPs) — **`AvgPathIncl = 72.61% / 11.71 mm`**, **`EdemaIncl = 74.57%`** *(Best)*, `AvgPathExcl = 58.34%`, `Scar = 70.65% / 9.53 mm`.
> - **Kỷ lục Dice toàn chiến dịch (`4/8` pass — Round 7):** **`R7_C6_B2GSA_S3AnatMean_RingCtx_L9`** ([lightweight_e5_b2gsa_s3_anat_mean_ring_ctx_l9.yaml](training/config/models/lightweight_e5_b2gsa_s3_anat_mean_ring_ctx_l9.yaml), `17,329,643` params, `10.9474 G` FLOPs) — **`AvgPathIncl Dice = 72.97%`**, **`AvgPathExcl Dice = 59.25%`**, **`EdemaExcl Dice = 46.77%`**.
> - **Mốc nền Wave 4–6 (Base Loss, tham chiếu trên MyoPS380):** **`E5-B1R-S3-NormSkip`** (`15.73M / 9.47G`, MyoPS380: `Scar = 72.05% / 4.30 vox`, `AvgPathIncl = 73.39%`) và **`E5-B1R-S3AnatMean`** (Wave 6B).
> **Cập nhật lần cuối:** `2026-11-19 14:20 UTC` (Hoàn thành Wave 7 + Rounds 1–7 Autoresearch `23` trial, Round 8 đang chạy; đã sao lưu toàn bộ log/metric + Top 3 checkpoints + git bundle lên Google Drive)

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

## 5. Giai đoạn 3 — Toàn bộ Kết quả Tối ưu trên nền `E5` (Đợt 1–3, `2026-11-15`)

Tất cả 9 thử nghiệm đều giữ nguyên 100% dataset split (`64/0/16`), augmentation, optimizer, scheduler, `seed=1234`, `300 epochs`, `bf16` và đều đã hoàn thành 300 epochs + đánh giá `test_vol` (`16` ca).

### 5.1 Bảng Tổng hợp Toàn bộ 9 Biến thể (Đợt 1, Đợt 2, Đợt 3) vs `E5`

| Đợt | Mã | Config (`training/config/models/`) | Thay đổi trên nền `E5` | Params | Train `mDice` / `loss` | Scar Dice *(med)* / HD95 | Edema Dice *(med)* / HD95 | Edema Incl | NormMyo | Ring Dice / HD95 | Avg Excl Dice *(med)* / HD95 | Avg Incl Dice *(med)* / HD95 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref** | **`E5`** | `lightweight_no_cross_attention` | Baseline đã chốt | `14,144,375` | `0.909 / 0.053` | `70.84%` *(74.39)* / **`8.54`** | `43.30%` *(45.13)* / `21.48` *(20.04)* | `72.46%` | `77.84%` / `4.99` | `88.05%` / `2.81` | `57.07%` *(58.75)* / **`15.01`** | `71.65%` *(75.15)* / **`11.61`** |
| **W1** | **`E5-A1`** | `lightweight_e5_a1_skip` | `skip_fusion: modality_weighted` (`TriModalSkipFusion`) | `14,278,736` | `0.9088 / 0.0536` | `70.38%` *(75.27)* / `10.95` | `42.58%` *(43.75)* / `21.28` *(18.64)* | `71.99%` | `77.10%` / `5.00` | `88.06%` / `2.77` | `56.48%` *(54.55)* / `16.11` | `71.19%` *(76.18)* / `12.84` |
| **W1** | **`E5-A2`** | `lightweight_e5_a2_boundary` | `decoder_refinement: boundary` (`BoundaryRefinementBlock` bản cũ) | `14,156,375` | `0.9077 / 0.0539` | `70.04%` *(74.19)* / `11.11` | `41.75%` *(46.41)* / `23.02` *(20.36)* | `71.75%` | **`78.39%`** / `5.02` | `88.21%` / `2.73` | `55.89%` *(54.40)* / `17.07` | `70.89%` *(75.99)* / `13.53` |
| **W1** | **`E5-L1`** | `lightweight_e5_l1_hier_loss` | `aar_weight=0.25, scar_weight=0.25` (`positive`) | `14,144,375` | `0.9001 / 0.1035` | `70.41%` *(74.76)* / `10.84` | `43.01%` *(49.87)* / `22.76` *(19.34)* | `72.10%` | `76.44%` / `5.08` | `87.31%` / `3.09` | `56.71%` *(57.80)* / `16.80` | `71.26%` *(75.00)* / `13.65` |
| **W2** | **`E5-B1`** | `lightweight_e5_b1_res_contrast` | `bottleneck_fusion: residual_contrast` (chưa `fork_rng`) | `14,311,415` | `0.9090 / 0.0532` | `70.18%` *(**75.72**)* / `8.99` *(med **5.33**)* | `42.29%` *(44.82)* / `22.27` *(21.06)* | `71.89%` | `77.44%` / `4.76` | `88.01%` / `2.78` | `56.23%` *(57.39)* / `15.63` | `71.03%` *(75.09)* / `12.40` |
| **W2** | **`E5-S2`** | `lightweight_e5_s2_spatial_skip` | `skip_fusion: spatial_residual` (zero-init + `(C+P+T)/3` anchor) | `14,241,757` | `0.9075 / 0.0544` | `70.36%` *(74.08)* / `9.72` *(med 5.61)* | `43.35%` *(**47.12**)* / `22.08` *(18.75)* | `72.51%` *(**76.35**)* | `77.37%` / `4.76` | `87.79%` / **`2.66`** | `56.86%` *(**59.59**)* / `15.90` | `71.43%` *(75.07)* / `12.80` |
| **W2** | **`E5-L2`** | `lightweight_e5_l2_anat_loss` | `anat_contrast_weight=0.15` (V1 chia `ring_area` — spike ep 1) | `14,144,375` | `0.7852 / 0.1727` | `65.80%` *(69.98)* / `12.06` | `40.94%` *(39.45)* / `22.89` *(20.02)* | `71.49%` | `74.40%` / `5.48` | `86.13%` / `3.25` | `53.37%` *(52.25)* / `17.47` | `68.64%` *(74.22)* / `14.09` |
| **W3** | **`E5-B1R`** *(BEST)* | **`lightweight_e5_b1r_res_contrast`** | **`ResidualContrastBottleneck` cô lập `fork_rng` khớp seed `E5`** | **`14,311,415`** | `0.9084 / 0.0537` | **`70.88%`** *(73.14)* / `11.04` *(med 7.54)* | `44.62%` *(46.86)* / **`20.55`** *(med **18.39**)* | `72.89%` *(76.21)* | `77.89%` / **`4.72`** | `88.05%` *(**89.58**)* / `2.80` | **`57.75%`** *(57.69)* / `15.80` | **`71.89%`** *(74.79)* / `12.73` |
| **W3** | **`E5-L3`** | `lightweight_e5_l3_bounded_loss` | Kiến trúc `E5` + Bounded `[0,1]` `anat_contrast_weight=0.10` | `14,144,375` | `0.9083 / 0.0620` | `68.08%` *(72.67)* / `10.65` *(med 6.39)* | `44.45%` *(43.04)* / `21.36` *(20.41)* | **`72.91%`** | `78.03%` / `4.96` | **`88.22%`** / `2.74` | `56.26%` *(54.88)* / `16.01` | `70.49%` *(75.97)* / `12.60` |
| **W3** | **`E5-Combo`** | `lightweight_e5_combo_arch_loss` | `B1R + S2 + Boundary(gamma+fork_rng) + BoundedLoss(0.10)` | `14,420,797` | `0.9092 / 0.0613` | `69.20%` *(72.54)* / `12.37` *(med 8.20)* | **`45.59%`** *(46.49)* / `22.76` *(21.78)* | `72.27%` | `77.23%` / `5.03` | `87.92%` / `2.92` | `57.40%` *(56.50)* / `17.56` | `70.73%` *(73.21)* / `13.86` |

#### Chi tiết Precision / Recall / ASD trên `test_vol` (các mốc chính & Đợt 3)
- `E5`: Scar `P 82.14% / R 65.73%`, Edema `P 44.14% / R 48.70%`, EdemaIncl `P 76.94% / R 70.55%`, Ring `P 89.69% / R 86.80%`
- `E5-A1`: Scar `P 81.03% / R 66.29%`, Edema `P 44.11% / R 48.32%`
- `E5-A2`: Scar `P 81.69% / R 66.61%`, Edema `P 43.67% / R 47.45%`
- `E5-L1`: Scar `P 80.20% / R 67.12%`, Edema `P 42.26% / R 50.96%`
- `E5-B1`: Scar `P 80.79% / R 66.85%` *(ASD `1.92`)*, Edema `P 44.68% / R 47.52%` *(ASD `5.63`)*, EdemaIncl `P 75.57% / R 70.59%`, Ring `P 89.48% / R 86.89%`
- `E5-S2`: Scar `P 81.08% / R 66.12%` *(ASD `2.10`)*, Edema `P 44.43% / R 50.06%` *(ASD `5.18`)*, EdemaIncl `P 75.86% / R 71.51%`, Ring `P 89.43% / R 86.56%`
- `E5-L2`: Scar `P 73.93% / R 65.32%` *(ASD `2.84`)*, Edema `P 39.83% / R 49.38%` *(ASD `6.10`)*, EdemaIncl `P 71.38% / R 74.38%`, Ring `P 85.86% / R 86.85%`
- **`E5-B1R` *(BEST)*:** Scar `P 81.29% / R 66.31%` *(ASD `2.33 mm`)*, Edema `P 45.98% / R 50.87%` *(ASD `4.71 mm`)*, EdemaIncl `P 76.95% / R 71.27%` *(ASD `2.59 mm`)*, NormMyo `P 79.17% / R 77.90%` *(ASD `0.90 mm`)*, Ring `P 89.57% / R 86.94%` *(ASD `0.49 mm`)*
- `E5-L3`: Scar `P 79.38% / R 64.00%` *(ASD `2.59 mm`)*, Edema `P 44.71% / R 50.94%` *(ASD `4.84 mm`)*, EdemaIncl `P 76.42% / R 71.15%`, Ring `P 89.36% / R 87.40%` *(ASD `0.46 mm`)*
- `E5-Combo`: Scar `P 81.10% / R 63.94%` *(ASD `2.36 mm`)*, Edema `P 45.58% / R 51.95%` *(ASD `4.89 mm`)*, EdemaIncl `P 76.27% / R 70.66%`, Ring `P 89.31% / R 86.90%`

---

### 5.2 Tại sao `E5-B1R` là Model Tốt Nhất Hiện tại & Chẩn đoán Gốc rễ W1–W3

1. **`E5-B1R` vượt `E5` toàn diện về Dice và phần lớn biên giải phẫu:**
   - **Thắng `E5` về Dice ở cả 5 vùng:** `AvgPathExcl` (`57.75%` vs `57.07%`, `+0.68 pp`), `AvgPathIncl` (`71.89%` vs `71.65%`, `+0.24 pp`), `Edema` (`44.62%` vs `43.30%`, `+1.32 pp`, kéo cả Precision `+1.84 pp` lẫn Recall `+2.17 pp`), `EdemaIncl` (`72.89%` vs `72.46%`, `+0.43 pp`), `Scar` (`70.88%` vs `70.84%`, `+0.04 pp`), `NormalMyo` (`77.89%` vs `77.84%`), `MyoRing` (`88.05%` = `88.05%`, median `89.58%` vs `89.01%`).
   - **Cải thiện HD95 ở 4/5 vùng:** `Edema HD95` giảm `-0.93 mm` (`20.55` vs `21.48 mm`, med `18.39` vs `20.04 mm`), `NormalMyo HD95` (`4.72` vs `4.99 mm`), `EdemaIncl HD95` (`14.42` vs `14.69 mm`), `MyoRing HD95` (`2.80` vs `2.81 mm`).
   - **Tác dụng của `torch.random.fork_rng`:** `E5-B1R` và `E5-B1` có kiến trúc y hệt (`14,311,415` params), chỉ khác `fork_rng` cô lập RNG khởi tạo nhánh `contrast_branch` trong `ResidualContrastBottleneck` (`training/models/modules/m2_plus.py`) để giữ nguyên quỹ đạo RNG gốc như `E5`. Nhờ giữ được quỹ đạo đó, **`AvgPathExcl` tăng `+1.52 pp`** (`57.75%` vs `56.23%`) và **`Edema` tăng `+2.33 pp`** (`44.62%` vs `42.29%`). Tham số `cross_attention.contrast_gamma` (512d) hội tụ sạch ở `absmean=0.03785` (`[-0.12253, +0.11514]`).
2. **Outlier `Case3003 z=2` (`GT Scar = 0`) — nguyên nhân duy nhất khiến `Scar HD95` của `E5-B1R` (`11.04 mm`) cao hơn `E5` (`8.54 mm`):**
   - Toàn bộ 9 biến thể W1–W3 đều vẫn dự đoán một cụm FP Scar trên lát `z=2` của `Case3003`, nhưng `E5-B1R` & `E5-Combo` đã kéo số voxel FP xuống thấp nhất (`60` và `65` voxels, so với `91–233` voxels ở các biến thể còn lại):
     - `E5-B1R`: `60` voxels (`70.9% / HD95 34.05 mm`); `E5-Combo`: `65` voxels (`66.1% / HD95 36.02 mm`)
     - `E5-S2`: `91` (`74.0% / 34.03`); `E5-B1`: `126` (`68.9% / 31.33`); `E5-L2`: `148` (`67.2% / 34.01`)
     - `E5-A1`: `150` (`65.1% / 36.27`); `E5-A2`: `194` (`68.3% / 36.78`); `E5-L1`: `202` (`65.5% / 36.67`); `E5-L3`: `233` (`54.6% / 36.82`)
   - Chỉ riêng cụm `60` voxel giả này trên `Case3003 z=2` đẩy mean `Scar HD95` của toàn tập test lên `~1.8 mm`.
3. **Tổng kết `E5-L3` & `E5-Combo` (Đợt 3) vs Đợt 1–2:**
   - **Bounded `[0, 1]` `anat_contrast_loss` (`E5-L3`):** Sửa triệt để lỗi spike ep 1 (`15.7384` ở `L2` → `0.742` ở `L3`, không collapse; train `mDice` lên `0.9083`), cải thiện `MyoRing` (`88.22%`, HD95 `2.74 mm`) và `Edema` (`44.45%`), kéo `Case3007 Edema` lên `31.1%`, nhưng trả giá bằng `Scar` giảm `-2.76 pp` (`68.08%`) và `Case3003 z=2` tăng FP (`233` voxels).
   - **`E5-Combo`:** Cũng thắng `E5` về `AvgPathExcl` (`57.40%`, thứ 2 toàn bảng), đạt **`Edema` cao nhất (`45.59%`, `Case3007 Edema = 40.9%`)** nhưng bị giảm `Scar` (`69.20%`) và `AvgIncl` (`70.73%`), `boundary_refine.gamma` mở khá lớn (`absmean=0.12526`, max `0.30572`).

---

## 6. Giai đoạn 3 (Đợt 4 — Wave 4, `2026-11-15`): Vượt Ngưỡng `58%+` `AvgPathExcl`

Cả 3 mô hình Wave 4 tiếp tục giữ nguyên 100% dataset split (`64/0/16`), augmentation, optimizer (`AdamW`), scheduler, `seed=1234`, `300 epochs`, `bf16` trên `NVIDIA A100-SXM4-40GB` (`amp_skipped_updates = 0` trên tất cả các run). Chi phí `FLOPs / MACs (G)` đo theo `training.benchmark.counted_flops` với input `(1, 1, 128, 128) × 3` (`1 MAC = 2 FLOPs`), `Peak VRAM` lấy max `train/gpu_peak_allocated_mb` & `train/gpu_peak_reserved_mb` ở `batch_size=16`.

### 6.1 Bảng So sánh Wave 4 vs `E5` & `E5-B1R` (`test_vol` `16` ca)

| Mã | Config (`training/config/models/`) | Thay đổi Kiến trúc | Params | FLOPs / MACs (G) | Peak VRAM (Alloc / Res) | Train `mDice` / Time | Scar Dice / HD95 | Edema Dice / HD95 | Edema Incl | NormMyo | Ring | **`AvgPathExcl`** | **`AvgPathIncl`** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5`** *(Ref)* | `lightweight_no_cross_attention` | Baseline đã chốt | `14,144,375` | `7.8629 / 3.9315` | — | `0.909` | `70.84% / 8.54` | `43.30% / 21.48` | `72.46%` | `77.84%` | `88.05%` | `57.07% / 15.01` | `71.65% / 11.61` |
| **`E5-B1R`** *(Best W3)* | `lightweight_e5_b1r_res_contrast` | `ResidualContrastBottleneck` (`fork_rng`) | `14,311,415` | `7.8840 / 3.9420` | `2,541.6 / 2,616.0 MB` | `0.9084 / 31.5m` | `70.88% / 11.04` | `44.62% / 20.55` | `72.89%` | `77.89%` | `88.05%` | `57.75% / 15.80` | `71.89% / 12.73` |
| **`E5-B2-MSContrast`** *(#2)* | `lightweight_e5_b2_ms_contrast` | `MultiScaleSymmetricContrastBottleneck` | `15,181,751` | **`7.9891 / 3.9946`** | **`2,561.4 / 2,634.0 MB`** | `0.9091 / 31.9m` | `70.76%` / **`9.85`** | `45.65%` / **`19.89`** | `73.69%` | `77.92%` | `88.06%` | `58.20%` / **`14.87`** | `72.23%` / **`11.66`** |
| **`E5-B1R-S3-NormSkip`** *(#1)* | **`lightweight_e5_b1r_s3_norm_skip`** | `B1R` + `NormalizedContrastSkipFusion` | `15,726,455` | `9.4695 / 4.7347` | `3,049.2 / 3,146.0 MB` | `0.9177 / 33.0m` | **`71.26%`** / `10.55` | **`45.69%`** / `20.57` | **`73.80%`** | **`79.04%`** | **`88.21%`** | **`58.47%`** / `15.56` | **`72.53%`** / `12.09` |
| **`E5-B1R-D2-WideFPN`** | `lightweight_e5_b1r_d2_wide_fpn` | `B1R` + `WideResidualFullConvDecoder` | `15,406,391` | `11.1577 / 5.5788` | `3,395.4 / 3,636.0 MB` | `0.9102 / 33.0m` | `70.15% / 12.28` | `45.11% / 19.92` | `73.08%` | `78.18%` | `87.92%` | `57.63% / 16.10` | `71.61% / 12.90` |

#### Chi tiết Precision / Recall / ASD & các ca khó trên `test_vol` (Wave 4)
- **`E5-B1R-S3-NormSkip` *(#1 BEST)*:**
  - `Scar`: `P 81.00% / R 67.41%` *(med `73.33%`)*, HD95 `10.55 mm` *(med `6.04`)*, ASD `2.32 mm`
  - `Edema`: `P 46.93% / R 51.31%` *(med `48.37%`)*, HD95 `20.57 mm` *(med `19.42`)*, ASD `4.47 mm`
  - `EdemaIncl`: `P 77.98% / R 71.87%` *(med `74.52%`)*, HD95 `13.63 mm`, ASD `2.46 mm`
  - `NormMyo`: `P 79.26% / R 79.61%`, HD95 `4.70 mm` *(med `3.64`)*, ASD `0.85 mm`
  - `MyoRing`: `P 89.58% / R 87.22%`, HD95 `2.90 mm`, ASD `0.49 mm`
  - Ca khó: `Case3003 scar = 66.4% / 36.68 mm` (`122` FP Scar voxels ở `z=2`), `Case2012 scar = 35.5% / 37.35 mm`, `Case2013 edema = 32.1% / 47.47 mm`, `Case3007 edema = 27.5% / 14.32 mm`.
- **`E5-B2-MSContrast` *(#2 & Best HD95)*:**
  - `Scar`: `P 79.96% / R 66.84%` *(med `74.19%`)*, HD95 **`9.85 mm`** *(med `6.72`)*, ASD **`2.00 mm`**
  - `Edema`: `P 45.97% / R 51.86%` *(med **`50.61%`**)*, HD95 **`19.89 mm`**, ASD `4.60 mm`
  - `EdemaIncl`: `P 76.65% / R 72.56%` *(med **`77.49%`**)*, HD95 **`13.48 mm`**, ASD **`2.35 mm`**
  - `NormMyo`: `P 79.70% / R 77.61%`, HD95 `4.74 mm`, ASD `0.89 mm`
  - `MyoRing`: `P 89.36% / R 87.08%`, HD95 `2.81 mm`, ASD `0.47 mm`
  - Ca khó: `Case3003 scar = 73.2% / 34.67 mm` (kéo FP Scar trên lát `z=2` xuống còn **`51` voxels** — thấp nhất trong tất cả 12 biến thể!), `Case2012 scar = 42.1% / 13.96 mm`, `Case2013 edema = 34.0% / 42.24 mm`, `Case3007 edema = 30.6% / 22.77 mm`.
- **`E5-B1R-D2-WideFPN`:**
  - `Scar`: `P 78.71% / R 66.88%` (ASD `2.48 mm`), `Edema`: `P 46.01% / R 50.46%` (ASD `4.55 mm`), `NormMyo`: `78.18%` (HD95 `4.64 mm`). Ca khó: `Case3003 z=2` `154` FP Scar voxels (`62.9% / 37.25 mm`).

### 6.2 Phân tích Cơ chế Wave 4 — Tại sao Hai Nhánh Tốt Nhất Bổ Trợ Nhau
1. **`NormalizedContrastSkipFusion` (`E5-B1R-S3-NormSkip`):** Chiếu CINE/PSIR/T2w skip qua `1x1 Conv + GroupNorm + ReLU` chung (`align_proj`) rồi tính 4 hiệu đối xứng có định hướng (`d_scar_cine`, `d_scar_t2`, `d_edema_pure`, `d_aar`) qua khối Conv `3x3` đầy đủ (`fork_rng`, `gamma` init=0). Giúp nâng Dice đồng loạt cả mô lành lẫn bệnh lý (`NormalMyo +1.15 pp`, `Scar +0.38 pp`, `Edema +1.07 pp` vs `E5-B1R`), đạt **`58.47% AvgPathExcl` / `72.53% AvgPathIncl`**. Tuy nhiên chưa có slice gate tại bottleneck nên `Case3003 z=2` vẫn còn `122` voxel Scar FP (`Scar HD95 = 10.55 mm`).
2. **`MultiScaleSymmetricContrastBottleneck` (`E5-B2-MSContrast`):** Mở rộng bottleneck bằng tương phản 4 chiều + nhánh đa tỉ lệ (`Conv3x3` + `Dilated3x3`) + **Global Peak-to-Mean Slice Gate (`m_avg`, `m_max - m_avg`)** với gần như **không tăng chi phí** (`7.9891 G` vs `7.8840 G` FLOPs; `2,561.4 MB` vs `2,541.6 MB` VRAM). Chính Slice Gate giúp giảm FP Scar ở `Case3003 z=2` xuống đáy (`51` voxels), kéo **`AvgPathExcl HD95` xuống `14.87 mm`** (thắng `E5` `15.01 mm` lẫn `E5-B1R` `15.80 mm`) và **`AvgPathExcl` lên `58.20%`**.
3. **`WideResidualFullConvDecoder` (`E5-B1R-D2-WideFPN`):** Tốn `11.16 G` FLOPs & `3.40 GB` Peak VRAM nhưng rớt `Scar` (`70.15%`, `12.28 mm`), khẳng định nút thắt nằm ở **tương phản đa phương thức tại bottleneck & skip**, không phải thiếu độ rộng Conv ở decoder.

---

## 7. Kiểm kê Tài nguyên, Backup Drive & Hướng dẫn Khôi phục Sau Khi Ngắt Colab

> Toàn bộ tài nguyên quan trọng đã được lưu lâu dài trên Google Drive (`/content/NCKH_V2` = `gdrive:NCKH_V2`) để đảm bảo **khi ngắt session Colab này, không mất bất kỳ code, trọng số hay log/metric nào**.

### 7.1 Vị trí lưu trữ trên Google Drive (`/content/NCKH_V2`)
- **#1 Best Model (`E5-B1R-S3-NormSkip` — CÓ trọng số `.pt`):**
  `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B1R_S3_NormSkip/`
  - `lightweighte5b1rs3normskip_seed1234.pt` (SHA256 verified), `lightweight_e5_b1r_s3_norm_skip.yaml`, `summary.json`, `logs/`, `tensorboard/`.
- **#2 Best HD95 / Efficiency (`E5-B2-MSContrast` — CÓ trọng số `.pt`):**
  `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B2_MSContrast/`
  - `lightweighte5b2mscontrast_seed1234.pt` (SHA256 verified), `lightweight_e5_b2_ms_contrast.yaml`, `summary.json`, `logs/`, `tensorboard/`.
- **Mốc nền trước đó (CÓ trọng số `.pt`):**
  - `E5-B1R` (`57.75%`): `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B1R_ResContrast/`
  - `E5` (`57.07%`): `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_Lightweight_No_CrossAttn/`
- **Phase 1 & 2 (`E1..E6`, `P2-1..P2-4`):** `Lightweight_Phase1_Ablation/`, `Lightweight_Phase2_Arch/`.
- **Logs & Metrics toàn bộ 12 thí nghiệm Phase 3 + Wave 4 (KHÔNG lưu `.pt` các biến thể không thắng như `A1/A2/L1/B1/S2/L2/L3/Combo/D2-WideFPN`):**
  `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/Lightweight_Phase3_E5_Optimization/`
  - `configs/`: đủ 13 YAML (`lightweight_no_cross_attention.yaml` + 12 YAML `lightweight_e5_*.yaml`).
  - `per_variant_logs/`: đủ **12** thư mục (`E5-A1`, `E5-A2`, `E5-L1`, `E5-B1`, `E5-S2`, `E5-L2`, `E5-B1R`, `E5-L3`, `E5-Combo`, `E5-B2-MSContrast`, `E5-B1R-S3-NormSkip`, `E5-B1R-D2-WideFPN`) chứa trọn `config.json`, `summary.json`, `metrics.csv`, `metrics.jsonl`, `train.log`, `splits/`, `test_results/` (kèm 16 mask dự đoán `Case*_pred.{nii.gz,npz}`).
  - Bảng tổng hợp 12 biến thể (đã cập nhật cả FLOPs/MACs/Peak VRAM): `phase3_summary_metrics.csv`, `phase3_per_case_long.csv`, `phase3_per_case_wide.csv`.
- **Gói Backup Code, Git History & Dataset Cache (phòng mất dữ liệu khi ngắt Colab):**
  `/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/SCAR_Code_Backup/`
  - `SCAR_main.bundle` *(đủ toàn bộ commit Git trên `main`, kể cả các commit sau `origin/main` `146ef02`)*
  - `SCAR_code_snapshot.tar.gz` *(code `training/`, `tests/`, `scripts/`, `pytest.ini`, `RESEARCH_HANDOFF.md`)*
  - `unpushed_from_origin.patch` *(patch đầy đủ từ `146ef02..HEAD`)*
  - `myopspp_unet2d_roi128_1mm_cache.tar.gz` *(bản cache 2D đã chuẩn bị sẵn của `data/myopspp_unet2d_roi128_1mm/cache` kèm `lists/` & `dataset_metadata.json`, giải nén chạy được ngay)*
- **Dataset gốc & Pretrained đã có sẵn trên Drive:**
  - Dataset NIfTI gốc: `/content/NCKH_V2/Dataset/MyoPSpp_UNet2D_ROI128_1mm.zip` (`92,880,113 bytes`)
  - Pretrained backbone: `/content/NCKH_V2/i_mmseg/ViT_checkpoint/R50-ViT-B_16.npz` (`440 MB`)
- **Tập `MyoPS380` (tách biệt, đơn vị voxel):** `/content/NCKH_V2/Checkpoints/MyoPS380/E5_Lightweight_No_CrossAttn/`.

### 7.2 Kiểm tra Code & Trạng thái Git
- Bộ unit test Wave 4 (`PYTHONPATH=. python3 -m pytest -q tests/test_lightweight_ablation.py tests/test_lightweight_phase2.py tests/test_shared_backbone_and_skip_gates.py`): **41 passed in 13.30s**. *(Lưu ý `tests/test_lightweight_m2_plus.py` cần package resolution khi chạy riêng có thể lỗi import `tests.test_sspanet`.)*
- Toàn bộ code (`m2_plus_net.py`, `decoder.py`, `fusion.py`, `m2_plus.py`, `losses.py`, `config_utils.py`, `train.py`, `trainer.py`), 12 YAML `lightweight_e5_*.yaml` và `RESEARCH_HANDOFF.md` đã được commit sạch trên nhánh `main` local và đóng gói vào `SCAR_main.bundle` trên Drive. *(Khi có xác thực GitHub, chạy `git push origin main` để đồng bộ lên remote).*

### 7.3 Hướng dẫn Khôi phục Trong Session Colab Mới (1 Lệnh — Không cần làm lại từ đầu)

```bash
# 1. Mount Google Drive tại /content/NCKH_V2 (qua colab drive.mount hoặc thư mục tương đương)
# 2. Khôi phục repo SCAR giữ nguyên toàn bộ commit Waves 1–4 từ Git bundle trên Drive:
git clone /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/SCAR_Code_Backup/SCAR_main.bundle /content/SCAR
cd /content/SCAR
git checkout main
git remote set-url origin https://github.com/thanhquan123hi1/SCAR.git

# 3. Khôi phục Cache Dataset MyoPS++ ROI 128 1mm, Dataset gốc & Pretrained R50-ViT-B_16:
mkdir -p data/ training/pretrained/
tar -xzf /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/SCAR_Code_Backup/myopspp_unet2d_roi128_1mm_cache.tar.gz -C data/
unzip -q /content/NCKH_V2/Dataset/MyoPSpp_UNet2D_ROI128_1mm.zip -d data/
ln -s /content/NCKH_V2/i_mmseg/ViT_checkpoint/R50-ViT-B_16.npz training/pretrained/R50-ViT-B_16.npz

# 4. Kiểm tra lại môi trường code:
PYTHONPATH=. python3 -m pytest -q tests/test_lightweight_ablation.py tests/test_lightweight_phase2.py tests/test_shared_backbone_and_skip_gates.py
```

### 7.4 Hướng đi Ưu tiên Ngay cho Phiên Làm Việc Sau (Wave 5)
1. **Thử nghiệm mũi nhọn `E5-B2-S3` (chưa từng chạy, tiềm năng cao nhất):**
   - Kết hợp **`bottleneck_fusion: multiscale_contrast`** (từ `E5-B2-MSContrast`: Slice Gate + pyramid 3x3 giúp hạ Scar FP `Case3003 z=2` xuống `51` voxels và `AvgPathExcl HD95 = 14.87 mm`, chỉ `+0.105G` FLOPs) với **`skip_fusion: norm_contrast`** (từ `E5-B1R-S3-NormSkip`: nâng toàn diện Dice lên `58.47%`).
   - Có thể chỉ cần tạo YAML `lightweight_e5_b2_s3.yaml` đặt cả `bottleneck_fusion: multiscale_contrast` lẫn `skip_fusion: norm_contrast` (không cần sửa code module!).
2. **Kiểm chứng đa seed (`seed 1234, 1024, 1098`) cho top ứng viên (`E5-B1R-S3-NormSkip`, `E5-B2-MSContrast`, và `E5-B2-S3` nếu thắng) vs `E5`:** Đo trung bình ± độ lệch chuẩn trên 3 seed để chốt kết quả cho bài báo.
3. **Tinh chỉnh loss nhẹ trên nền kiến trúc thắng (optional):** Sau khi test `B2-S3`, có thể khảo sát thêm bounded `anat_contrast_weight` nhỏ hơn (`0.03–0.05` thay cho `0.10`) để giữ `Scar` không tụt như `E5-L3`.

### 7.5 Lệnh Tái lập Best Model Hiện tại trên Session Mới

```bash
# Train #1 Best Model E5-B1R-S3-NormSkip (AvgPathExcl 58.47%)
python3 train.py \
  --config training/config/models/lightweight_e5_b1r_s3_norm_skip.yaml \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --pretrained training/pretrained/R50-ViT-B_16.npz \
  --seed 1234 --epochs 300 --batch-size 16 --img-size 128 --no-val --patience 0 --min-delta 0 --amp bf16

# Đánh giá trực tiếp từ trọng số #1 trên Drive
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B1R_S3_NormSkip/lightweighte5b1rs3normskip_seed1234.pt \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --amp bf16 --save-predictions

# Đánh giá trực tiếp từ trọng số #2 (E5-B2-MSContrast, Best HD95 14.87 mm) trên Drive
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B2_MSContrast/lightweighte5b2mscontrast_seed1234.pt \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --amp bf16 --save-predictions
```

---

## 8. Giai đoạn 4 (Wave 5 & Wave 6 — `2026-11-16`): Chẩn đoán Chuyên sâu MyoPS++ / MyoPS380 & Kỷ lục Mới Về `AvgPathExcl` / `HD95` / `Edema`

Mục tiêu của Wave 5 & Wave 6 là tìm kiến trúc vượt `E5-B1R-S3-NormSkip` trên **MyoPS++** trước (train ~33 phút/model, giữ nguyên 100% loss `0.2*CE + 0.8*BatchDice`, `seed=1234`, `300 epochs`, `bf16`, `batch_size=16`, `amp_skipped_updates=0` trên mọi run) mà không làm giảm Edema, làm bước đệm trước khi train dài (~vài giờ/model) trên **MyoPS380** (nơi `E5-B1R-S3-NormSkip` đạt `Scar 72.05% / 4.30 vox`, đang sát hoặc kém nhẹ `M2FTrans 72.39%` / `AWSnet 73.21%` / `MyoPS-Net 74.18%`).

### 8.1 Bảng Tổng hợp Toàn bộ 7 Thí nghiệm Wave 5–6 vs `E5-B1R-S3-NormSkip` (`16` ca `test_vol`, MyoPS++)

| Đợt | Mã | Config (`training/config/models/`) | Kiến trúc (trên nền `E5`) | Params | FLOPs / MACs (G) | Peak VRAM (Alloc / Res MB) | Train `mDice` / `loss` / Time | Scar Dice / HD95 | Scar P / R *(ASD)* | Edema Excl Dice / HD95 | Edema Incl Dice / HD95 | NormMyo Dice / HD95 | Ring Dice / HD95 | **`AvgPathExcl`** | **`AvgPathIncl`** | `Case3003 z=2` Scar FP |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W4-Ref** | **`Base #1: E5-B1R-S3`** | `lightweight_e5_b1r_s3_norm_skip` | `B1R + S3(NormSkip)` | `15,726,455` | `9.4695 / 4.7347` | `3049.2 / 3146.0` | `0.9177 / 0.0482 / 33.0m` | **`71.26%` / 10.55** | `81.00 / 67.41` *(2.32)* | `45.69% / 20.57` | `73.80% / 13.63` | **`79.04%` / 4.70** | `88.21% / 2.90` | `58.47% / 15.56` | **`72.53%` / 12.09** | `122` *(per W4)* |
| **W5-0** | **`E5-B2-S3`** | `lightweight_e5_b2_s3` | `B2(MSContrast) + S3(NormSkip)` | `16,596,791` | `9.5746 / 4.7873` | `3064.1 / 3160.0` | `0.9170 / 0.0486 / 26.7m` | `70.59% / 10.19` | `80.16 / 65.82` *(1.99)* | `45.98% / 19.70` | `73.26% / 13.95` | `78.48% / 4.66` | `88.30% / 2.78` | `58.28% / 14.95` | `71.93% / 12.07` | `71` |
| **W5-M1** | **`M1-B1RGate-DecSkip`** | `lightweight_e5_b1rgate_decoupled_skip` | `B1RGate + DecoupledAnatContrastSkip` (`groups=3`) | `15,636,738` | **`9.0932 / 4.5466`** | `3371.9 / 3520.0` | `0.9155 / 0.0496 / 33.8m` | `70.64% / 9.95` | `79.40 / 66.60` *(**1.88**)* | `45.81% / 20.50` | `73.61% / 13.85` | `77.92% / 4.80` | `88.20% / 2.75` | `58.23% / 15.23` | `72.12% / 11.90` | **`48`** |
| **W5-M2** | **`M2-B1R-S3-HRAnat`** | `lightweight_e5_b1r_s3_hr_anat` | `B1R + S3 + HRAnatCalib` (Z-score slice ở `128x128`) | `15,754,472` | `10.3770 / 5.1885` | `3521.1 / 3624.0` | `0.9163 / 0.0490 / 33.4m` | `70.75% / 9.67` | `80.56 / 66.74` *(2.20)* | `45.08% / 19.68` | `73.68% / 13.64` | `77.63% / 4.71` | **`88.31%` / 2.73** | `57.92% / 14.68` | `72.21% / 11.65` | `214` |
| **W5-M3** | **`M3-B2-Dec-HRAnat`** | `lightweight_e5_b2_decoupled_hr_anat` | `B2 + DecoupledSkip + HRAnatCalib` | `16,489,875` | `10.1016 / 5.0508` | `3863.9 / 3976.0` | `0.9149 / 0.0496 / 34.2m` | `71.21% / 10.50` | **`81.71`** `/ 67.26` *(2.16)* | `44.81% / 21.69` | `73.15% / 14.70` | `77.85% / 4.67` | `88.27% / 2.73` | `58.01% / 16.09` | `72.18% / 12.60` | `77` |
| **W6-A** | **`6A-B1RGate-S3`** | `lightweight_e5_b1rgate_s3` | `B1RGate + S3` (cô lập `B1RGate`) | `15,771,671` | `9.4737 / 4.7368` | `3054.6 / 3150.0` | `0.9168 / 0.0488 / 32.2m` | `69.36% / 10.46` | `79.38 / 64.91` *(2.13)* | `44.64% / 19.83` | `73.31% / 13.74` | `77.80% / 4.67` | `88.27% / 2.70` | `57.00% / 15.14` | `71.33% / 12.10` | `108` |
| **W6-B** | **`6B-B1R-S3AnatMean`** *(#1 Excl & HD95)* | `lightweight_e5_b1r_s3_anat_mean` | `B1R + AnatMeanNormalizedContrastSkipFusion` | `15,894,634` | `9.6275 / 4.8137` | `3164.8 / 3302.0` | `0.9163 / 0.0490 / 32.6m` | `70.84%` / **`9.76`** | `81.31 / 66.45` *(2.09)* | `46.14%` / **`19.57`** | `73.62%` / **`13.34`** | `77.98%` / **`4.51`** | `88.03% / 2.79` | **`58.49% / 14.66`** | `72.23% / 11.55` | `111` |
| **W6-C** | **`6C-B1R-S3MS-RawHDR`** *(#1 Edema)* | `lightweight_e5_b1r_s3_ms_hdr` | `B1R + S3MS + RawModalHDRRefinement` (`128x128`) | `15,753,479` | `10.2288 / 5.1144` | `3412.2 / 3560.0` | `0.9170 / 0.0484 / 32.7m` | `70.36% / 10.66` | `79.58 / 66.82` *(2.47)* | **`46.47%`** `/ 20.36` | **`74.26%`** `/ 14.28` | `78.40% / 4.54` | `88.26% / 2.71` | `58.41% / 15.51` | `72.31% / 12.47` | `163` |

### 8.2 Chẩn đoán Gốc rễ & Bằng chứng Dữ liệu Thật (MyoPS++ & MyoPS380)

1. **Đo phân bố cường độ pixel thực tế trên tập train cả 2 dataset (lý giải tại sao trừ thô `LGE - T2w` hay `LGE - CINE` dễ gãy):**
   - **MyoPS++ (`296` lát train, `4,849,664` pixels):**
     - `BG` (`86.51%`): `CINE=0.343`, `LGE=0.437`, `T2w=0.227` $\Rightarrow$ **`LGE - T2w = +0.210`** *(lớn hơn cả Scar `+0.096`, nên `GroupNorm` hay phép trừ không có `wall_gate` sẽ nhiễu mạnh ở vùng máu sáng/BG)*.
     - `NormMyo` (`7.55%`): `CINE=0.393`, `LGE=0.344`, `T2w=0.367` (`LGE - CINE = -0.049`, `LGE - T2w = -0.023`).
     - `Edema` (`3.27%`): `CINE=0.487`, `LGE=0.509`, `T2w=0.618` (`LGE - CINE = +0.022`, `LGE - T2w = -0.109`).
     - `Scar` (`2.66%`): `CINE=0.469`, `LGE=0.644`, `T2w=0.548` (`LGE - CINE = +0.174`, `LGE - T2w = +0.096`).
   - **MyoPS380 (`1,972` lát train, `legacy` labels `0=BG, 1=NormMyo, 2=Scar, 3=Edema`):**
     - `BG` (`93.14%`): `CINE=0.146`, `LGE=0.440`, `T2w=0.121` (`LGE - CINE = +0.294`, `LGE - T2w = +0.319`).
     - `NormMyo` (`3.09%`): `CINE=0.149`, `LGE=0.356`, `T2w=0.188`.
     - `Scar` (`2.14%`): `CINE=0.167`, `LGE=0.556`, `T2w=0.257` (`LGE - CINE = +0.389`, `LGE - T2w = +0.299`).
     - `Edema` (`1.62%`): `CINE=0.189`, `LGE=0.465`, `T2w=0.257` (`LGE - CINE = +0.276`, `LGE - T2w = +0.208`).

2. **Tại sao `HRAnatCalib` (`M2`/`M3`) thất bại:**
   - Chia `(x - mean_myo) / std_myo` trong mask cơ tim mềm `p_ring` theo từng lát cắt 2D đã khuếch đại nhiễu trên lát cắt gần lành (`std_myo` nhỏ $\Rightarrow$ noise thành `+3 sigma` giống sẹo giả), đẩy `Case3003 z=2` từ `122` lên **`214` voxel Scar FP** (`Case3003 Scar` rơi `66.4% -> 55.5%`), đồng thời kéo tụt `NormalMyo` `-1.41 pp` và `Case2030` `-12%`.

3. **Tại sao `DecoupledAnatContrastSkipFusion` (`M1`/`M3`, `groups=3`) tụt ở `Case2030` và `NormalMyo`:**
   - Đo trực tiếp trên **MyoPS++ test cache** của **`Case2030`**: `NormalMyo (CINE=0.532, LGE=0.397, T2w=0.524)` vs **`Scar (CINE=0.763, LGE=0.638, T2w=0.673)`**. Ở ca này (và `Case2013`), **`LGE < CINE` (`-0.125`) và `LGE < T2w` (`-0.035`) ngay cả trong lõi Scar**, nhưng **`LGE_Scar = 0.638 > LGE_Norm = 0.397`**.
   - `S3` (`NormalizedContrastSkipFusion`) giữ được `Case2030` và `NormalMyo 79.04%` nhờ chiếu chung 3 modality (`shared align_proj`) và xử lý gộp cả 4 hiệu tương phản trong cùng 1 khối `3x3`. Việc `DecoupledSkip` tách sang `groups=3` (lệch hệ toạ độ kênh) và tách đôi nhánh Scar/Edema khiến `Case2030` nhầm nhãn bên trong vòng cơ tim (dù `Ring Dice` vẫn `91.4%`).

4. **Tại sao `6A` loại `B1RGate`, còn `6B` và `6C` bổ trợ nhau rất rõ:**
   - **`6A (B1RGate-S3)`** giữ nguyên `S3` chỉ thay `B1R -> B1RGate`: rơi xuống **`57.00%`** `AvgExcl` $\Rightarrow$ chứng minh `B1R` tốt hơn `B1RGate`.
   - **`6B (B1R-S3AnatMean)`**: giữ nguyên `shared align_proj` và khối `3x3` gộp của `S3`, chỉ bổ sung soft CINE `wall_gate` + **`k - mu_wall` (KHÔNG chia `std`)**. Kết quả **vượt `Base #1` về `AvgPathExcl` (`58.49%` vs `58.47%`)**, giảm **`AvgExcl HD95` `-0.90 mm` (`14.66` vs `15.56 mm`)**, **`AvgIncl HD95` `-0.54 mm` (`11.55` vs `12.09 mm`)**, `Edema Excl` tăng lên **`46.14%` (`19.57 mm`)**, thắng `8/16` ca Scar (`Case2012 +11.4%` và HD95 `37.3 -> 20.2 mm`, `Case3007 Edema +17.7%`).
   - **`6C (B1R-S3MS-RawHDR)`**: thay khối `3x3` ở `S3` bằng `3x3 + dilated d=2` (`norm_contrast_ms`) + nhánh `RawModalHDRRefinement` ở `128x128` (bù cho việc `dec4` trong `SlimDecoderBlock` trước đây chỉ upsample + `1x1` mà chưa có `3x3` ở độ phân giải `1mm`, dùng thẳng `[CINE, PSIR, T2w]` không Z-score). Kết quả lập kỷ lục **`Edema Excl = 46.47%`** và **`Edema Incl = 74.26%`**, thắng **`9/16` ca `NormalMyo`**, đẩy `Case2012 Scar` lên **`47.5%`** (`+11.9%`).
   - **Giải mã yếu tố may rủi `Case2030` ở `Base #1`:** Đối chiếu `per_case.csv` trên **>20 model lịch sử** (W1–W4, P2, M2-Plus) cho thấy **19/20 model** chỉ đạt `Case2030` Scar `41–58.8%` / Edema `26–39.3%` / Norm `55–67%`, chỉ riêng 1 run `Base #1` (`seed=1234`) đạt `S=63.4% / E=48.5% / N=74.6%`. Do đó ở `seed=1234`, `Base #1` được lợi tới `+10–13%` từ riêng 1 ca đơn lẻ này; dù vậy **`6B` vẫn thắng `AvgPathExcl` (`58.49% > 58.47%`) và HD95 toàn diện**.

5. **Đối chiếu lỗi Scar của `E5-B1R-S3-NormSkip` trên MyoPS380 (`76` ca test, `75` ca có Scar GT, `0` ca zero-GT FP):**
   - Thua `M2FTrans` (`72.84%` eval seed2026 / `72.39±0.48%`) chủ yếu ở 2 nhóm:
     1) **Thiếu Recall trên Scar mờ / lõi tối (`Precision >> Recall`)**: `case0011` (`50.6%` vs `78.6%`, `R=35.7%`), `case0066` (`48.6%` vs `58.3%`, `R=35.4%`), `case0356` (`7.0%`, `R=3.6%`).
     2) **Đốm outlier làm tăng Scar HD95**: `case0326` (`hd95=31.26`), `case0306` (`hd95=22.52`), `case0361` (`hd95=12.24`).

### 8.3 Kiểm kê Code, Unit Tests & Backup Đợt Waves 5–6

- **Module mã nguồn mới đã thêm & tích hợp:**
  - `training/models/modules/m2_plus.py`: `GatedSymmetricContrastBottleneck` (`bottleneck_fusion: residual_contrast_gated`).
  - `training/models/modules/fusion.py`: `DecoupledAnatContrastSkipFusion` (`decoupled_anat_contrast`), **`AnatMeanNormalizedContrastSkipFusion` (`norm_contrast_anat_mean`, 6B)**, **`MultiScaleNormalizedContrastSkipFusion` (`norm_contrast_ms`, 6C)**.
  - `training/models/modules/decoder.py`: `HRAnatCalibRefinement` (`hr_anat_calib`), **`RawModalHDRRefinement` (`hr_raw_edge`, 6C)**.
  - `training/models/m2_plus_net.py` & `tests/test_lightweight_phase2.py`: nối đầy đủ + kiểm thử step-0 exact equivalence (`0.00e+00` vs `E5`), zero-input finite, `bf16` grads, FLOPs & checkpoint roundtrip.
  - **Kết quả `pytest` gần nhất:** `pytest -q tests/test_lightweight_ablation.py tests/test_lightweight_phase2.py tests/test_shared_backbone_and_skip_gates.py` $\Rightarrow$ **`47 passed in 17.94s`**.
- **Tài nguyên mới trên Google Drive (`/content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/`):**
  - Đủ 7 thư mục run Wave 5 & 6 (mỗi thư mục gồm YAML, `.pt`, `logs/`):
    - `E5_B2_S3_MSNormSkip/`, `E5_B1RGate_DecoupledSkip/`, `E5_B1R_S3_HRAnatCalib/`, `E5_B2_DecoupledHRAnat/` *(Wave 5)*
    - `E5_B1RGate_S3/`, **`E5_B1R_S3AnatMean/`** *(Wave 6B — #1 AvgExcl & HD95)*, **`E5_B1R_S3MS_RawHDR/`** *(Wave 6C — #1 Edema)*
  - Gói log/CSV tổng hợp toàn bộ Waves 5–6: `Lightweight_Wave5_6_Ablation/` (`wave5_6_summary_metrics.csv`, `wave5_6_per_case_long.csv`, `wave5_6_per_case_wide.csv`, `configs/`, `per_variant_logs/`).
  - Gói code & Git cập nhật mới nhất trong `SCAR_Code_Backup/` (`SCAR_main.bundle`, `SCAR_code_snapshot.tar.gz`, `unpushed_from_origin.patch`) + bản mới nhất của `RESEARCH_HANDOFF.md`.

### 8.4 Hướng đi Khuyến nghị Cho Phiên Tiếp Theo (Wave 7 $\rightarrow$ Chốt Qua MyoPS380)

Để có 1 model **thắng dứt điểm trên MyoPS++ ở cả `AvgPathExcl`, `AvgPathIncl`, `Scar` lẫn `Edema`** trước khi tốn thời gian train trên **MyoPS380**:

1. **Wave 7A — Ghép trực tiếp ưu điểm `6B` + `6C` (`E5-B1R-S3AnatMean-RawHDR`):**
   - Tạo YAML đặt `bottleneck_fusion: residual_contrast`, **`skip_fusion: norm_contrast_anat_mean`** *(từ `6B`: Best Scar/HD95/Precision/AvgExcl)*, **`decoder_refinement: hr_raw_edge`** *(từ `6C`: Best Edema Excl/Incl & NormalMyo `9/16` wins)* — **không cần viết thêm module mới**!
2. **Wave 7B — Tích hợp Multi-Scale ngay trong `S3AnatMean` + `RawHDR` (`S3AnatMeanMS-RawHDR`):**
   - Nâng cấp `AnatMeanNormalizedContrastSkipFusion` thay khối `Conv3x3` ở `contrast_fuse` thành `Conv3x3 + DWConv3x3(d=2)` (lấy thêm receptive field đa tỉ lệ của `6C`) rồi ghép `hr_raw_edge`.
3. **Wave 7C — Kiểm chứng đa seed (`seed=2026`, `seed=1024`) cho `E5-B1R-S3-NormSkip` vs `6B` (và `7A`):**
   - Triệt tiêu ảnh hưởng may rủi từ ca đơn `Case2030`, xác nhận trung bình $\pm$ std chắc chắn trước khi chạy trên **MyoPS380** (`train304_test76`).

```bash
# Lệnh đánh giá trực tiếp trên MyoPS++ từ checkpoint 6B (#1 AvgPathExcl 58.49%, Best HD95 14.66mm)
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B1R_S3AnatMean/lightweighte5b1rs3anatmean_seed1234.pt \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --amp bf16 --save-predictions

# Lệnh đánh giá trực tiếp từ checkpoint 6C (#1 Edema Excl 46.47%, Edema Incl 74.26%)
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/The_New_World/E5_B1R_S3MS_RawHDR/lightweighte5b1rs3msrawhdr_seed1234.pt \
  --data-root data/myopspp_unet2d_roi128_1mm/cache \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --amp bf16 --save-predictions
```


---

## 9. Chiến dịch Autoresearch Tối ưu Toàn diện (Wave 7 + Rounds 1–7 & Round 8 — `2026-11-19`)

Mục tiêu chiến dịch ngày `2026-11-19` là tối ưu có hệ thống cả **Loss** (Hierarchical Dice/Tversky, Karimi HD-DT, Far-FP dead-zone, per-region scaling, per-slice normalization) và **Kiến trúc** (`B2`, `B2SA`, `B2GSA`, `S3AnatMean`, `AnatRingContextRefinement`, `SSPA-GN`, `CoherentAnatContrastSkip`) trên cùng protocol cố định (`myopspp_unet2d_roi128_1mm`, `seed=1234`, `300 epochs`, `batch_size=16`, `bf16`, `--no-val`, `test_vol` 16 ca), đồng thời theo dõi 8 tiêu chí Pass định lượng.

### 9.1 Bộ 8 Tiêu chí Pass (`all_8_passed`)

| Nhóm | Scar | Edema Inclusive (`{2,3}`) | Myo Ring (`{1,2,3}`) | Avg Path Inclusive |
| :--- | :---: | :---: | :---: | :---: |
| **Dice (%)** *(càng cao càng tốt)* | `>= 68.91` | `>= 72.75` | `>= 87.97` | `>= 70.83` |
| **HD95 (mm)** *(càng thấp càng tốt)* | `<= 9.65` | `<= 12.21` | `<= 2.67` | `<= 10.93` |

### 9.2 Bảng Xếp hạng Toàn bộ 23 Trial đã hoàn thành (`Wave 7A/B` + `Rounds 1–7`)

*(Nguồn ledger gốc: `outputs/myopspp_unet2d_roi128_1mm/autoresearch_campaign/results_ledger.csv` & `.jsonl`; sắp xếp theo `Pass / 8` giảm dần → `AvgPathIncl Dice` giảm dần → `AvgPathIncl HD95` tăng dần.)*

| # | Trial ID | Loại | Pass | AvgIncl Dice / HD95 | Scar Dice / HD95 | EdemaIncl Dice / HD95 | MyoRing Dice / HD95 | AvgExcl Dice / HD95 | EdemaExcl Dice / HD95 | NormMyo Dice / HD95 | Params / FLOPs | FP3003 | Commit |
|---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`R4_L5_CalibTversky`** ★ | `loss_only` | **6/8** | `72.09` / **`10.72`** | `70.42` / **`7.65`** | `73.76` / `13.79` | `88.37` / `2.74` | `57.45` / `14.33` | `44.47` / `21.01` | `78.30` / `4.50` | `15.73M / 9.47G` | `22` | `17e1a0b` |
| **2** | **`R6_L8_MildScaleKarimi`** ★ | `loss_only` | **5/8** | `72.90` / `11.06` | **`72.29`** / `7.78` | `73.51` / `14.35` | `88.46` / **`2.70`** | `57.71` / `14.58` | `43.14` / `21.38` | `78.17` / `4.76` | `15.73M / 9.47G` | `26` | `dd336a3` |
| **3** | **`R4_A3_B2SA_S3AnatMean`** ★ | `arch_only` | **5/8** | `72.61` / `11.71` | `70.65` / `9.53` | **`74.57`** / `13.88` | `88.20` / `2.82` | `58.34` / `14.80` | `46.03` / `20.07` | `78.43` / `4.51` | `17.29M / 9.80G` | `185` | `17e1a0b` |
| 4 | `R1_L2_KarimiHDDT` | `loss_only` | 5/8 | `72.27` / `11.54` | `70.58` / `9.00` | `73.95` / `14.08` | `88.04` / `2.84` | `57.87` / `14.82` | `45.15` / `20.63` | `77.72` / `4.75` | `15.73M / 9.47G` | `241` | `38c386f` |
| 5 | `R3_C1_B2_S3AnatMean_L2` | `combo` | 5/8 | `71.78` / `11.56` | `69.89` / `9.63` | `73.66` / `13.50` | `88.07` / `2.82` | `57.25` / `14.62` | `44.60` / **`19.61`** | `77.94` / **`4.40`** | `16.76M / 9.73G` | `192` | `8154e74` |
| 6 | `R5_L7_ScaleKarimi` | `loss_only` | 5/8 | `71.39` / `12.47` | `69.89` / `9.28` | `72.89` / `15.66` | `88.30` / `2.83` | `56.37` / `15.35` | `42.86` / `21.42` | `78.09` / `4.75` | `15.73M / 9.47G` | `24` | `72090c0` |
| 7 | `R7_C6_B2GSA_S3AnatMean_RingCtx_L9` | `combo` | 4/8 | **`72.97`** / `11.89` | `71.74` / `10.43` | `74.19` / `13.35` | `88.41` / `2.77` | **`59.25`** / `15.45` | **`46.77`** / `20.47` | `78.49` / `4.74` | `17.33M / 10.95G` | `162` | `0183d69` |
| 8 | `R2_A1_B2_S3AnatMean` | `arch_only` | 4/8 | `72.37` / `11.02` | `71.83` / `7.86` | `72.91` / `14.17` | `87.96` / `2.89` | `57.89` / **`14.32`** | `43.94` / `20.78` | `77.70` / `4.96` | `16.76M / 9.73G` | **`0`** | `53df72b` |
| 9 | `R5_C3_B2SA_S3AnatMean_L5` | `combo` | 4/8 | `72.26` / `11.92` | `69.97` / `10.70` | `74.54` / **`13.13`** | `88.34` / `2.71` | `57.10` / `15.27` | `44.22` / `19.85` | **`78.73`** / `4.60` | `17.29M / 9.80G` | `214` | `72090c0` |
| 10 | `R3_C2_B2_S3AnatMean_RingCtx_L4` | `combo` | 4/8 | `72.22` / `12.09` | `70.53` / `10.56` | `73.91` / `13.62` | `88.14` / `2.85` | `57.64` / `15.60` | `44.74` / `20.65` | `77.86` / `4.53` | `16.80M / 10.88G` | `231` | `8154e74` |
| 11 | `R6_C4_B2SA_S3AnatMean_RingCtx_L5` | `combo` | 4/8 | `72.22` / `12.25` | `70.38` / `10.74` | `74.06` / `13.76` | **`88.49`** / `2.72` | `57.75` / `15.65` | `45.12` / `20.56` | `78.67` / `4.55` | `17.33M / 10.95G` | `42` | `dd336a3` |
| 12 | `R2_A2_S3_AnatRingCtx` | `arch_only` | 4/8 | `72.07` / `11.95` | `70.43` / `9.98` | `73.71` / `13.91` | `88.33` / `2.79` | `58.11` / `14.98` | `45.80` / `19.98` | `78.16` / `4.43` | `15.76M / 10.61G` | `199` | `53df72b` |
| 13 | `R3_L4_HybridFarFP` | `loss_only` | 4/8 | `71.72` / `11.97` | `69.04` / `9.88` | `74.39` / `14.06` | `88.27` / `2.73` | `56.48` / `15.86` | `43.92` / `21.84` | `77.78` / `4.66` | `15.73M / 9.47G` | `204` | `8154e74` |
| 14 | `R1_Anchor_E5_B1R_S3` | `anchor` | 4/8 | `71.50` / `12.98` | `69.91` / `10.70` | `73.09` / `15.26` | `88.23` / `2.77` | `57.00` / `16.07` | `44.08` / `21.44` | `78.03` / `4.59` | `15.73M / 9.47G` | `96` | `38c386f` |
| 15 | `R2_L3_StrongKarimi` | `loss_only` | 4/8 | `71.19` / `13.29` | `69.24` / `11.08` | `73.14` / `15.49` | `88.31` / `2.72` | `56.55` / `16.27` | `43.85` / `21.45` | `77.23` / `5.20` | `15.73M / 9.47G` | `180` | `53df72b` |
| 16 | `R6_A5_B2GatedSA_S3Coherent_RingCtx` | `arch_only` | 3/8 | `72.02` / `12.19` | `71.03` / `9.95` | `73.01` / `14.44` | `87.96` / `2.85` | `57.61` / `15.62` | `44.20` / `21.29` | `77.45` / `4.70` | `17.50M / 11.10G` | `47` | `dd336a3` |
| 17 | `Wave7A_S3AnatMean_RawHDR` | `pre_campaign` | 3/8 | `72.01` / `12.59` | `71.43` / `10.55` | `72.59` / `14.63` | `88.14` / `2.84` | `57.59` / `15.62` | `43.75` / `20.69` | `77.63` / `4.77` | `15.92M / 10.38G` | `58` | `ae6f101` |
| 18 | `Wave7B_S3AnatMeanMS_RawHDR` | `pre_campaign` | 3/8 | `71.33` / `12.27` | `69.95` / `10.09` | `72.71` / `14.44` | `87.97` / `2.74` | `57.26` / `15.23` | `44.56` / `20.37` | `77.25` / `4.72` | `15.92M / 10.39G` | `159` | `ae6f101` |
| 19 | `R1_L1_SymDTBCE` | `loss_only` | 3/8 | `71.15` / `12.15` | `69.32` / `10.74` | `72.98` / `13.55` | `87.92` / `2.90` | `57.01` / `16.23` | `44.70` / `21.72` | `77.56` / `4.66` | `15.73M / 9.47G` | `182` | `38c386f` |
| 20 | `R5_A4_E5B1RS3_SSPAGN` | `arch_only` | 3/8 | `71.05` / `12.14` | `69.17` / `9.67` | `72.92` / `14.62` | `87.80` / `2.91` | `56.82` / `16.07` | `44.47` / `22.47` | `77.20` / `4.76` | `15.73M / 9.47G` | `161` | `72090c0` |
| 21 | `R4_L6_CalibFarFP` | `loss_only` | 2/8 | `71.21` / `12.58` | `69.70` / `10.94` | `72.71` / `14.22` | `87.94` / `2.97` | `56.26` / `15.78` | `42.83` / `20.61` | `77.10` / `4.87` | `15.73M / 9.47G` | `93` | `17e1a0b` |
| 22 | `R7_C5_B2SA_S3AnatMean_L8` | `combo` | 1/8 | `70.75` / `12.33` | `68.86` / `10.80` | `72.64` / `13.85` | `87.98` / `2.86` | `55.91` / `15.70` | `42.97` / `20.59` | `77.91` / `5.03` | `17.29M / 9.80G` | `191` | `0183d69` |
| 23 | `R7_L9_SliceHierKarimi` | `loss_only` | 1/8 | `70.29` / `13.33` | `68.70` / `10.67` | `71.89` / `15.99` | `88.01` / `2.77` | `55.31` / `16.37` | `41.92` / `22.06` | `77.62` / `4.65` | `15.73M / 9.47G` | `109` | `0183d69` |

*(Ký hiệu ★: Top 3 model đã sao lưu đầy đủ trọng số `.pt` + config + logs lên Google Drive.)*

### 9.3 Tóm tắt Diễn biến & Bài học Rút ra (Từ Wave 7 đến Round 8)

1. **Wave 7A/7B (`ae6f101`):** Ghép `S3AnatMean` + `RawModalHDRRefinement` (`7A`: `AvgIncl 72.01 / 12.59`, `7B`: `71.33 / 12.27`) không vượt được `E5-B1R-S3` về `EdemaIncl` (`3/8` pass) $\Rightarrow$ chuyển sang chiến dịch tối ưu đồng thời Loss & Kiến trúc (Rounds 1–8).
2. **Round 1 (`38c386f`) — Thiết lập Anchor cục bộ & GPU Distance-Transform Loss:**
   - Tái lập anchor `R1_Anchor_E5_B1R_S3` trên môi trường mới (`torch 2.14.1+cu130` A100): `4/8` pass (`71.50 / 12.98`).
   - **`R1_L2_KarimiHDDT`** (HierBatchDice `0.20` + Karimi HD-DT MSE `0.25`) bật lên **`5/8` pass** (`72.27 / 11.54`, giảm `AvgIncl HD95` `-1.44 mm`, `Scar HD95` `10.70 -> 9.00 mm`), vượt rõ `R1_L1_SymDTBCE` (`3/8`).
3. **Round 2 (`53df72b`) — Kiểm chứng StrongKarimi & Kiến trúc mới (`B2+S3AnatMean`, `AnatRingCtx`):**
   - `R2_L3_StrongKarimi` (`weight=0.60`) quá mạnh làm tụt cả Dice lẫn HD95 (`4/8`, `71.19 / 13.29`).
   - **`R2_A1_B2_S3AnatMean`** (arch-only, base loss) đạt **`Scar = 71.83% / 7.86 mm`**, **`AvgExcl HD95 = 14.32 mm` (Best)**, và triệt tiêu hoàn toàn FP trên lát khó (`Case3003 z=2 FP = 0`).
   - `R2_A2_S3_AnatRingCtx` cải thiện `EdemaExcl` (`45.80%`) và `AvgExcl` (`58.11%`) với chi phí nhỏ.
4. **Round 3 (`8154e74`) — Dead-zone Far-FP & Combo đầu tiên:**
   - `R3_C1_B2_S3AnatMean_L2` đạt `5/8` (`71.78 / 11.56`, `EdemaExcl HD95 = 19.61 mm` Best, `NormMyo HD95 = 4.40 mm` Best).
   - `R3_L4_HybridFarFP` đẩy `EdemaIncl Dice = 74.39%` nhưng chưa cải thiện `Scar HD95` (`9.88 mm`).
5. **Round 4 (`17e1a0b`) — Bước nhảy vọt lớn nhất về Loss (`R4_L5`) & Kiến trúc (`R4_A3`):**
   - **`R4_L5_CalibTversky` (#1 hiện tại — `6/8` pass):** Kết hợp `CE(0.40)+Dice(0.60)` + hiệu chỉnh P/R (`MyoRing Tversky β=0.60`, `AAR Tversky β=0.42`) + `Karimi HD-DT (0.25)` trên đúng khung nhẹ `E5-B1R-S3`. Lần đầu vượt ngưỡng **`AvgPathIncl HD95 = 10.72 mm <= 10.93`**, **`Scar HD95 = 7.65 mm` (Best toàn chiến dịch)**, `FP3003 = 22`.
   - **`R4_A3_B2SA_S3AnatMean` (#3 hiện tại — `5/8` pass):** Thêm post-fusion 2D Self-Attention (`8x8` bottleneck, zero-init) + `S3AnatMean` dưới base loss $\Rightarrow$ **`EdemaIncl Dice = 74.57%` (Best)**, `AvgPathIncl = 72.61%`, `AvgPathExcl = 58.34%`.
   - `R4_L6_CalibFarFP` bị quá tay (`2/8`) $\Rightarrow$ loại hướng `FarFP` mạnh.
6. **Round 5 (`72090c0`) — Per-region Karimi scales, SSPA-GN & Combo `A3+L5`:**
   - `R5_A4_E5B1RS3_SSPAGN` (thay BN bằng GN trong SSPA) giảm xuống `3/8` $\Rightarrow$ giữ BN gốc.
   - `R5_L7_ScaleKarimi` (`max_norm=2.5` clamp) đạt `5/8` và `FP3003=24`, nhưng clamp làm yếu gradient xa nên `EdemaIncl HD95` lên `15.66 mm`.
   - `R5_C3_B2SA_S3AnatMean_L5` (`4/8`) đạt **`EdemaIncl HD95 = 13.13 mm` (Best)** và **`NormMyo Dice = 78.73%` (Best)**, nhưng tổ hợp `B2SA + L5` làm `Scar HD95` tăng lại `10.70 mm`.
7. **Round 6 (`dd336a3`) — Mild Per-Region Karimi (`R6_L8`) lập kỷ lục Scar & sát ngưỡng Ring HD95:**
   - **`R6_L8_MildScaleKarimi` (#2 hiện tại — `5/8` pass):** Giữ nguyên class/Tversky weights của `R4_L5`, bỏ `max_norm` clamp, dùng scales nhẹ `[8.0, 13.0, 15.0]`, `ring_beta=0.62`, `empty_val=1.75` $\Rightarrow$ **`Scar Dice = 72.29% / 7.78 mm`**, **`AvgPathIncl = 72.90% / 11.06 mm`**, **`MyoRing HD95 = 2.70 mm` (Best — chỉ còn cách ngưỡng `2.67` đúng `0.03 mm`)**, `FP3003 = 26`.
   - `R6_C4_B2SA_S3AnatMean_RingCtx_L5` (`4/8`) lập kỷ lục **`MyoRing Dice = 88.49%`**.
8. **Round 7 (`0183d69`) — Per-slice HierTversky & Per-slice Karimi (`L9`):**
   - `R7_L9_SliceHierKarimi` (`1/8`) và `R7_C5_B2SA_S3AnatMean_L8` (`1/8`) bị tụt mạnh (xác nhận `B2SA` xung đột với Karimi HD-DT mạnh và per-slice weight trên anchor bị nhiễu).
   - Tuy nhiên **`R7_C6_B2GSA_S3AnatMean_RingCtx_L9`** (`4/8`) lập **kỷ lục Dice cao nhất toàn lịch sử**: **`AvgPathIncl Dice = 72.97%`**, **`AvgPathExcl Dice = 59.25%`**, **`EdemaExcl Dice = 46.77%`** (HD95 chưa đạt ngưỡng).
9. **Round 8 (`32871ab`) — Đang chạy nền lúc bàn giao (`13:56 UTC`):**
   - Gồm 3 trial tập trung vào **`FullRangeKarimi L10`** (`max_steps=64` để không bị cắt cụt distance transform trên vùng Edema rộng) + các tổ hợp **không dùng Self-Attention** (`R8_L10_FullRangeKarimi`, `R8_C7_B2_S3AnatMean_L10`, `R8_C8_E5B1RS3_RingCtx_L10`).

### 9.4 Kiểm kê Code, Git & Vị trí Sao lưu trên Google Drive

- **Chuỗi Git commits mới (đã đóng gói trong `SCAR_main.bundle`):**
  - `38c386f` → `53df72b` → `8154e74` → `17e1a0b` → `72090c0` → `dd336a3` → `0183d69` → **`32871ab` (`HEAD -> main`)**.
- **Tài nguyên đã sao lưu đầy đủ lên Google Drive (`gdrive:NCKH_V2/Checkpoints/MyoPS++/The_New_World/` — mount cục bộ tại `/content/Checkpoints/MyoPS++/The_New_World/`):**
  1. **Toàn bộ Log, Metric & Configs (23 runs Wave 7 + R1–R7):**
     - `Autoresearch_R1_R7_Summary/results_ledger.csv` & `results_ledger.jsonl`
     - `Autoresearch_R1_R7_Summary/configs/` *(toàn bộ YAML `lightweight_e5_*.yaml`)*
     - `Autoresearch_R1_R7_Summary/per_run_logs/<run_id>/logs_*/` *(đầy đủ `train.log`, `metrics.csv`, `metrics.jsonl`, `summary.json`, `config.json`, `test_results/`)*
  2. **Trọng số (`.pt`) + Config + Logs của Top 3 model tốt nhất:**
     - **`R4_L5_CalibTversky/`** (`lightweighte5b1rs3l5calibtversky_seed1234.pt`, YAML, `logs_*`) — `#1: 6/8 pass, AvgIncl 72.09 / 10.72 mm, Scar HD95 7.65 mm`
     - **`R6_L8_MildScaleKarimi/`** (`lightweighte5b1rs3l8mildscalekarimi_seed1234.pt`, YAML, `logs_*`) — `#2: 5/8 pass, AvgIncl 72.90 / 11.06 mm, Scar 72.29%, Ring HD95 2.70 mm`
     - **`R4_A3_B2SA_S3AnatMean/`** (`lightweighte5b2sas3anatmean_seed1234.pt`, YAML, `logs_*`) — `#3: 5/8 pass, Arch-only best, EdemaIncl 74.57%, AvgExcl 58.34%`
  3. **Code & Tài liệu bàn giao mới nhất:**
     - `SCAR_Code_Backup/SCAR_main.bundle` *(chứa toàn bộ lịch sử git tới `32871ab`)*
     - `RESEARCH_HANDOFF.md` *(bản cập nhật này)*

### 9.5 Hướng dẫn Tiếp tục Phiên Sau (Không Cần Làm Lại Từ Đầu)

1. **Khôi phục repo & code từ Drive (nếu mở session Colab mới):**
   ```bash
   git clone /content/Checkpoints/MyoPS++/The_New_World/SCAR_Code_Backup/SCAR_main.bundle /content/SCAR
   cd /content/SCAR && git checkout main
   ```
2. **Đánh giá lại trực tiếp Top 3 checkpoint mà không cần train lại:**
   ```bash
   # #1 R4_L5_CalibTversky (6/8 pass, AvgIncl HD95 = 10.72 mm)
   python3 test.py \
     --checkpoint /content/Checkpoints/MyoPS++/The_New_World/R4_L5_CalibTversky/lightweighte5b1rs3l5calibtversky_seed1234.pt \
     --data-root data/myopspp_unet2d_roi128_1mm/cache \
     --dataset myopspp_unet2d_roi128_1mm --split test_vol --amp bf16 --save-predictions

   # #2 R6_L8_MildScaleKarimi (5/8 pass, Scar 72.29%, Ring HD95 = 2.70 mm)
   python3 test.py \
     --checkpoint /content/Checkpoints/MyoPS++/The_New_World/R6_L8_MildScaleKarimi/lightweighte5b1rs3l8mildscalekarimi_seed1234.pt \
     --data-root data/myopspp_unet2d_roi128_1mm/cache \
     --dataset myopspp_unet2d_roi128_1mm --split test_vol --amp bf16 --save-predictions
   ```
3. **Bước cải tiến ưu tiên tiếp theo để đạt `8/8` pass:**
   - **Kiểm tra kết quả Round 8 (`L10_FullRangeKarimi`, `max_steps=64`):** Nếu session bị ngắt trước khi Round 8 hoàn tất, chỉ cần chạy lại 3 config Round 8 đã có sẵn trong commit `32871ab` (`lightweight_e5_b1r_s3_l10_fullrange_karimi.yaml`, `lightweight_e5_b2_s3_anat_mean_l10.yaml`, `lightweight_e5_b1r_s3_ring_ctx_l10.yaml`).
   - **Tập trung vào 2 nút thắt cuối cùng chưa model nào pass:**
     1) **`MyoRing HD95 <= 2.67 mm`** *(rất gần: `R6_L8` đã đạt `2.70 mm` — chỉ lệch `0.03 mm`)* $\Rightarrow$ tinh chỉnh nhẹ `ring_beta` / `ring_hd_weight` quanh cấu hình `R4_L5` + `R6_L8`.
     2) **`EdemaIncl HD95 <= 12.21 mm`** *(hiện tốt nhất là `13.13 mm` ở `R5_C3`, còn cách `0.92 mm`)* $\Rightarrow$ kiểm tra per-case outlier Edema trên `R4_L5` & `R6_L8` (và kết quả `max_steps=64` của `L10`), kết hợp không dùng `B2SA` (vốn hay làm tăng `Scar HD95` khi đi với Karimi loss).
   - Sau khi chốt model vượt `7/8–8/8` trên MyoPS++, chạy **kiểm chứng đa seed (`2026`, `1024`)** và chuyển sang **MyoPS380**.
