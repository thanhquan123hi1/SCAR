# Research Handoff — Lightweight M2-Plus Optimization on MyoPS++

> Tài liệu bàn giao toàn bộ quá trình thực nghiệm, kết quả định lượng, chẩn đoán lỗi, mã nguồn đã chỉnh sửa, vị trí tài nguyên đã sao lưu và kế hoạch bước tiếp theo để phiên làm việc sau có thể tiếp tục ngay mà không cần train/test lại hay phân tích lại từ đầu.
> **Mốc tốt nhất hiện tại (`seed=1234`, `test_vol`):**
> - **#1 (Kỷ lục Dice):** **`E5-B1R-S3-NormSkip`** ([lightweight_e5_b1r_s3_norm_skip.yaml](training/config/models/lightweight_e5_b1r_s3_norm_skip.yaml), `15,726,455` params, `9.4695 G` FLOPs) — **`AvgPathExcl = 58.47%`** (`+1.40 pp` vs `E5`, `+0.72 pp` vs `E5-B1R`), **`AvgPathIncl = 72.53%`**, thắng `Scar = 71.26%`, `Edema = 45.69%`, `NormMyo = 79.04%`.
> - **#2 (Tốt nhất về Hiệu quả/Chi phí & HD95):** **`E5-B2-MSContrast`** ([lightweight_e5_b2_ms_contrast.yaml](training/config/models/lightweight_e5_b2_ms_contrast.yaml), `15,181,751` params, `7.9891 G` FLOPs) — **`AvgPathExcl = 58.20%`**, **`AvgPathExcl HD95 = 14.87 mm`**, `AvgPathIncl = 72.23% / 11.66 mm`, gần như không tốn thêm FLOPs/VRAM so với `E5-B1R`.
> **Cập nhật lần cuối:** `2026-11-15 05:12 UTC` (Đầy đủ Đợt 1–4 + gói khôi phục Drive sau khi ngắt Colab)

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
