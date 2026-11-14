# Research Handoff — Lightweight M2-Plus Optimization on MyoPS++

> Tài liệu tóm tắt toàn bộ quá trình thực nghiệm, kết quả định lượng, quyết định kiến trúc và kế hoạch bước tiếp theo để có thể tiếp tục nghiên cứu ngay trong phiên làm việc mới mà không cần train/test lại các mốc đã hoàn thành.

---

## 1. Bối cảnh & Protocol Thực nghiệm Chung

- **Dataset profile:** `myopspp_unet2d_roi128_1mm`
  - **Split cố định (`--no-val`):** `64` bệnh nhân train (`296` lát cắt 2D), `0` val, `16` bệnh nhân `test_vol` (lưu tại `data/myopspp_unet2d_roi128_1mm/cache`).
  - **Không gian & nhãn:** ảnh `128x128`, spacing XY `1.0 mm`, nhãn canonical `0=background`, `1=normal_myocardium`, `2=edema`, `3=scar` (`edema_inclusive = {2,3}`, `myocardial_ring = {1,2,3}`). Đánh giá trên predicted ROI grid, single-pass `argmax_logits` (`single_pass_argmax_v1`).
- **Cấu hình training chuẩn (giữ cố định giữa các so sánh công bằng):**
  - `seed: 1234`, `epochs: 300`, `batch_size: 16`, `accum_steps: 1`
  - Optimizer: `AdamW` (`lr: 0.0003333`, `weight_decay: 0.0001`), scheduler polynomial (`power: 0.9`)
  - Loss mặc định: `0.2 * CE + 0.8 * BatchDice(weights=[0.5, 1.0, 1.0, 1.0], normalization='classes', smooth=1e-5)` (`aar_weight=0.0`, `scar_weight=0.0`)
  - Precision: `bf16` (GPU A100), `--pretrained training/pretrained/R50-ViT-B_16.npz`
  - Data augmentation: `RandomGenerator` mặc định (protocol I-MMSeg trong `training/dataset/myops_dataset.py`)

---

## 2. Kết quả So sánh 3 Model Nền (Seed `1234`, `test_vol` `16` ca)

| Model | Params | Scar Dice / HD95 | Edema Excl Dice / HD95 | Edema Incl Dice / HD95 | Normal Myo | Myo Ring | Avg Path Excl Dice / HD95 | Avg Path Incl Dice / HD95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M2-Plus`** (`m2_plus.yaml`) | `63,748,336` | `67.34% / 9.83` | `39.10% / 25.21` | `70.66% / 17.14` | `76.12%` | `87.61%` | `53.22% / 17.52` | `69.00% / 13.49` |
| **`M2-Plus-Paper-Formula`** | `48,016,342` | `66.27% / 8.90` | `43.03% / 20.69` | `70.90% / 16.16` | `75.71%` | `87.39%` | `54.65% / 14.79` | `68.58% / 12.53` |
| **`Lightweight-M2-Plus`** | **`14,670,711`** | **`67.45% / 12.66`** | `42.56% / 20.86` | **`71.61% / 14.74`** | **`76.41%`** | `87.42%` | **`55.01% / 16.76`** | **`69.53% / 13.70`** |

$\Rightarrow$ **`Lightweight-M2-Plus`** nhẹ nhất (`14.67M`, giảm ~77% tham số so với `M2-Plus`) nhưng đạt `Avg Excl` và `Avg Incl` cao nhất, nên được chọn làm model chính.

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

**Kết luận GĐ1:** `Skip Gates`, `FiLM`, và độ sâu `ResNet [3,4,9]` đều thực sự có ích (`E2/E4/E6` đều giảm mạnh khi cắt). Trái lại, **`Cross-Attention MHA` (`E5`) gây hại trên ảnh `128x128` (`8x8` bottleneck)** — cắt `MHA` giúp `Scar +3.39 pp`, `Avg Incl +2.12 pp`, `Avg Excl +2.06 pp`, giảm `Scar HD95` `-4.12 mm`.

---

## 4. Giai đoạn 2A + 2B & Đối chiếu Chốt Nền `E5` vs `P2-1..P2-4`

Tất cả các biến thể `P2-1..P2-4` đều thử đặt trên nền **bỏ `OptimizedSSPANet` + bỏ `CrossAttn MHA`** (`sspanet_gate: identity`):

| Mã | Config (`training/config/models/`) | Kiến trúc | Params | MACs (G) | Scar Dice / HD95 | Edema Excl Dice / HD95 | Edema Incl | NormMyo | Ring | Avg Excl Dice / HD95 | Avg Incl Dice / HD95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E5`** *(Giữ)* | `lightweight_no_cross_attention` | Giữ `SSPANet`, bỏ `CrossAttn` | `14,144,375` | `3.9315` | `70.84 / 8.54` | `43.30 / 21.48` | `72.46` | **`77.84`** | `88.05` | `57.07 / 15.01` | **`71.65 / 11.61`** |
| **`P2-1`** | `lightweight_p2_clean_pruned` | `E5` bỏ toàn bộ `OptimizedSSPANet` | `12,759,671` | `3.9238` | `68.51 / 9.70` | `44.77 / 21.39` | **`72.67`** | `77.38` | **`88.42`** | `56.64 / 15.55` | `70.59 / 12.25` |
| **`P2-2`** | `lightweight_p2_contrast_fusion` | `P2-1 + LocalModalityContrastFusion` | `13,471,767` | `3.9346` | **`70.94` / 9.85** | `42.29 / 22.24` | `71.70` | `76.61` | `87.77` | `56.61 / 16.05` | `71.32 / 12.81` |
| **`P2-3`** | `lightweight_p2_adaptive_skip` | `P2-1 + TriModalSkipFusion` | `12,894,032` | `3.9288` | `69.80 / 10.87` | `43.06 / 23.76` | `71.80` | `77.58` | `88.00` | `56.43 / 17.32` | `70.80 / 13.32` |
| **`P2-4`** | `lightweight_p2_v2` | `P2-1 + Contrast + Skip + Boundary` | `13,618,096` | `4.1336` | `70.28 / 11.29` | **`44.97` / 24.19** | `72.56` | `76.86` | `87.91` | **`57.62` / 17.74** | `71.42 / 14.01` |

### Kết luận Chốt Kiến trúc Nền:
1. **`E5` trừ `OptimizedSSPANet` chính xác là `P2-1`** (trong `m2_plus.py`, `use_cross_attention=False` tự động bypass `use_pos_enc` nên hai cấu hình chỉ khác đúng `sspanet_gate: optimized` vs `identity`).
2. **Vai trò thực sự của `OptimizedSSPANet` trên `E5`:** Khi đã bỏ `Cross-Attention`, giữ `OptimizedSSPANet` (`E5`) giúp **`Scar Dice` cao hơn `+2.33 pp`** (`70.84%` vs `68.51%`), **`Scar HD95` tốt hơn `1.16 mm`** (`8.54` vs `9.70 mm`), và **`Avg Inclusive Dice` cao hơn `+1.06 pp`** (`71.65%` vs `70.59%`), đổi lại tốn `1,384,704` params và `Edema Excl` thấp hơn `-1.48 pp`.
3. Toàn bộ 4 biến thể bỏ `SSPANet` (`P2-1..P2-4`) đều có `Scar HD95` (`9.70–11.29 mm`) và `Avg Incl HD95` (`12.25–14.01 mm`) kém hơn `E5` (`8.54 / 11.61 mm`).
4. **Quyết định đã chốt:** **Giữ `E5` (`Lightweight-No-CrossAttn`, `14.14M`, giữ `OptimizedSSPANet`) làm kiến trúc gốc** cho các cải tiến tiếp theo.

---

## 5. Phân tích Chẩn đoán Trên `E5` & Định hướng Bước Tiếp Theo

### Điểm yếu còn lại của `E5` (từ `per_case.csv` trên 16 ca test)
- `Scar`: `Precision = 82.14%` nhưng `Recall = 65.73%` (`1,915.7` vs `2,382.8` voxels GT) $\rightarrow$ **thiếu recall ở lõi Scar**, một phần Scar bị gán nhầm sang `Edema` dù tổng vùng `Edema Inclusive` rất sát (`4,232.5` vs `4,365.2` voxels, `Dice 72.46%`). Ví dụ: `Case2012` Scar Rec `26.64%` / Prec `92.27%`.
- `Edema Excl`: `Dice = 43.30%`, `Precision = 44.14%`, `Recall = 48.70%`, `HD95 = 21.48 mm` (`2,316.8` vs `1,982.4` voxels) $\rightarrow$ bị **nhầm `Normal Myo ↔ Edema` / dương tính giả rải rác** dù `Myo Ring` rất tốt (`88.05%`, `HD95 2.81 mm`). Ví dụ: `Case2013` Edema `pred 7,313` vs `GT 2,016` (`HD95 = 50.34 mm`), `Case3044` Edema `Prec = 9.07%`.

### Kế hoạch đề xuất cho phiên tiếp theo (Ràng buộc đã chốt: GIỮ NGUYÊN 100% training config & data augmentation gốc; CHỈ cải tiến Kiến trúc + Loss trên `E5`)

- **Đợt 1 (3 biến thể đơn-biến trên nền `E5`, song song ~30 phút trên A100):**
  1. **`E5-A1` (`E5 + TriModalSkipFusion`)**: `14,278,736` params (`+134K` vs `E5`, vẫn nhẹ hơn Baseline `392K`). Thay skip `(cine+psir+t2w)/3` bằng softmax cạnh tranh `TriModalSkipFusion` (đã có sẵn trong `training/models/modules/fusion.py`) nhưng **vẫn giữ `OptimizedSSPANet`**.
  2. **`E5-A2` (`E5 + TriModalSkipFusion + BoundaryRefinement`)**: `14,290,704` params (`+146K` vs `E5`). Ghép thêm `BoundaryRefinementBlock` (đã có trong `training/models/modules/decoder.py`) trước `head` trên nền có `SSPANet`. *(Lưu ý: `E5 + LocalModalityContrastFusion(latent=192)` nặng `14,856,471` > Baseline `14.67M` và ở `P2-2` module này làm giảm Edema `-2.49 pp`, nên không ưu tiên.)*
  3. **`E5-L1` (`E5 + Hierarchical Loss`)**: Giữ nguyên kiến trúc `E5` (`14,144,375` params), bật `--aar-weight 0.25 --scar-weight 0.25` (đã có sẵn trong `training/loss/losses.py`) để ép `Scar ⊂ AAR`, kéo `Scar Recall` lên và tách `Scar ↔ Edema`.
- **Đợt 2 (~30 phút):**
  1. **`E5-L2` (`E5 + Focal / Soft Boundary Loss`)**: Thêm tùy chọn opt-in trong `training/loss/losses.py` (`Focal CE` + GPU `Soft Morphological Boundary Edge Loss`) kết hợp `AAR + Scar` nhằm trừng phạt mạnh các cụm dương tính giả xa biên như `Case2013` (`HD95 = 50.34 mm`).
  2. **`E5-Best`**: Tổ hợp **Kiến trúc thắng** + **Hàm Loss thắng** từ 2 đợt trên nền `E5`.

*(Ghi chú mở rộng tùy chọn ngoài ràng buộc hiện tại: chưa chạy ablation cắt lẻ 3 nhánh bên trong `OptimizedSSPANet` gồm `SSPA_ZPoolChannelAttention` `393K`, `SSPA_DirectionalStripPooling` `399K`, `SSPA_StylePooling` `593K` trên nền `E5`.)*

---

## 6. Kiểm kê Tài nguyên trên Google Drive & Lệnh Tái lập

### Cấu trúc đã lưu tại Google Drive (`/content/NCKH_V2/Checkpoints/MyoPS++/`)
- **`E5_Lightweight_No_CrossAttn/`** *(model gốc hiện tại)*: `lightweightnocrossattn_seed1234.pt` (`163 MB`), `lightweight_no_cross_attention.yaml`, `summary.json`, và `logs/` đầy đủ (`metrics.csv`, `train.log`, `config.json`, `splits/`, `test_results/` kèm 16 mask `.nii.gz/.npz`).
- **`Lightweight_Phase1_Ablation/`**: 6 CSV tổng hợp (`phase1_summary_metrics.csv`, `phase1_benchmark_summary.csv`, `phase1_per_case_long.csv`, `phase1_per_case_wide.csv`, `phase1_training_curves_full.csv`, `phase1_win_loss_and_outliers.csv`), `configs/` (7 YAML GĐ1), `per_variant_logs/{Baseline,E1..E6}/`.
- **`Lightweight_Phase2_Arch/`**: 6 CSV tổng hợp (`phase2_arch_*.csv`), `configs/` (8 YAML GĐ1 tham chiếu + GĐ2), `per_variant_logs/`, và `phase2_decision_and_winner.json` (đã chốt `E5`).
- **`Lightweight_M2_Plus/`**, **`M2_Plus/`**, **`M2_Plus_Paper_Formula/`**, **`I_MMSeg/`**, **`TransUnet/`**, và `comparison_3models_per_case_seed1234.csv`.

### Lệnh chạy chuẩn cho `E5` (trên `myopspp_unet2d_roi128_1mm`)
*(Lưu ý: cần có `data/myopspp_unet2d_roi128_1mm/cache` và `training/pretrained/R50-ViT-B_16.npz` — không track trong Git theo `.gitignore`.)*

```bash
# Train E5 từ đầu
python3 train.py \
  --config training/config/models/lightweight_no_cross_attention.yaml \
  --dataset myopspp_unet2d_roi128_1mm \
  --pretrained training/pretrained/R50-ViT-B_16.npz \
  --seed 1234 --epochs 300 --batch-size 16 --amp bf16

# Đánh giá trên test_vol (16 bệnh nhân)
python3 test.py \
  --checkpoint /content/NCKH_V2/Checkpoints/MyoPS++/E5_Lightweight_No_CrossAttn/lightweightnocrossattn_seed1234.pt \
  --dataset myopspp_unet2d_roi128_1mm \
  --split test_vol --batch-size 8 --save-predictions
```
