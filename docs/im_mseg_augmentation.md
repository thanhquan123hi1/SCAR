# Augmentation chuẩn I-MMSeg

`training/dataset/myops_dataset.py::RandomGenerator` dùng augmentation của
`I-MMSeg-Fork/datasets/dataset_Myops.py`, được đối chiếu ngày 2026-11-06.
SHA256 file nguồn tham chiếu:

```text
2823a845d686b72133895885f5354e8f19bf841479eead862e8b508b8b2a6318
```

Áp dụng tự động cho training của các model/profile dùng chung `RandomGenerator`,
qua `train.py` và `run_all.py`; không cần flag mới.

## Thứ tự biến đổi

1. Chuỗi `if/elif/elif`, mỗi lần kiểm tra lấy một `random.random() > 0.5` mới:
   - Xoay `k * 90°`, `k = np.random.randint(0, 4)`, rồi flip theo trục ngẫu nhiên:
     xác suất 50%.
   - Nếu không vào nhánh trên: xoay góc nguyên `np.random.randint(-20, 20)` bằng
     `ndimage.rotate(order=0, reshape=False)` cho cả ảnh và mask: xác suất tổng 25%.
   - Nếu không vào hai nhánh trên: Gaussian noise với mean=0, std=0.2, **một noise
     map dùng chung cho ba modality**, clip `[0,1]`: xác suất tổng 12.5%.
   - Không thực hiện ba biến đổi này: xác suất 12.5%.
2. Dịch chuyển với xác suất 50%: hai số nguyên `random.randint(-5, 5)` dùng cho
   cả ảnh và mask, `np.roll(..., axis=(0, 1))` có vòng qua biên.
3. Brightness với xác suất 50%: một factor `random.uniform(0.8, 1.2)` dùng chung
   ba modality; `np.clip(image * factor, 0, 255)`.
4. Gamma với xác suất 50%: một gamma `random.uniform(0.8, 1.2)` dùng chung;
   `np.power(image / 255.0, gamma) * 255.0`.
5. Crop với xác suất 50%: lấy top/left từ 0 đến chiều ảnh tương ứng chia nguyên
   cho 5, cắt tối đa `output_size` cho cả ảnh và mask. Nếu crop nhỏ hơn đích,
   bước tiếp theo resize; không thêm padding.
6. Nếu kích thước khác `output_size`: `scipy.ndimage.zoom` với `order=3` cho ảnh,
   `order=0` cho mask; giữ các tùy chọn SciPy mặc định như nguồn tham chiếu.
7. Ảnh thành tensor float32 `(1,H,W)`; mask qua float32 rồi thành tensor long
   `(H,W)` như I-MMSeg.

Giữ nguyên cả chiều so sánh `> 0.5`, số lần/thứ tự gọi Python RNG và NumPy RNG.
Các phép hình học luôn đồng bộ ba modality và mask. Các bước shift, brightness,
gamma và crop có thể kết hợp, nhưng noise loại trừ xoay/flip và xoay góc nhỏ.

## Phạm vi tương thích

Ảnh cache vẫn được chuẩn hóa `[0,1]` trước augmentation. Công thức brightness/
gamma theo 255, noise dùng chung và dịch vòng biên được giữ nguyên theo chuẩn
I-MMSeg đã chọn. Output augmentation có thể ngoài `[0,1]`; không thêm clamp,
normalization hoặc phép biến đổi sau đó vì sẽ khác mã tham chiếu.

SCAR giữ thêm metadata của sample (`case_name`, spacing, v.v.). Việc đổi nhãn
legacy sang canonical tiếp tục nằm ở loader, trước augmentation. Cùng input,
`output_size`, Python/NumPy RNG state và phiên bản thư viện cho kết quả tensor
giống hệt I-MMSeg; seed toàn bộ run chưa đủ để làm hai trainer có chuỗi mẫu
giống nhau vì DataLoader, số worker và các thao tác ngẫu nhiên khác có thể khác.

`ResizeGenerator` cho validation và `predict_volume` cho inference giữ quy trình
riêng. Thay đổi này chỉ chuẩn hóa augmentation train, không đổi loss hoặc metric.
Bắt đầu run mới để dùng augmentation này xuyên suốt; protocol no-val hiện có
kiểm tra fingerprint mã nguồn khi resume nên checkpoint tạo bằng augmentation
cũ sẽ bị từ chối resume khi mã nguồn thay đổi.

## Kiểm chứng

`tests/fixtures/im_mseg_augmentation.py` lưu nguyên các hàm/class augmentation
từ nguồn tham chiếu để kiểm thử không phụ thuộc checkout I-MMSeg bên ngoài.
`tests/test_augmentation_fidelity.py` so sánh:

- Tensor bằng nhau chính xác, không dùng sai số cho phép.
- Toàn bộ Python/NumPy RNG state sau augmentation.
- 64 seed trên ba kích thước ảnh, hai kiểu dữ liệu mask.
- Bốn nhánh xoay/flip, xoay, noise, không biến đổi; mỗi nhánh đi với đủ 16 tổ hợp
  bật/tắt shift, brightness, gamma và crop.
- Metadata SCAR được giữ và input không bị sửa tại chỗ.

```powershell
python -m pytest -q tests/test_augmentation_fidelity.py
```
