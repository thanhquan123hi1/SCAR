# MyoPS++ predicted U-Net2D ROI128/1mm

Profile: `myopspp_unet2d_roi128_1mm`, 64 train / 16 test, không validation.
Nguồn mặc định: `D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm`.

```powershell
python -m preprocessing.myopspp_roi
```

Lệnh đóng gói export đã có thành `data/myopspp_unet2d_roi128_1mm/cache`, không ghi đè
nguồn hoặc cache đã tồn tại. Ảnh và hình học được giữ nguyên; mapping raw labels:
`0/500/600 -> 0`, `200 -> 1`, `1220 -> 2` (edema), `2221 -> 3` (scar).

Xem [hướng dẫn train/resume/evaluate](../docs/myopspp_roi_training.md) và
[manifest bệnh nhân](splits/myopspp_unet2d_roi128_1mm/patients.json).
`myopspp_unet2d.py` và `myopspp_unet2d_roi192.py` là công cụ xuất dữ liệu/localizer lịch sử;
không cần chạy chúng để train trên export hiện có.
