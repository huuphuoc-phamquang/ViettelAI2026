# Pipeline test 3DGS trên Kaggle (T4) — Viettel AI 2026 / Novel View Synthesis

Mục tiêu giai đoạn này: **kiểm tra pipeline chạy đúng đầu-cuối** trên 1 scene —
train 3DGS ít iteration (7000), render tại các pose trong `test_poses.csv`,
lưu ảnh, và (với public set) tính PSNR để chắc chắn ảnh ra hợp lý.

## Data (đã khảo sát)
```
<scene>/
  train/
    images/           200 ảnh (đã resize 1/4)
    sparse/0/         COLMAP: cameras.bin (PINHOLE, 1 cam), images.bin, points3D.bin/.ply
  test/
    images/           50 ảnh GT   <-- CHỈ CÓ Ở public_set
    test_poses.csv    pose để render: qw..qz (w2c), tx..tz, fx,fy,cx,cy, width,height
```
- Camera model = **PINHOLE**, principal point ở tâm ảnh → dùng projection FoV chuẩn của 3DGS, không cần undistort.
- `test_poses.csv` dùng convention COLMAP giống `images.bin` (world→camera). Script render đã xử lý đúng: `R = qvec2rotmat(q).T`, `T = t`.
- **private_set1** không có `test/images` → chỉ render để nộp, không tính PSNR local được.

## Chuẩn bị trên Kaggle
1. Notebook: **Accelerator = GPU T4**, **Internet = On**.
2. Upload data (thư mục `public_set` / `private_set1`) làm **Kaggle Dataset**, ví dụ mount tại `/kaggle/input/vt2026/...`.
3. Upload luôn thư mục `pipeline/` này (kèm trong dataset, hoặc git repo của bạn). Giả sử nó ở `/kaggle/input/vt2026/pipeline/` — copy ra working:
   ```python
   !mkdir -p /kaggle/working/pipeline && cp /kaggle/input/vt2026/pipeline/*.py /kaggle/working/pipeline/
   ```

## Các cell chạy tuần tự

### 1. Cài đặt repo 3DGS + build CUDA rasterizer cho T4
```python
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"   # T4 = sm_75, phải set TRƯỚC khi build
%cd /kaggle/working
!git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
%cd /kaggle/working/gaussian-splatting
!pip -q install plyfile
!pip -q install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
```
(build ~5–8 phút. Nếu `fused-ssim` lỗi có thể bỏ — train.py fallback về SSIM thường.)

### 2. Chọn scene + kiểm tra data
```python
SCENE = "/kaggle/input/vt2026/public_set/hcm0031"   # đổi sang scene khác / private tùy ý
!python /kaggle/working/pipeline/inspect_data.py --scene $SCENE
```

### 3. Train 7000 iteration
```python
!python train.py -s $SCENE/train -m /kaggle/working/output/hcm0031 \
    --iterations 7000 --save_iterations 7000 --test_iterations 7000 -r 1
```
- KHÔNG dùng `--eval` (ta muốn dùng cả 200 ảnh để train; test lấy từ CSV riêng).
- `-r 1`: giữ nguyên độ phân giải ảnh (đã downscale sẵn).
- Checkpoint lưu ở `output/hcm0031/point_cloud/iteration_7000/point_cloud.ply`.

### 4. Render tại các test pose và lưu ảnh
```python
%cd /kaggle/working/gaussian-splatting
!python /kaggle/working/pipeline/render_test_poses.py \
    --model /kaggle/working/output/hcm0031 \
    --poses $SCENE/test/test_poses.csv \
    --out /kaggle/working/output/hcm0031/test_renders \
    --iteration 7000
```
> Phải chạy từ trong `gaussian-splatting/` để import được module của repo.
> Ảnh lưu đúng tên `image_name` trong CSV (đây thường là định dạng nộp).

### 5. (public) Tính PSNR để sanity-check
```python
!python /kaggle/working/pipeline/eval_metrics.py \
    --renders /kaggle/working/output/hcm0031/test_renders \
    --gt $SCENE/test/images
```
7000 iter là để test pipeline, đừng kỳ vọng điểm cao — chỉ cần PSNR "hợp lý"
(thường ~20+ dB với view gần) và ảnh không đen/vỡ là pipeline OK.

### 6. Xem thử vài ảnh
```python
import matplotlib.pyplot as plt
from PIL import Image
import os
d = "/kaggle/working/output/hcm0031/test_renders"
fs = sorted(os.listdir(d))[:3]
plt.figure(figsize=(15,5))
for i,f in enumerate(fs):
    plt.subplot(1,3,i+1); plt.imshow(Image.open(os.path.join(d,f))); plt.axis("off"); plt.title(f[:18])
plt.show()
```

## Checklist "pipeline OK"
- [ ] Build submodule không lỗi (import `diff_gaussian_rasterization` được).
- [ ] `inspect_data.py`: 200 train / 50 pose, camera PINHOLE, tên ảnh test khớp CSV.
- [ ] Train chạy hết 7000 iter, có file `point_cloud/iteration_7000/point_cloud.ply`.
- [ ] Render ra đủ 50 ảnh, đúng tên, nội dung nhìn ra được cảnh (không đen/nhiễu hoàn toàn).
- [ ] PSNR public ra số dương hợp lý (nếu bị ~5–8 dB → sai convention pose/intrinsics, cần soi lại).

## Ghi chú khi mở rộng
- Chạy full: bỏ `--iterations 7000` (mặc định 30000) → chất lượng cao hơn nhiều.
- Lặp qua tất cả scene: bọc bước 3–4 trong vòng for theo danh sách thư mục scene.
- Kaggle giới hạn ~9h/phiên GPU và ~20GB working; nhiều scene nên tách phiên hoặc lưu output ra dataset.
