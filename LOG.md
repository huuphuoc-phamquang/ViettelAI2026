# LOG dự án — Viettel AI 2026 Phase 1: Novel View Synthesis (3DGS)

> File này là "bộ nhớ" của dự án. Chuyển máy mới: đọc file này + README tương ứng
> với môi trường định chạy (Kaggle / GPU thuê / vast.ai) là đủ ngữ cảnh làm tiếp.

## Bài toán
- Mỗi scene: ảnh drone (đã resize 1/4, 1320×989) + COLMAP sparse → train 3D Gaussian
  Splatting → render tại các pose trong `test/test_poses.csv` → nộp ảnh render.
- **public_set** (5 scene) có GT test để đo PSNR local; **private_set1** (8 scene)
  không có GT — là phần BTC chấm, chỉ render nộp.
- Data đặt ngoài repo: `phase1/public_set/`, `phase1/private_set1/`.

## Các phát hiện quan trọng về data (đã xác minh bằng parse bytes)
1. `cameras.bin` là **SIMPLE_RADIAL** (f, cx, cy, k) — KHÔNG phải PINHOLE → 3DGS
   gốc assert lỗi. k đa số ~0.008, riêng **HNI0131 k=−0.115** (méo mạnh).
2. `images.bin` chứa nhiều entry hơn số ảnh phát hành (vd hcm0031: 388 entry =
   200 train + 50 test + 138 không có file) → phải lọc, tránh crash + leak pose test.
3. `test_poses.csv` convention COLMAP world→camera (giống images.bin) →
   render dùng `R = qvec2rotmat(q).T`, `T = t`. Số pose KHÔNG đều: 26–60/scene.
4. ⇒ ra đời `prepare_scene.py` (BẮT BUỘC chạy trước train): SIMPLE_RADIAL→PINHOLE,
   undistort cv2 khi |k|>1e-3, lọc images.bin, xuất scene ra thư mục ghi được.

## Cấu trúc repo
| File | Vai trò |
|---|---|
| `prepare_scene.py` | chuẩn hoá scene (bắt buộc trước train) |
| `render_test_poses.py` | render từ CSV — mảnh 3DGS gốc không có; tự thêm repo vào sys.path (env `GS_REPO`); lưu PIL JPEG q95 |
| `run_all_kaggle.py` | chạy nhiều scene, resume theo FILE THỰC CÓ (không tin status.json), log tiến độ `status.json` |
| `inspect_data.py` / `eval_metrics.py` | kiểm tra data / PSNR (public) |
| `vt2026_3dgs_kaggle.ipynb` | notebook Kaggle tự chứa (snapshot — bản "sống" ở `phase1/` ngoài repo) |
| `README_kaggle.md` / `README_gpu.md` / `README_vastai.md` | hướng dẫn theo môi trường |
| `setup_gpu.sh` / `vastai_onstart.sh` | setup máy GPU thuê / on-start template vast.ai |

## Kết quả đến nay (10/07/2026)

### Public set — pipeline validated ✅ (@7000 iter, Kaggle T4)
| Scene | PSNR test (dB) | Scene | PSNR test (dB) |
|---|---|---|---|
| HCM0181 | 20.04 | hcm0031 | 21.31 |
| HCM0193 | 21.57 | hcm0034 | 20.94 |
| HCM0204 | 20.17 | **TB** | **20.8** |

Gap train↔test ~1.5 dB → convention pose đúng. ~25–29 phút/scene trên T4.

### Private set — DANG DỞ 3/8 scene (@30000 iter, Kaggle T4)
- ✅ Xong: **HCM1439** (26 ảnh), **HNI0131** (60), **HNI0265** (52)
  — renders đã tải về `phase1/kaggle_output/output/` (72.8MB, đã đối chiếu đủ theo CSV)
- ❌ OOM: HCM0249, HCM0254, HCM0276, HNI0366 | ⏳ HNI0437 mới prepare
- Nguyên nhân OOM: T4 14.5GB không đủ cho 30k iter (ảnh chiếm ~3.7GB VRAM + ~4GB
  phân mảnh). Fix có sẵn trong lịch sử git (`git show 8ebd0db`): `--data_device cpu`
  + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — ĐÃ REVERT (8ea82d9) vì
  chuyển hướng thuê GPU 24GB; thuê máy 16GB thì lấy lại fix này.

### Kế hoạch hiện tại
Thuê GPU trên **vast.ai** (RTX 3090/4090 24GB) chạy nốt private @30k (~4–8h, $1–3).
→ Làm theo `README_vastai.md` (template + onstart đã sẵn). 3 scene đã xong có thể
train lại cho đồng nhất (checkpoint không mang theo) hoặc giữ kết quả cũ.

## Số liệu tham khảo
- T4: 7k ≈ 24'/scene; 30k ≈ 2.3–2.6h/scene. 4090 ước tính nhanh hơn ~4–5x.
- Kaggle dataset của user mount tại `/kaggle/input/datasets/huphcphmquang/viettel-ai-2026/phase1/...`
- Tải output Kaggle không cần CLI: API `kernels/output` + header `Authorization: Bearer KGAT_...`
  (basic-auth bị 403), phân trang `pageToken`. Token cũ đã lộ trong chat → **nên thu hồi và tạo mới**.

## Việc còn lại (theo thứ tự)
1. Thuê máy vast.ai theo `README_vastai.md`, chạy nốt private_set1 @30k
2. Kiểm tra bằng mắt renders (nhất là HNI0131 — méo mạnh, không có GT)
3. Đối chiếu format `submission.zip` (`<scene>/<image_name>.JPG`) với thể lệ BTC rồi nộp
4. (Nâng điểm) đo 7k vs 30k trên public; thử chỉnh densification / exposure nếu còn thời gian

## Ghi chú môi trường dev
- Máy Windows local KHÔNG có Python thật (chỉ stub WindowsApps) → không chạy code local;
  verify binary bằng PowerShell. Notebook build/patch bằng script PowerShell.
- Remote: `https://github.com/huuphuoc-phamquang/ViettelAI2026.git` — chuyển máy
  chỉ cần `git clone` URL này. Data không nằm trong repo (tự copy riêng).
- Thư mục `phase1/pipeline/` là bản copy đồng bộ của repo (dư — có thể xoá).
