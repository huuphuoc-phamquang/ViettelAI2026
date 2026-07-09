# Chạy pipeline trên máy GPU thuê (vast.ai / runpod / server riêng)

Code pipeline **không dính cứng vào Kaggle** — mọi đường dẫn đều là tham số,
chỉ cần làm theo 5 bước dưới.

## 0. Chọn máy

| GPU | VRAM | Ước tính 30k iter/scene | Ghi chú |
|---|---|---|---|
| RTX 3090 | 24GB | ~45–60' | rẻ nhất trên vast.ai, quá đủ VRAM |
| RTX 4090 | 24GB | ~25–35' | nhanh nhất trong tầm giá |
| A5000 | 24GB | ~50–70' | ổn định, hay có trên runpod |
| T4 (tham chiếu Kaggle) | 16GB | ~150' | @30k từng bị OOM 4/8 scene |

- **VRAM ≥ 24GB** là an toàn tuyệt đối cho 30k iter (16GB phải thêm `--data_device cpu` vào lệnh train trong `run_all_kaggle.py`).
- Disk ≥ 50GB (data ~15GB + scenes prepared ~15GB + output vài GB).
- 8 scene private × 30k ≈ **4–8 giờ** trên 3090/4090 → thuê theo giờ rất rẻ.
- Template: chọn image **PyTorch** (có sẵn torch + CUDA); cần thêm `git`, `tmux`.

## 1. Upload code + data lên máy

Từ máy Windows (PowerShell, dùng scp — thay `user@host -p PORT` theo thông tin máy thuê):
```powershell
# code
scp -P PORT -r D:\4_Code\Viettel_AI_2026\phase1\ViettelAI2026 user@host:~/vt2026/
# data (nặng ~15GB — nếu chậm, nén trước hoặc dùng link tải riêng)
scp -P PORT -r D:\4_Code\Viettel_AI_2026\phase1\private_set1 user@host:~/vt2026/data/
scp -P PORT -r D:\4_Code\Viettel_AI_2026\phase1\public_set  user@host:~/vt2026/data/   # tuỳ chọn, để đo PSNR
```
Mẹo nhanh hơn: upload data 1 lần lên Google Drive/Kaggle Dataset rồi tải trực tiếp trên máy thuê
(`gdown <file_id>` hoặc `curl` — băng thông datacenter nhanh hơn mạng nhà nhiều).

## 2. Setup môi trường (1 lệnh)

```bash
cd ~/vt2026/ViettelAI2026 && bash setup_gpu.sh
```
Script tự: kiểm tra GPU → dò compute capability (không cần tra bảng sm_XX) →
clone + build 3DGS → smoke test import. Xong in sẵn lệnh chạy.

## 3. Chạy (trong tmux để rớt SSH không mất tiến trình)

```bash
tmux new -s train
python ~/vt2026/ViettelAI2026/run_all_kaggle.py \
    --data ~/vt2026/data/private_set1 \
    --repo ~/vt2026/gaussian-splatting \
    --pipe ~/vt2026/ViettelAI2026 \
    --scenes-dir ~/vt2026/scenes \
    --out ~/vt2026/output \
    --iters 30000
# rời tmux: Ctrl+B rồi D  |  quay lại: tmux attach -t train
```
- Resume tự động: máy/tiến trình chết → chạy lại đúng lệnh trên, scene xong tự skip (theo file thực có).
- Public set (đo PSNR): đổi `--data .../public_set` và thêm `--eval`.
- Theo dõi: `watch -n 60 cat ~/vt2026/output/status.json` (cửa sổ tmux khác).

## 4. Đóng gói + tải kết quả về

Trên máy thuê:
```bash
cd ~/vt2026/output
mkdir -p /tmp/submission
for d in */test_renders; do s=$(dirname $d); mkdir -p /tmp/submission/$s; cp $d/* /tmp/submission/$s/; done
cd /tmp && zip -rq ~/submission.zip submission
```
Về máy Windows:
```powershell
scp -P PORT user@host:~/submission.zip D:\4_Code\Viettel_AI_2026\phase1\
```

## 5. Trước khi tắt máy thuê (tính tiền theo giờ!)

- [ ] `status.json`: đủ 8 scene `render: done`, không scene nào có `error`
- [ ] Số ảnh mỗi scene khớp CSV (private: 26–60 pose/scene, KHÔNG đều nhau)
- [ ] Đã tải `submission.zip` về và mở ra xem thử vài ảnh
- [ ] (nếu muốn giữ) tải thêm checkpoint `output/*/point_cloud/` — không tải thì mất khi huỷ máy
