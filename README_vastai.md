# Template vast.ai cho pipeline 3DGS

## 1. Tạo template (làm 1 lần)

Vast.ai Console → **Templates** → **+ New** (hoặc Edit khi thuê) — điền:

| Trường | Giá trị | Tại sao |
|---|---|---|
| **Docker Image** | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` | Phải là bản **`-devel`** (có nvcc) mới build được CUDA extension. Bản `-runtime` sẽ fail. |
| **Launch Mode** | SSH | chạy pipeline qua terminal/tmux |
| **On-start Script** | dán nội dung `vastai_onstart.sh` | tự cài tool + clone + build 3DGS khi máy boot (~5-8') |
| **Disk Space** | **60 GB** | data 15GB + scenes prepared 15GB + output + hệ điều hành |
| Ports / Env | để trống | không cần |

## 2. Chọn máy thuê (Search)

Bộ lọc khuyến nghị trên trang Search:
- **GPU**: RTX 3090 hoặc RTX 4090 (1 GPU, ≥24GB VRAM)
- **Reliability** ≥ 99%, **Verified** ✓
- **Inet Down** ≥ 200 Mbps (tải data 15GB cho nhanh)
- **Max Duration** ≥ 1 ngày
- Giá tham khảo: 3090 ~$0.15–0.25/h, 4090 ~$0.30–0.45/h
  → cả job 8 scene @30k ≈ 4–8h ≈ **$1–3**

Bấm **Rent** với template ở bước 1.

## 3. Sau khi máy chạy

Lấy lệnh SSH từ nút **Connect** (dạng `ssh -p PORT root@sshX.vast.ai`).

```bash
# kiểm tra on-start đã build xong chưa (đợi có file này rồi hãy chạy)
ls /workspace/SETUP_DONE || tail -f /var/log/onstart.log
```

Upload code + data từ máy Windows (PowerShell):
```powershell
scp -P PORT -r D:\4_Code\Viettel_AI_2026\phase1\ViettelAI2026 root@sshX.vast.ai:/workspace/
scp -P PORT -r D:\4_Code\Viettel_AI_2026\phase1\private_set1  root@sshX.vast.ai:/workspace/data/
```
(Mạng nhà chậm? Đẩy data lên Google Drive 1 lần rồi trên máy thuê:
`pip install gdown && gdown <FILE_ID>` — băng thông datacenter nhanh hơn nhiều.)

## 4. Chạy pipeline (trong tmux)

```bash
tmux new -s train
python /workspace/ViettelAI2026/run_all_kaggle.py \
    --data /workspace/data/private_set1 \
    --repo /workspace/gaussian-splatting \
    --pipe /workspace/ViettelAI2026 \
    --scenes-dir /workspace/scenes \
    --out /workspace/output \
    --iters 30000
# rời tmux: Ctrl+B, D — quay lại: tmux attach -t train
# theo dõi: watch -n 60 cat /workspace/output/status.json
```
Resume tự động: instance bị restart → chạy lại đúng lệnh trên (`/workspace` được giữ),
scene đã xong tự skip theo file thực có.

## 5. Đóng gói + tải về + TẮT MÁY

```bash
cd /workspace/output
mkdir -p /tmp/sub && for d in */test_renders; do s=$(dirname $d); mkdir -p /tmp/sub/$s; cp $d/* /tmp/sub/$s/; done
cd /tmp && zip -rq /workspace/submission.zip sub
```
```powershell
scp -P PORT root@sshX.vast.ai:/workspace/submission.zip D:\4_Code\Viettel_AI_2026\phase1\
```

Checklist trước khi **Destroy** instance (đang tính tiền theo giờ + tiền disk):
- [ ] `status.json`: đủ 8 scene `render: done`, không `error`
- [ ] Số ảnh khớp CSV (private: 26–60 pose/scene, không đều nhau)
- [ ] Đã scp `submission.zip` về và mở xem thử vài ảnh
- [ ] (tuỳ chọn) tải checkpoint `output/*/point_cloud/` nếu muốn render lại sau này
- Lưu ý: **Stop** vẫn tính tiền disk; **Destroy** mới hết phí hoàn toàn.

## Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử |
|---|---|
| build extension fail `nvcc not found` | image không phải `-devel` — đổi image trong template |
| `ImportError: libGL.so.1` khi prepare | đã phòng bằng `opencv-python-headless` trong on-start; nếu vẫn gặp: `apt-get install -y libgl1` |
| script `.sh` báo `\r: command not found` | file dính CRLF từ Windows: `sed -i 's/\r$//' <file>.sh` |
| SSH rớt giữa chừng | vô hại nếu chạy trong tmux — `tmux attach -t train` |
| OOM (chỉ khi lỡ thuê máy 16GB) | thêm `--data_device cpu` vào lệnh train trong run_all_kaggle.py (xem commit 8ebd0db) |
