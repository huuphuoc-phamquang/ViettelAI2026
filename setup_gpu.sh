#!/usr/bin/env bash
# =============================================================================
# Setup 3DGS pipeline trên máy GPU thuê (vast.ai / runpod / server Ubuntu)
#
# Yêu cầu trước khi chạy script này:
#   - Image/máy có sẵn: Ubuntu 20.04+, NVIDIA driver, PyTorch + CUDA
#     (trên vast.ai / runpod chọn template "PyTorch" là đủ)
#   - Đã upload thư mục code ViettelAI2026/ và data lên máy (xem README_gpu.md)
#
# Chạy:  bash setup_gpu.sh
# =============================================================================
set -e

WORK=${WORK:-$HOME/vt2026}
mkdir -p "$WORK" && cd "$WORK"

echo "===== 0. Kiểm tra GPU ====="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"

echo "===== 1. Tự dò compute capability để build rasterizer ====="
export TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(f"{cc[0]}.{cc[1]}")
PY
)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "===== 2. Clone + build gaussian-splatting ====="
if [ ! -d gaussian-splatting ]; then
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
fi
pip install -q plyfile tqdm opencv-python pillow numpy
pip install -q gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -q gaussian-splatting/submodules/simple-knn
pip install -q gaussian-splatting/submodules/fused-ssim || echo "(fused-ssim lỗi — bỏ qua được)"

echo "===== 3. Smoke test import ====="
python -c "import diff_gaussian_rasterization, simple_knn; print('build OK')"

echo ""
echo "===== SETUP XONG. Chạy pipeline: ====="
echo "tmux new -s train"
echo "python $WORK/ViettelAI2026/run_all_kaggle.py \\"
echo "    --data $WORK/data/private_set1 \\"
echo "    --repo $WORK/gaussian-splatting \\"
echo "    --pipe $WORK/ViettelAI2026 \\"
echo "    --scenes-dir $WORK/scenes \\"
echo "    --out $WORK/output \\"
echo "    --iters 30000"
