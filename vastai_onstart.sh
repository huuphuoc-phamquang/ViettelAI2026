#!/bin/bash
# =============================================================================
# On-start script cho template vast.ai — dán vào ô "On-start Script"
# Tự chạy khi instance khởi động: cài tool + clone + build 3DGS.
# Khi SSH vào thấy file /workspace/SETUP_DONE là môi trường đã sẵn sàng,
# chỉ còn upload data và chạy.
# Log của script này: /var/log/onstart.log (xem bằng: tail -f /var/log/onstart.log)
# =============================================================================
exec > /var/log/onstart.log 2>&1
set -x

apt-get update -y
apt-get install -y git tmux zip unzip rsync

cd /workspace

# clone 3DGS (idempotent — restart instance không clone lại)
if [ ! -d gaussian-splatting ]; then
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
fi

# tự dò compute capability của GPU được cấp
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch;cc=torch.cuda.get_device_capability(0);print(f'{cc[0]}.{cc[1]}')")
echo "ARCH=$TORCH_CUDA_ARCH_LIST"

# deps — dùng opencv HEADLESS: image docker không có libGL, bản thường sẽ lỗi import
pip install plyfile tqdm opencv-python-headless pillow numpy

# build CUDA extensions (cần image *-devel mới có nvcc)
pip install gaussian-splatting/submodules/diff-gaussian-rasterization
pip install gaussian-splatting/submodules/simple-knn
pip install gaussian-splatting/submodules/fused-ssim || echo "fused-ssim fail (bo qua duoc)"

python -c "import diff_gaussian_rasterization, simple_knn; print('BUILD OK')" && touch /workspace/SETUP_DONE
