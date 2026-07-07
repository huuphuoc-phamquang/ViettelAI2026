"""
Render 3DGS tại các camera pose cho trong test_poses.csv.

Chạy TỪ TRONG thư mục repo gaussian-splatting (để import được các module của repo).
Ví dụ:
    cd /kaggle/working/gaussian-splatting
    python /kaggle/working/pipeline/render_test_poses.py \
        --model output/hcm0031 \
        --poses /kaggle/input/vt2026/public_set/hcm0031/test/test_poses.csv \
        --out output/hcm0031/test_renders \
        --iteration 7000

Test poses dùng convention COLMAP (giống images.bin): quaternion (qw,qx,qy,qz) và
translation (tx,ty,tz) là phép biến đổi WORLD -> CAMERA. Đây đúng là convention mà
dataset_readers.py của 3DGS dùng khi đọc COLMAP, nên ta tái sử dụng y hệt:
    R = qvec2rotmat(q).T ;  T = t
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
from torchvision.utils import save_image

# ---- import từ repo gaussian-splatting (cwd phải là repo root) ----
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from arguments import PipelineParams


def qvec2rotmat(q):
    """Quaternion (w,x,y,z) -> rotation matrix. Giống colmap_loader của repo."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def build_minicam(row, znear=0.01, zfar=100.0):
    q = [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])]
    t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
    W, H = int(row["width"]), int(row["height"])
    fx, fy = float(row["fx"]), float(row["fy"])

    R = qvec2rotmat(q).T          # cam-to-world rotation, đúng như repo
    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)

    world_view = torch.tensor(getWorld2View2(R, t)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    cam = MiniCam(W, H, FoVy, FoVx, znear, zfar, world_view, full_proj)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="thư mục -m khi train (chứa point_cloud/ và cfg_args)")
    ap.add_argument("--poses", required=True, help="đường dẫn test_poses.csv")
    ap.add_argument("--out", required=True, help="thư mục lưu ảnh render")
    ap.add_argument("--iteration", type=int, default=7000)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--white_background", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ply = os.path.join(args.model, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    assert os.path.exists(ply), f"Không thấy checkpoint: {ply}"

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(ply)

    bg = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    # pipe với đầy đủ attribute mặc định, không phụ thuộc version repo
    pipe = PipelineParams(argparse.ArgumentParser()).extract(
        argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                            debug=False, antialiasing=False))

    with open(args.poses, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[render] {len(rows)} poses -> {args.out}")
    for i, row in enumerate(rows):
        cam = build_minicam(row)
        with torch.no_grad():
            img = render(cam, gaussians, pipe, bg)["render"].clamp(0.0, 1.0)
        save_image(img, os.path.join(args.out, row["image_name"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}")
    print("[render] done")


if __name__ == "__main__":
    main()
