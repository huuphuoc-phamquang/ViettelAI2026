"""
Chuẩn hoá 1 scene về dạng 3DGS train được, xuất ra thư mục ghi được.

Vấn đề của data gốc (đã xác minh trên mọi scene):
  1. cameras.bin dùng model SIMPLE_RADIAL (f, cx, cy, k) -> train.py của 3DGS
     assert vì chỉ nhận PINHOLE/SIMPLE_PINHOLE.
  2. images.bin chứa NHIỀU entry hơn số ảnh phát hành (gồm cả ảnh test và ảnh
     không có file) -> crash thiếu file + leak pose test vào train.
  3. Hệ số méo k đa số nhỏ (~0.008) nhưng HNI0131 k=-0.115 (méo mạnh).

Script này:
  - Undistort ảnh train bằng cv2 (k1, đúng convention COLMAP SIMPLE_RADIAL)
    nếu |k| > --k-thresh, ngược lại copy/symlink ảnh gốc.
  - Ghi cameras.bin mới: PINHOLE (fx=fy=f, cx, cy).
  - Lọc images.bin: chỉ giữ entry có file trong train/images (copy raw bytes
    từng entry nên nhanh, không cần decode points2D).
  - Copy points3D.bin / points3D.ply. Bỏ frames.bin/rigs.bin (3DGS không đọc).

Dùng:
    python prepare_scene.py --scene /kaggle/input/vt2026/public_set/hcm0031 \
                            --out   /kaggle/working/scenes/hcm0031
Sau đó train với:  -s /kaggle/working/scenes/hcm0031/train
"""
import os
import shutil
import struct
import argparse

CAMERA_MODELS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                 3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}
MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}


def read_cameras_bin(path):
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        cams = []
        for _ in range(num):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            n = MODEL_NUM_PARAMS[model_id]
            params = struct.unpack("<" + "d" * n, f.read(8 * n))
            cams.append(dict(id=cam_id, model=model_id, w=w, h=h, params=params))
    return cams


def write_pinhole_cameras_bin(path, cams):
    """Ghi cameras.bin với mọi camera đã quy về PINHOLE (fx, fy, cx, cy)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            fx, fy, cx, cy = c["pinhole"]
            f.write(struct.pack("<ii", c["id"], 1))          # model 1 = PINHOLE
            f.write(struct.pack("<QQ", c["w"], c["h"]))
            f.write(struct.pack("<dddd", fx, fy, cx, cy))


def filter_images_bin(src, dst, keep_names):
    """Giữ lại các entry có tên trong keep_names. Copy raw bytes từng entry."""
    data = open(src, "rb").read()
    num = struct.unpack("<Q", data[:8])[0]
    off = 8
    kept = []
    for _ in range(num):
        start = off
        off += 4 + 32 + 24 + 4                    # id + qvec + tvec + cam_id
        name_start = off
        while data[off] != 0:
            off += 1
        name = data[name_start:off].decode("utf-8")
        off += 1                                   # null
        n2d = struct.unpack("<Q", data[off:off + 8])[0]
        off += 8 + n2d * 24                        # x,y (2d) + point3D_id (q)
        if name in keep_names:
            kept.append(data[start:off])
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for chunk in kept:
            f.write(chunk)
    return num, len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="thư mục scene gốc (read-only OK)")
    ap.add_argument("--out", required=True, help="thư mục xuất (ghi được)")
    ap.add_argument("--k-thresh", type=float, default=1e-3,
                    help="|k| lớn hơn ngưỡng này thì undistort ảnh")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy ảnh thay vì symlink (khi symlink không dùng được)")
    args = ap.parse_args()

    src_sparse = os.path.join(args.scene, "train", "sparse", "0")
    src_images = os.path.join(args.scene, "train", "images")
    dst_train = os.path.join(args.out, "train")
    dst_sparse = os.path.join(dst_train, "sparse", "0")
    dst_images = os.path.join(dst_train, "images")
    os.makedirs(dst_sparse, exist_ok=True)

    # ---- 1. Camera: SIMPLE_RADIAL -> PINHOLE ----
    cams = read_cameras_bin(os.path.join(src_sparse, "cameras.bin"))
    need_undistort = False
    for c in cams:
        m = c["model"]
        if m == 1:                                  # PINHOLE sẵn
            c["pinhole"] = c["params"]
            c["k"] = 0.0
        elif m == 0:                                # SIMPLE_PINHOLE: f,cx,cy
            f_, cx, cy = c["params"]
            c["pinhole"] = (f_, f_, cx, cy)
            c["k"] = 0.0
        elif m in (2, 3):                           # SIMPLE_RADIAL / RADIAL
            f_, cx, cy, k = c["params"][:4]
            c["pinhole"] = (f_, f_, cx, cy)
            c["k"] = k
            if abs(k) > args.k_thresh:
                need_undistort = True
        else:
            raise SystemExit(f"Camera model {CAMERA_MODELS.get(m, m)} chưa hỗ trợ")
        print(f"[camera] id={c['id']} {CAMERA_MODELS[m]} k={c['k']:.5f} "
              f"-> PINHOLE {tuple(round(p, 2) for p in c['pinhole'])}"
              f"{'  (sẽ undistort ảnh)' if abs(c['k']) > args.k_thresh else ''}")
    write_pinhole_cameras_bin(os.path.join(dst_sparse, "cameras.bin"), cams)

    # ---- 2. images.bin: lọc theo file thực có ----
    train_files = set(os.listdir(src_images))
    total, kept = filter_images_bin(os.path.join(src_sparse, "images.bin"),
                                    os.path.join(dst_sparse, "images.bin"),
                                    train_files)
    print(f"[images.bin] {total} entry -> giữ {kept} (khớp {len(train_files)} file train)")

    # ---- 3. points3D ----
    for fn in ("points3D.bin", "points3D.ply"):
        p = os.path.join(src_sparse, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_sparse, fn))

    # ---- 4. Ảnh: undistort hoặc link/copy ----
    if need_undistort:
        import cv2
        import numpy as np
        if os.path.islink(dst_images):
            os.unlink(dst_images)      # symlink cũ trỏ vào input read-only
        os.makedirs(dst_images, exist_ok=True)
        c = cams[0]                                 # các scene này đều 1 camera
        fx, fy, cx, cy = c["pinhole"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([c["k"], 0, 0, 0], dtype=np.float64)   # k1 only
        names = sorted(train_files)
        for i, n in enumerate(names):
            img = cv2.imread(os.path.join(src_images, n))
            und = cv2.undistort(img, K, dist)       # giữ nguyên K -> pinhole
            cv2.imwrite(os.path.join(dst_images, n),
                        und, [cv2.IMWRITE_JPEG_QUALITY, 98])
            if (i + 1) % 50 == 0:
                print(f"[undistort] {i + 1}/{len(names)}")
        print(f"[undistort] xong {len(names)} ảnh (k={c['k']:.5f})")
    else:
        if os.path.islink(dst_images) or os.path.isdir(dst_images):
            pass                                    # đã có từ lần chạy trước
        elif args.copy_images:
            shutil.copytree(src_images, dst_images)
        else:
            os.symlink(src_images, dst_images)
        print("[images] k nhỏ, dùng ảnh gốc (symlink/copy)")

    print(f"[prepare] OK -> {dst_train}")


if __name__ == "__main__":
    main()
