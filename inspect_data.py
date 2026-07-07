"""
Kiểm tra & parse nhanh 1 scene: đếm ảnh, đọc cameras.bin (model + intrinsics),
kiểm tra test_poses.csv, đối chiếu tên ảnh test giữa CSV và thư mục GT (nếu có).

    python pipeline/inspect_data.py --scene /kaggle/input/vt2026/public_set/hcm0031
"""
import os
import csv
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
            n = MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack("<" + "d" * n, f.read(8 * n))
            cams.append((cam_id, CAMERA_MODELS.get(model_id, model_id), w, h, params))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    s = args.scene

    train_imgs = os.path.join(s, "train", "images")
    n_train = len([x for x in os.listdir(train_imgs)]) if os.path.isdir(train_imgs) else 0
    print(f"train images : {n_train}")

    cam_bin = os.path.join(s, "train", "sparse", "0", "cameras.bin")
    if os.path.exists(cam_bin):
        for c in read_cameras_bin(cam_bin):
            print(f"camera       : id={c[0]} model={c[1]} WxH={c[2]}x{c[3]} params={tuple(round(p,2) for p in c[4])}")

    poses = os.path.join(s, "test", "test_poses.csv")
    with open(poses, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"test poses   : {len(rows)}  cols={list(rows[0].keys())}")
    r0 = rows[0]
    print(f"  sample WxH={r0['width']}x{r0['height']} fx={r0['fx']} cx={r0['cx']} cy={r0['cy']}")

    gt = os.path.join(s, "test", "images")
    if os.path.isdir(gt):
        gt_names = set(os.listdir(gt))
        csv_names = {r["image_name"] for r in rows}
        print(f"GT test imgs : {len(gt_names)} | khớp CSV: {len(gt_names & csv_names)} | "
              f"CSV thiếu GT: {len(csv_names - gt_names)}")
    else:
        print("GT test imgs : (không có — đây là private set, chỉ render để nộp)")


if __name__ == "__main__":
    main()
