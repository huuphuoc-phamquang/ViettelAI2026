"""
Sanity-check trên PUBLIC set: so ảnh render với ảnh GT trong test/images.
Chỉ dùng được cho public_set (private không có GT).

    python pipeline/eval_metrics.py \
        --renders output/hcm0031/test_renders \
        --gt /kaggle/input/vt2026/public_set/hcm0031/test/images
"""
import os
import argparse
import numpy as np
from PIL import Image


def load(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else -10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    names = sorted(os.listdir(args.renders))
    scores = []
    missing = 0
    for n in names:
        gt_path = os.path.join(args.gt, n)
        if not os.path.exists(gt_path):
            missing += 1
            continue
        r = load(os.path.join(args.renders, n))
        g = load(gt_path, size=(r.shape[1], r.shape[0]))
        scores.append(psnr(r, g))

    if scores:
        print(f"[eval] N={len(scores)}  PSNR mean={np.mean(scores):.2f} dB  "
              f"min={np.min(scores):.2f}  max={np.max(scores):.2f}")
    if missing:
        print(f"[eval] {missing} ảnh render không có GT tương ứng")


if __name__ == "__main__":
    main()
