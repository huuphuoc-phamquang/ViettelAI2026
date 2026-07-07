"""
Chạy pipeline cho NHIỀU scene, có checkpoint (status.json) để RESUME.

    python pipeline/run_all_kaggle.py \
        --data /kaggle/input/vt2026/public_set \
        --repo /kaggle/working/gaussian-splatting \
        --pipe /kaggle/working/pipeline \
        --out  /kaggle/working/output \
        --iters 7000 --eval

Cơ chế đánh dấu tiến độ: sau MỖI bước (train / render) ghi ngay vào
<out>/status.json. Chạy lại script -> scene nào 'render=done' sẽ bỏ qua,
scene đang dở sẽ tiếp tục từ bước còn thiếu (train skip nếu đã có checkpoint).
"""
import os
import json
import time
import argparse
import datetime


def load_status(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_status(path, s):
    tmp = path + ".tmp"
    json.dump(s, open(tmp, "w"), indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # ghi atomic, tránh hỏng file nếu bị kill giữa chừng


def sh(cmd):
    print(">>", cmd, flush=True)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"exit {rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục chứa các scene (public_set / private_set1)")
    ap.add_argument("--repo", default="/kaggle/working/gaussian-splatting")
    ap.add_argument("--pipe", default="/kaggle/working/pipeline")
    ap.add_argument("--out", default="/kaggle/working/output")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--eval", action="store_true", help="tính PSNR (chỉ public set có GT)")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ chạy các scene này")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "status.json")
    status = load_status(status_path)

    scenes = sorted(d for d in os.listdir(args.data)
                    if os.path.isdir(os.path.join(args.data, d)))
    if args.only:
        scenes = [s for s in scenes if s in args.only]

    done = [k for k, v in status.items() if v.get("render") == "done"]
    print(f"Tổng {len(scenes)} scene | đã xong {len(done)}: {done}")

    for scene in scenes:
        st = status.get(scene, {})
        if st.get("render") == "done":
            print(f"[skip] {scene}")
            continue

        s_in = os.path.join(args.data, scene)
        m_out = os.path.join(args.out, scene)
        renders = os.path.join(m_out, "test_renders")
        ckpt = os.path.join(m_out, "point_cloud", f"iteration_{args.iters}", "point_cloud.ply")
        t0 = time.time()
        print(f"\n===== {scene} =====", flush=True)
        try:
            # 1) TRAIN (bỏ qua nếu đã có checkpoint)
            if st.get("train") != "done" or not os.path.exists(ckpt):
                sh(f"cd {args.repo} && python train.py -s {s_in}/train -m {m_out} "
                   f"--iterations {args.iters} --save_iterations {args.iters} "
                   f"--test_iterations {args.iters} -r 1")
                st["train"] = "done"
                status[scene] = st
                save_status(status_path, status)   # <-- checkpoint sau train

            # 2) RENDER test poses
            sh(f"cd {args.repo} && python {args.pipe}/render_test_poses.py "
               f"--model {m_out} --poses {s_in}/test/test_poses.csv "
               f"--out {renders} --iteration {args.iters}")
            st["render"] = "done"

            # 3) EVAL (public)
            gt = os.path.join(s_in, "test", "images")
            if args.eval and os.path.isdir(gt):
                os.system(f"python {args.pipe}/eval_metrics.py --renders {renders} --gt {gt}")

            st["sec"] = round(time.time() - t0, 1)
            st["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
            st.pop("error", None)
        except Exception as e:
            st["error"] = str(e)
            print(f"[FAIL] {scene}: {e}", flush=True)

        status[scene] = st
        save_status(status_path, status)          # <-- checkpoint sau mỗi scene

    print("\n=== STATUS ===")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
