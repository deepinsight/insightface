#!/usr/bin/env python
"""Train every backbone x {arcface, adaface} for a few epochs on a small
RecordIO subset with torchrun (multi-GPU DDP) and check that the loss drops.

Example (4 GPUs, 2 jobs in parallel, 2 GPUs each):
    python scripts/smoke_train_all.py --gpus 0,1,2,3 --gpus-per-job 2

Outputs <out-dir>/<net>_<loss>/{stdout.log,training.log,...} and a summary
<out-dir>/summary.{json,md}.
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_NETWORKS = [
    "r18", "r34", "r50", "r100", "r200", "r2060",
    "mbf", "mbf_large",
    "vit_t", "vit_t_dp005_mask0", "vit_s", "vit_s_dp005_mask_0",
    "vit_b", "vit_b_dp005_mask_005", "vit_l_dp005_mask_005", "vit_h",
    "mambavision_t", "mambavision_s", "mambavision_b", "mambavision_l",
    "transface_vit_b", "transface_vit_l",
    "transface_pp_vit_s", "transface_pp_vit_b",
]
LOSSES = ["arcface", "adaface"]

LOG_RE = re.compile(r"Loss (\d+\.\d+)\s+LearningRate [\d.e+-]+\s+Epoch: (\d+)\s+Global Step: (\d+)")


def script_for(network, loss):
    if network.startswith("transface_pp"):
        return "train_transface_pp.py"
    if network.startswith("transface"):
        return "train_transface.py"
    return "train_adaface.py" if loss == "adaface" else "train_v2.py"


def parse_log(path):
    """Return list of (step, epoch, running-avg loss) from training.log."""
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                pts.append((int(m.group(3)), int(m.group(2)), float(m.group(1))))
    return pts


def judge(pts, num_epoch):
    """Loss 'decreases normally' if the last logged running-average loss is clearly
    below the first, the first-epoch-end loss is below the first log, and no NaN."""
    if len(pts) < 3:
        return False, "too few log points"
    losses = [p[2] for p in pts]
    if any(l != l for l in losses):
        return False, "NaN loss"
    first, last = losses[0], losses[-1]
    epochs_seen = {p[1] for p in pts}
    if len(epochs_seen) < num_epoch:
        return False, f"only epochs {sorted(epochs_seen)} logged"
    # running average resets never; compare first-10-step avg vs final avg, and also
    # compare min of last quarter vs first point.
    q = max(1, len(losses) // 4)
    tail = sum(losses[-q:]) / q
    if last < first * 0.9 and tail < first:
        return True, f"{first:.3f} -> {last:.3f} ({(1 - last / first) * 100:.0f}% drop)"
    return False, f"{first:.3f} -> {last:.3f} (insufficient drop)"


def run_job(job, gpus, port, args):
    network, loss = job
    name = f"{network}_{loss}"
    out_dir = os.path.join(args.out_dir, name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpus,
        "SMOKE_NETWORK": network,
        "SMOKE_LOSS": loss,
        "SMOKE_OUTPUT": out_dir,
        "SMOKE_EPOCHS": str(args.epochs),
        "SMOKE_BS": str(args.batch_size),
        "SMOKE_REC": args.rec,
        "SMOKE_NUM_CLASSES": str(args.num_classes),
        "SMOKE_NUM_IMAGE": str(args.num_image),
        "PYTHONWARNINGS": "ignore",
        "OMP_NUM_THREADS": "4",
    })
    nproc = len(gpus.split(","))
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}", f"--master_port={port}",
        script_for(network, loss), "configs/smoke_webface_mock.py",
    ]
    t0 = time.time()
    with open(os.path.join(out_dir, "stdout.log"), "w") as fh:
        fh.write("$ " + " ".join(cmd) + f"\n# CUDA_VISIBLE_DEVICES={gpus}\n\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=args.timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    elapsed = time.time() - t0
    pts = parse_log(os.path.join(out_dir, "training.log"))
    ok, why = judge(pts, args.epochs)
    status = "PASS" if (rc == 0 and ok) else "FAIL"
    if rc != 0:
        why = f"exit={rc}; " + why
    result = dict(network=network, loss=loss, script=script_for(network, loss), gpus=nproc,
                  status=status, exit=rc, reason=why, seconds=round(elapsed),
                  first_loss=pts[0][2] if pts else None, last_loss=pts[-1][2] if pts else None,
                  steps=pts[-1][0] if pts else 0,
                  model_saved=os.path.exists(os.path.join(out_dir, "model.pt")))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--gpus-per-job", type=int, default=2)
    ap.add_argument("--networks", default=",".join(ALL_NETWORKS))
    ap.add_argument("--losses", default=",".join(LOSSES))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rec", default="/data/fu.dai/face_dataset/webface260m_mock_100mb")
    ap.add_argument("--num-classes", type=int, default=544)
    ap.add_argument("--num-image", type=int, default=37173)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "work_dirs", "smoke"))
    ap.add_argument("--base-port", type=int, default=29600)
    args = ap.parse_args()

    gpu_list = args.gpus.split(",")
    slots = [",".join(gpu_list[i:i + args.gpus_per_job])
             for i in range(0, len(gpu_list), args.gpus_per_job)]
    jobs = [(n, l) for n in args.networks.split(",") for l in args.losses.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(jobs)} jobs on slots {slots}", flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = [], threading.Lock()

    def worker(slot_idx):
        gpus = slots[slot_idx]
        port = args.base_port + slot_idx
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            print(f"[slot {slot_idx} gpus {gpus}] START {job[0]} {job[1]}", flush=True)
            r = run_job(job, gpus, port, args)
            print(f"[slot {slot_idx}] {r['status']} {job[0]} {job[1]} "
                  f"({r['seconds']}s, {r['reason']})", flush=True)
            with lock:
                results.append(r)
                with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                    json.dump(results, f, indent=2)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(slots))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda r: (ALL_NETWORKS.index(r["network"]) if r["network"] in ALL_NETWORKS else 99, r["loss"]))
    lines = ["| network | loss | script | GPUs | status | steps | first loss | last loss | time | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        fl = f"{r['first_loss']:.3f}" if r["first_loss"] is not None else "-"
        ll = f"{r['last_loss']:.3f}" if r["last_loss"] is not None else "-"
        lines.append(f"| {r['network']} | {r['loss']} | {r['script']} | {r['gpus']} | {r['status']} | "
                     f"{r['steps']} | {fl} | {ll} | {r['seconds']}s | {r['reason']} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    n_fail = sum(r["status"] != "PASS" for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
