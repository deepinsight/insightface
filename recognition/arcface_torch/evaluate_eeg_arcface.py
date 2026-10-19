import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_dataset import EEGTrialDataset
from eeg_model import TemporalSpatialEEGEncoder


def encode_role(cache, manifest, role, model, device, batch_size=64, workers=0):
    ds = EEGTrialDataset(cache, manifest, role, subject_to_label={})
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=(device.type == "cuda")
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for x, _labels, meta in loader:
            z = model(x.to(device)).cpu().numpy()
            subjects = meta["subject"].numpy()
            sessions = meta["session"].numpy()
            freqs = meta["frequency"].numpy()
            files = meta["file"]
            for i in range(len(z)):
                rows.append({
                    "subject": int(subjects[i]),
                    "session": int(sessions[i]),
                    "frequency": float(freqs[i]),
                    "file": str(files[i]),
                    "embedding": z[i].astype(np.float32),
                })
    return rows


def make_index(rows, role):
    index = {}
    for row in rows:
        key = (row["subject"], round(row["frequency"], 4))
        if key in index:
            raise ValueError(
                f"Expected one trial per subject/frequency for role={role}; "
                f"duplicate key={key}: {index[key]['file']} and {row['file']}"
            )
        index[key] = row
    return index


def score_pairs(enroll_rows, probe_rows):
    """All probe-vs-enrollment scores, restricted to equal stimulus frequency."""
    enroll_by_freq = {}
    for r in enroll_rows:
        enroll_by_freq.setdefault(round(r["frequency"], 4), []).append(r)

    scores, labels = [], []
    for probe in probe_rows:
        freq = round(probe["frequency"], 4)
        candidates = enroll_by_freq.get(freq, [])
        if not candidates:
            continue
        p = probe["embedding"]
        for enrolled in candidates:
            score = float(np.dot(p, enrolled["embedding"]) /
                          (max(np.linalg.norm(p), 1e-12) *
                           max(np.linalg.norm(enrolled["embedding"]), 1e-12)))
            scores.append(score)
            labels.append(int(probe["subject"] == enrolled["subject"]))
    if not scores or len(set(labels)) < 2:
        raise ValueError("Could not form both genuine and impostor comparisons.")
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def rates(scores, labels, threshold):
    accepted = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0
    far = float(np.mean(accepted[impostor])) if np.any(impostor) else float("nan")
    frr = float(np.mean(~accepted[genuine])) if np.any(genuine) else float("nan")
    return {"threshold": float(threshold), "FAR": far, "FRR": frr, "TAR": 1.0 - frr}


def eer_threshold(scores, labels):
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    thresholds = np.unique(np.concatenate([scores, [np.nextafter(scores.max(), np.inf)]]))
    best = None
    for threshold in thresholds:
        far = np.mean(impostor_scores >= threshold)
        frr = np.mean(genuine_scores < threshold)
        gap = abs(float(far) - float(frr))
        if best is None or gap < best[0]:
            best = (gap, float(threshold), float(far), float(frr))
    return {"threshold": best[1], "FAR": best[2], "FRR": best[3],
            "EER_approx": (best[2] + best[3]) / 2.0}


def auc_rank(scores, labels):
    # Mann–Whitney ROC-AUC with average ranks for ties; NumPy-only implementation.
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0  # 1-based average rank
        i = j
    rank_sum = ranks[labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_max_far(scores, labels, target_far):
    impostor = np.sort(scores[labels == 0])
    if len(impostor) == 0:
        raise ValueError("No impostor calibration scores.")
    # Pick a threshold just above the order statistic so empirical FAR does not
    # exceed the target due to >= threshold acceptance or ties.
    allowed = int(math.floor(target_far * len(impostor)))
    idx = max(0, len(impostor) - allowed - 1)
    return float(np.nextafter(impostor[idx], np.inf))


def summarize(name, scores, labels, thresholds):
    result = {
        "dataset": name,
        "comparisons_total": int(len(labels)),
        "genuine_comparisons": int((labels == 1).sum()),
        "impostor_comparisons": int((labels == 0).sum()),
        "ROC_AUC": auc_rank(scores, labels),
    }
    for key, threshold in thresholds.items():
        rr = rates(scores, labels, threshold)
        result[f"{key}_threshold"] = rr["threshold"]
        result[f"{key}_FAR"] = rr["FAR"]
        result[f"{key}_FRR"] = rr["FRR"]
        result[f"{key}_TAR"] = rr["TAR"]
    eer = eer_threshold(scores, labels)
    result["EER_threshold"] = eer["threshold"]
    result["EER_FAR"] = eer["FAR"]
    result["EER_FRR"] = eer["FRR"]
    result["EER_approx"] = eer["EER_approx"]
    return result


def main():
    p = argparse.ArgumentParser(
        description="Threshold-calibrated cross-session EEG ArcFace verification."
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(device)
    model.load_state_dict(checkpoint["encoder"], strict=True)
    model.eval()

    manifest_obj = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    seen = set(map(int, manifest_obj["identity_split"]["seen_subjects"]))
    unseen = set(map(int, manifest_obj["identity_split"]["unseen_subjects"]))
    if seen & unseen:
        raise ValueError("Manifest identity split overlaps between seen and unseen subjects.")

    roles = [
        "seen_threshold_enrollment",
        "seen_threshold_test",
        "unseen_enrollment",
        "unseen_final_test",
    ]
    encoded = {}
    for role in roles:
        print(f"Encoding role={role} ...", flush=True)
        encoded[role] = encode_role(
            args.cache, args.manifest, role, model, device,
            batch_size=args.batch_size, workers=args.workers
        )
        print(f"  trials={len(encoded[role])}", flush=True)

    # Guard against identity leakage and incorrect role contents.
    for role in ("seen_threshold_enrollment", "seen_threshold_test"):
        ids = {r["subject"] for r in encoded[role]}
        if not ids or not ids.issubset(seen):
            raise ValueError(f"{role} contains identities outside the seen split.")
    for role in ("unseen_enrollment", "unseen_final_test"):
        ids = {r["subject"] for r in encoded[role]}
        if not ids or not ids.issubset(unseen):
            raise ValueError(f"{role} contains identities outside the unseen split.")

    cal_scores, cal_labels = score_pairs(
        encoded["seen_threshold_enrollment"], encoded["seen_threshold_test"]
    )
    eer_cal = eer_threshold(cal_scores, cal_labels)
    t_far_1 = threshold_at_max_far(cal_scores, cal_labels, 0.01)
    t_far_01 = threshold_at_max_far(cal_scores, cal_labels, 0.001)
    thresholds = {
        "calibrated_EER": eer_cal["threshold"],
        "calibrated_FAR_1pct": t_far_1,
        "calibrated_FAR_0_1pct": t_far_01,
    }

    test_scores, test_labels = score_pairs(
        encoded["unseen_enrollment"], encoded["unseen_final_test"]
    )
    calibration_result = summarize("seen_threshold_calibration", cal_scores, cal_labels, {})
    calibration_result["calibration_EER_threshold"] = eer_cal["threshold"]
    calibration_result["calibration_EER_approx"] = eer_cal["EER_approx"]
    calibration_result["threshold_for_max_1pct_FAR"] = t_far_1
    calibration_result["threshold_for_max_0_1pct_FAR"] = t_far_01
    final_result = summarize("unseen_final_test_40_identities", test_scores, test_labels, thresholds)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "device": str(device),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "seen_identity_count": len(seen),
            "unseen_identity_count": len(unseen),
            "score": "cosine similarity of L2-normalized frozen encoder embeddings",
            "pairing": "probe compared only to enrollment trials at the same SSVEP frequency",
            "threshold_rule": "thresholds calibrated on seen identities only; unseen final test not used for threshold selection",
            "notes": [
                "EER on the unseen final test is descriptive only and is not used to choose the operating threshold.",
                "The 0.1% FAR operating point may be coarse because the calibration impostor set is finite.",
                "All pairwise comparisons share trials and are not statistically independent."
            ],
        },
        "calibration": calibration_result,
        "unseen_final_test": final_result,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    csv_path = output / "verification_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset"] + sorted(
            (set(calibration_result) | set(final_result)) - {"dataset"}
        ))
        writer.writeheader()
        writer.writerow(calibration_result)
        writer.writerow(final_result)

    print("\n=== SEEN-IDENTITY THRESHOLD CALIBRATION ===")
    print(f"Calibration comparisons: {len(cal_labels):,}; genuine={(cal_labels == 1).sum():,}; impostor={(cal_labels == 0).sum():,}")
    print(f"Calibration EER (approx): {eer_cal['EER_approx']:.4%}")
    print(f"Threshold for max 1% FAR:   {t_far_1:.6f}")
    print(f"Threshold for max 0.1% FAR: {t_far_01:.6f}")
    print("\n=== HELD-OUT 40-IDENTITY FINAL TEST ===")
    print(f"Trials: enrollment={len(encoded['unseen_enrollment'])}, probe={len(encoded['unseen_final_test'])}")
    print(f"Comparisons: {len(test_labels):,}; genuine={(test_labels == 1).sum():,}; impostor={(test_labels == 0).sum():,}")
    print(f"ROC-AUC: {final_result['ROC_AUC']:.6f}")
    print(f"Descriptive test EER (not used for threshold selection): {final_result['EER_approx']:.4%}")
    for key in thresholds:
        print(f"{key}: threshold={final_result[key + '_threshold']:.6f}, "
              f"FAR={final_result[key + '_FAR']:.4%}, "
              f"FRR={final_result[key + '_FRR']:.4%}, "
              f"TAR={final_result[key + '_TAR']:.4%}")
    print(f"\nSaved: {output / 'verification_report.json'}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
