import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class EEGTrialDataset(Dataset):
    def __init__(self, cache_dir, manifest_path, role, subject_to_label=None,
                 normalize_trial=True):
        self.cache_dir = Path(cache_dir)
        self.manifest = load_manifest(manifest_path)
        self.records = [r for r in self.manifest["records"] if r["role"] == role]
        if not self.records:
            raise ValueError(f"No records for role={role!r} in {manifest_path}")
        self.channels = list(self.manifest.get("channels_in_order", CHANNELS))
        self.normalize_trial = normalize_trial
        if subject_to_label is None:
            subjects = sorted({int(r["subject"]) for r in self.records})
            self.subject_to_label = {s: i for i, s in enumerate(subjects)}
        else:
            self.subject_to_label = dict(subject_to_label)
        with open(self.cache_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata, dict):
            for key in ("trials", "records", "items", "data"):
                if isinstance(metadata.get(key), list):
                    metadata = metadata[key]
                    break
        self.channel_names = metadata[0]["channel_names"]
        self.channel_indices = [self.channel_names.index(c) for c in self.channels]
        # Use metadata to validate the selected paths/shapes.
        self.metadata_by_file = {r["file"]: r for r in metadata}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        path = self.cache_dir / r["file"]
        x = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if x.shape != (64, 1500):
            raise ValueError(f"Unexpected shape {x.shape} for {path}")
        x = x[self.channel_indices, :]
        if self.normalize_trial:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            x = (x - mean) / np.maximum(std, 1e-6)
        label = self.subject_to_label.get(int(r["subject"]), -1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(label, dtype=torch.long), {
            "subject": int(r["subject"]),
            "session": int(r["session"]),
            "frequency": float(r["frequency"]),
            "file": r["file"],
        }
