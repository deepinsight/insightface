import argparse, torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGTrialDataset
from eeg_model import TemporalSpatialEEGEncoder, ArcFaceHead

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--manifest", required=True)
args = p.parse_args()

manifest = __import__("json").loads(open(args.manifest, encoding="utf-8").read())
seen = sorted(map(int, manifest["identity_split"]["seen_subjects"]))
labels = {s:i for i,s in enumerate(seen)}
ds = EEGTrialDataset(args.cache, args.manifest, "train", labels)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
x, y, meta = next(iter(loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalSpatialEEGEncoder().to(device)
head = ArcFaceHead(32, len(seen)).to(device)
x, y = x.to(device), y.to(device)
z = model(x)
logits = head(z, y)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
assert z.shape == (4, 32), z.shape
assert logits.shape == (4, 60), logits.shape
assert torch.isfinite(loss)
assert any(p.grad is not None for p in model.parameters())
print("SMOKE TEST PASSED")
print("device:", device)
print("input:", tuple(x.shape), "embedding:", tuple(z.shape), "logits:", tuple(logits.shape))
print("loss:", float(loss.detach()), "embedding_norms:", z.norm(dim=1).detach().cpu().tolist())
print("sample metadata:", meta["subject"], meta["session"], meta["frequency"])
