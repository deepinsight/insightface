import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eeg_dataset import EEGTrialDataset
from eeg_model import TemporalSpatialEEGEncoder, ArcFaceHead

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def validate(backbone, head, loader, device):
    backbone.eval(); head.eval()
    losses, correct, total = [], 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        z = backbone(x)
        logits = head(z, y)
        losses.append(F.cross_entropy(logits, y).item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"loss": float(np.mean(losses)), "accuracy": correct / max(total, 1), "n": total}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    seen = sorted(map(int, manifest["identity_split"]["seen_subjects"]))
    subject_to_label = {s:i for i,s in enumerate(seen)}
    train_ds = EEGTrialDataset(args.cache, args.manifest, "train", subject_to_label)
    val_ds = EEGTrialDataset(args.cache, args.manifest, "seen_validation", subject_to_label)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))
    model = TemporalSpatialEEGEncoder(channels=8, embedding_dim=32).to(device)
    head = ArcFaceHead(32, len(seen), scale=32.0, margin=0.35).to(device)
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = float("inf")
    history = []
    print(f"Device={device}; train={len(train_ds)}; validation={len(val_ds)}; classes={len(seen)}")
    print(f"Parameters: encoder={sum(p.numel() for p in model.parameters()):,}; head={sum(p.numel() for p in head.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        losses, correct, total = [], 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(x)
            logits = head(embeddings, y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            optimizer.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        scheduler.step()
        val = validate(model, head, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "train_accuracy": correct/max(total,1), "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.3f} val_loss={val['loss']:.4f} "
              f"val_acc={val['accuracy']:.3f}")
        # Checkpoint selection uses session 2 validation only.
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"epoch": epoch, "encoder": model.state_dict(),
                        "arcface_head": head.state_dict(), "optimizer": optimizer.state_dict(),
                        "seed": args.seed, "embedding_dim": 32,
                        "channels": manifest["channels_in_order"],
                        "seen_subjects": seen, "manifest_path": str(Path(args.manifest).resolve())},
                       output / "best_checkpoint.pt")
        torch.save({"epoch": epoch, "encoder": model.state_dict(),
                    "arcface_head": head.state_dict()}, output / "last_checkpoint.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output / 'best_checkpoint.pt'}")
    print("Training finished. This is not final biometric verification; run the separate evaluator next.")

if __name__ == "__main__":
    main()
