"""Fine-tune the Hybrid CNN-Transformer (Section 3.5).

* Pretrained backbones (ImageNet) for EfficientNet-B4 and ViT-S/16
* Augmentation: horizontal flip, rotation, scaling (training/dataset.py)
* BCE loss (BCEWithLogitsLoss) on the classification head
* AdamW optimiser, cosine learning-rate schedule
* Early stopping on validation ROC-AUC; best model -> weights/hybrid_best.pt

CPU:    python -m training.train --data-root data --epochs 2 --batch-size 4
Colab:  python -m training.train --data-root /content/drive/MyDrive/faces --epochs 20 --batch-size 16 --amp
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from pipeline.model import build_model, default_device  # noqa: E402
from training.dataset import FaceCropDataset, eval_transform, train_transform  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            _, logit = model(x.to(device))
        scores.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.numpy())
    s, l = np.concatenate(scores), np.concatenate(labels)
    auc = roc_auc_score(l, s) if len(set(l.tolist())) == 2 else float("nan")
    acc = float(((s >= 0.5) == l).mean())
    return auc, acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(config.BASE_DIR, "data"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=4, help="early stopping patience (epochs)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU / Colab)")
    ap.add_argument("--out", default=config.HYBRID_WEIGHTS_PATH)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests)")
    # Architecture overrides exist only for quick smoke tests; defaults follow the paper.
    ap.add_argument("--cnn", default=config.CNN_BACKBONE)
    ap.add_argument("--vit", default=config.VIT_BACKBONE)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    ap.add_argument("--vit-size", type=int, default=config.VIT_INPUT_SIZE)
    ap.add_argument("--max-steps", type=int, default=None, help="limit steps per epoch (smoke tests)")
    args = ap.parse_args(argv)

    device = args.device or default_device()
    torch.manual_seed(42)
    train_ds = FaceCropDataset(args.data_root, "train", train_transform(args.image_size))
    val_ds = FaceCropDataset(args.data_root, "val", eval_transform(args.image_size))
    print(f"train {train_ds.class_counts()}  val {val_ds.class_counts()}  device={device}")
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=device == "cuda", drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers)

    arch = {"cnn_name": args.cnn, "vit_name": args.vit, "embedding_dim": config.EMBEDDING_DIM,
            "vit_input_size": args.vit_size}
    model, source = build_model(pretrained=not args.no_pretrained, cnn_name=args.cnn, vit_name=args.vit,
                                vit_input_size=args.vit_size)
    print(f"backbones: {source}")
    model.to(device)

    # Class-imbalance-aware BCE: pos_weight = #real / #fake
    counts = train_ds.class_counts()
    pos_weight = torch.tensor([counts["real"] / max(counts["fake"], 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    best_auc, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp and device == "cuda"):
                _, logit = model(x)
                loss = criterion(logit.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        auc, acc = evaluate(model, val_dl, device, args.amp)
        print(f"epoch {epoch:02d}  loss {np.mean(losses):.4f}  val_auc {auc:.4f}  val_acc {acc:.4f}  "
              f"({time.time() - t0:.0f}s)")
        score = -1.0 if np.isnan(auc) else auc
        if score > best_auc:
            best_auc, bad_epochs = score, 0
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "arch": arch, "epoch": epoch,
                        "val_auc": auc, "model_version": f"{config.MODEL_VERSION}-e{epoch}",
                        "backbones": source}, args.out)
            print(f"  saved best model -> {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: val AUC did not improve for {args.patience} epochs.")
                break
    print(f"Best val AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    main()
