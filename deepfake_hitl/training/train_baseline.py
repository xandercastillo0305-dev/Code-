"""XceptionNet baseline for comparison (Section 3.7).

XceptionNet is the standard FaceForensics++ baseline. In timm >= 0.9 the
original Xception is registered as ``legacy_xception`` (299x299 input).

    python -m training.train_baseline --data-root data --epochs 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from pipeline.model import default_device  # noqa: E402
from training.dataset import FaceCropDataset, eval_transform, train_transform  # noqa: E402

XCEPTION_NAME = "legacy_xception"
XCEPTION_SIZE = 299


class XceptionBaseline(nn.Module):
    """Returns (None, logit) so it shares the evaluation code with the hybrid model."""

    def __init__(self, name=XCEPTION_NAME, pretrained=True):
        super().__init__()
        import timm
        try:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=1)
            self.pretrained = pretrained
        except Exception as exc:
            print(f"Could not load pretrained {name} ({exc}); using random init.")
            self.net = timm.create_model(name, pretrained=False, num_classes=1)
            self.pretrained = False

    def forward(self, x):
        return None, self.net(x).squeeze(1)


def load_baseline(path=config.BASELINE_WEIGHTS_PATH, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = XceptionBaseline(ckpt.get("name", XCEPTION_NAME), pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    s, l = [], []
    for x, y in loader:
        s.append(torch.sigmoid(model(x.to(device))[1]).cpu().numpy())
        l.append(y.numpy())
    s, l = np.concatenate(s), np.concatenate(l)
    return roc_auc_score(l, s) if len(set(l.tolist())) == 2 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.path.join(config.BASE_DIR, "data"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=config.BASELINE_WEIGHTS_PATH)
    args = ap.parse_args(argv)
    device = args.device or default_device()
    torch.manual_seed(42)

    train_ds = FaceCropDataset(args.data_root, "train", train_transform(XCEPTION_SIZE))
    val_ds = FaceCropDataset(args.data_root, "val", eval_transform(XCEPTION_SIZE))
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers)
    model = XceptionBaseline(pretrained=not args.no_pretrained).to(device)
    counts = train_ds.class_counts()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts["real"] / max(counts["fake"], 1)],
                                                             device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best, bad = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            opt.zero_grad()
            loss = criterion(model(x.to(device))[1], y.to(device))
            loss.backward()
            opt.step()
        auc = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d}  val_auc {auc:.4f}")
        score = -1.0 if np.isnan(auc) else auc
        if score > best:
            best, bad = score, 0
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "name": XCEPTION_NAME, "val_auc": auc}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                break
    print(f"Best val AUC: {best:.4f} -> {args.out}")
    return best


if __name__ == "__main__":
    main()
