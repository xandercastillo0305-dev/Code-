"""Build a pairs.csv (suspect, reference, label) for the similarity method.

The paper's pipeline compares a suspect image with a reference image of the
same subject, so calibrating tau and evaluating the full pipeline needs PAIRS.
This helper pairs every crop in <split>/real and <split>/fake with a
*different* real crop of the same identity.

The identity is taken from the file name with --identity-regex (first capture
group). Defaults match crops produced by extract_faces.py from a folder of
frames named <video>__<frame>.png, where <video> starts with the identity
(e.g. Celeb-DF "id0_0001" / "id0_id16_0001": identity "id0").
Check the pairs manually; the correct identity key depends on your dataset.

    python -m training.build_pairs --split-dir data/val
"""
import argparse
import csv
import os
import random
import re
from collections import defaultdict

EXTS = (".jpg", ".jpeg", ".png")


def build_pairs(split_dir, identity_regex=r"^(id\d+)", seed=0):
    rx = re.compile(identity_regex)
    rng = random.Random(seed)

    def files(cls):
        d = os.path.join(split_dir, cls)
        return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS)) if os.path.isdir(d) else []

    reals_by_id = defaultdict(list)
    for f in files("real"):
        m = rx.search(f)
        if m:
            reals_by_id[m.group(1)].append(f"real/{f}")
    rows = []
    for cls in ("real", "fake"):
        for f in files(cls):
            m = rx.search(f)
            if not m:
                continue
            candidates = [r for r in reals_by_id.get(m.group(1), []) if r != f"{cls}/{f}"]
            if candidates:
                rows.append({"suspect": f"{cls}/{f}", "reference": rng.choice(candidates), "label": cls})
    out = os.path.join(split_dir, "pairs.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["suspect", "reference", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} pairs to {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--identity-regex", default=r"^(id\d+)")
    a = ap.parse_args()
    build_pairs(a.split_dir, a.identity_regex)
