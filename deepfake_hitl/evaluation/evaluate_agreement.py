"""AI-vs-human agreement from the case store (Section 3.7).

    python -m evaluation.evaluate_agreement [--cases data/cases.json]

Definitions used:
* Reviewed cases  = status verified or flagged.
* Decisive cases  = Confirm or Override (the analyst gave Real/Deepfake).
* Agreement rate  = Confirm / (Confirm + Override)  over decisive cases.
* Cohen's kappa   = between ai_classification and final_classification over
                    decisive cases. Flagged cases are "Inconclusive" (not a
                    class the AI can output), so they are reported as a
                    count and excluded from agreement and kappa.
* Strict agreement (secondary) counts flags as disagreement.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from evaluation.metrics import agreement_rate, cohens_kappa, cohens_kappa_manual  # noqa: E402


def agreement_summary(cases):
    reviewed = [c for c in cases if c["status"] in ("verified", "flagged")]
    decisive = [c for c in reviewed if c["review_decision"] in ("Confirm", "Override")]
    counts = {d: sum(c["review_decision"] == d for c in reviewed) for d in ("Confirm", "Override", "Flag")}
    ai = [c["ai_classification"] for c in decisive]
    human = [c["final_classification"] for c in decisive]
    kappa = cohens_kappa(ai, human) if decisive else float("nan")
    return {
        "total_cases": len(cases),
        "pending": sum(c["status"] == "pending" for c in cases),
        "verified": sum(c["status"] == "verified" for c in cases),
        "flagged": sum(c["status"] == "flagged" for c in cases),
        "reviewed": len(reviewed),
        "decisive": len(decisive),
        "confirm": counts["Confirm"],
        "override": counts["Override"],
        "flag": counts["Flag"],
        "agreement_rate": agreement_rate(ai, human),
        "strict_agreement_rate": (counts["Confirm"] / len(reviewed)) if reviewed else None,
        "cohens_kappa": None if math.isnan(kappa) else kappa,
        "cohens_kappa_manual": None if not decisive or math.isnan(cohens_kappa_manual(ai, human))
        else cohens_kappa_manual(ai, human),
    }


def _fmt(v):
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=config.CASES_PATH)
    args = ap.parse_args(argv)
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    s = agreement_summary(cases)
    print("AI-Human Agreement (Section 3.7)")
    print("-" * 40)
    for k, v in s.items():
        print(f"{k:<22}{_fmt(v)}")
    return s


if __name__ == "__main__":
    main()
