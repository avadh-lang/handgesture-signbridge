"""Compare the rule-based classifier vs. the trained model on
data/fingerspell_dataset/ - the "before vs. after" number for the paper.

    python tools/eval_fingerspell.py
    python tools/eval_fingerspell.py --holdout avadh

Unlike tools/train_fingerspell.py (which trains on everyone except
--holdout, then scores on --holdout), this always evaluates BOTH
classifiers on the exact same held-out samples, so the comparison is
apples-to-apples. Needs a trained model to exist already (run
tools/train_fingerspell.py first) and at least 2 people's data for the
comparison to mean anything - see its warning for why.

Writes data/eval/fingerspell_comparison.json (gitignored) with both
classifiers' accuracy + confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sibling modules

from fingerspell import classify_letter, classify_letter_ml, ml_model_info

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fingerspell_dataset"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "fingerspell_comparison.json"


def load_raw() -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    """Returns (hands, handedness, letters, people) - kept as raw landmarks
    (not hand_feature vectors) so both classifiers can be run on the exact
    same samples: classify_letter() wants raw landmarks, classify_letter_ml
    computes hand_feature() internally from the same raw input."""
    hands, handed, letters, people = [], [], [], []
    for person_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for npz_path in sorted(person_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            for h, lab in zip(d["hands"], d["handedness"]):
                hands.append(h)
                handed.append(str(lab))
                letters.append(str(d["letter"]))
                people.append(person_dir.name)
    return hands, handed, letters, people


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", help="only evaluate on this person's samples "
                     "(should match what you trained with --holdout)")
    args = ap.parse_args()

    if ml_model_info() is None:
        raise SystemExit("no trained model found - run tools/train_fingerspell.py first")

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise SystemExit(f"no data in {DATA_DIR} - run tools/collect_fingerspell.py first")

    hands, handed, letters, people = load_raw()
    unique_people = sorted(set(people))
    if args.holdout:
        if args.holdout not in unique_people:
            raise SystemExit(f"no data for person {args.holdout!r}; have {unique_people}")
        keep = [i for i, p in enumerate(people) if p == args.holdout]
    elif len(unique_people) >= 2:
        args.holdout = unique_people[-1]
        keep = [i for i, p in enumerate(people) if p == args.holdout]
        print(f"(no --holdout given, using {args.holdout!r})")
    else:
        keep = list(range(len(hands)))
        print("WARNING: only one person's data - this is NOT a generalization test, "
              "see tools/train_fingerspell.py's warning.")

    rule_correct = ml_correct = 0
    rule_confusion: Counter = Counter()
    ml_confusion: Counter = Counter()

    for i in keep:
        hand, lab, truth = hands[i], handed[i], letters[i]
        rule_pred = classify_letter(hand, lab).letter
        ml_result = classify_letter_ml(hand, lab)
        ml_pred = ml_result.letter if ml_result else None

        rule_correct += rule_pred == truth
        if ml_pred is not None:
            ml_correct += ml_pred == truth
        if rule_pred != truth:
            rule_confusion[f"{truth}->{rule_pred}"] += 1
        if ml_pred is not None and ml_pred != truth:
            ml_confusion[f"{truth}->{ml_pred}"] += 1

    n = len(keep)
    rule_acc = rule_correct / n if n else 0.0
    ml_acc = ml_correct / n if n else 0.0

    print(f"\nevaluated on {n} samples" + (f' from "{args.holdout}"' if args.holdout else ""))
    print(f"  rule-based (classify_letter):    {rule_acc:.3f}  ({rule_correct}/{n})")
    print(f"  trained model (classify_letter_ml): {ml_acc:.3f}  ({ml_correct}/{n})")
    print(f"\ntop rule-based confusions: {rule_confusion.most_common(8)}")
    print(f"top ML confusions: {ml_confusion.most_common(8)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_samples": n,
        "holdout_person": args.holdout,
        "is_generalization_test": bool(args.holdout) and len(unique_people) >= 2,
        "rule_based_accuracy": rule_acc,
        "ml_accuracy": ml_acc,
        "rule_based_confusions": rule_confusion.most_common(20),
        "ml_confusions": ml_confusion.most_common(20),
        "model_info": ml_model_info(),
    }, indent=2))
    print(f"\nsaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
