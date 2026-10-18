"""Train a classifier on data/fingerspell_dataset/ (see
tools/collect_fingerspell.py), to replace fingerspell.py's hand-tuned rules
on the letters they're ambiguous on - A/S/T/M/N (near-fists), U/V/H (spread
and orientation), D/G/P/Q (orientation).

    python tools/train_fingerspell.py
    python tools/train_fingerspell.py --holdout avadh
    python tools/train_fingerspell.py --loo        # leave-one-person-out CV, best with 3+ people

**This only helps once there's data from SEVERAL DIFFERENT PEOPLE.** With
one person's data the "held-out" evaluation is really just held-out
LETTERS from the same hand - it will look great and prove nothing about
generalizing to anyone else, which is the entire point of this phase (see
docs/REWORK_ROADMAP.md). The script warns loudly when that's the situation
it's in; read the warning, don't just read the accuracy number.

With 2 people, --holdout <name> (or the default) is the whole story: train
on one, test on the other. With 3+, prefer --loo - it trains N models (one
per person held out), reports each person's accuracy plus the mean and
spread across all of them, which is a far more honest number than any
single arbitrary holdout: one person's accuracy alone tells you nothing
about how much accuracy varies hand to hand.

Saves the trained model to data/models/fingerspell_rf.joblib (gitignored)
+ a .meta.json describing what it was trained/evaluated on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sibling modules (skeleton, ...)

from skeleton import hand_feature

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fingerspell_dataset"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "fingerspell_rf.joblib"
META_PATH = MODEL_PATH.with_suffix("").with_suffix(".meta.json")


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Returns (X, y, people) - X is (N, 46) hand_feature vectors, y is the
    letter per row, people is the person_id per row (for a held-out split)."""
    X_parts, y_parts, people_parts = [], [], []
    if not DATA_DIR.exists():
        return None, None, None
    for person_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for npz_path in sorted(person_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            hands = d["hands"]              # (K, 21, 2)
            handed = d["handedness"]        # (K,)
            letter = str(d["letter"])
            feats = np.stack([
                hand_feature(h, mirror=str(lab).lower().startswith("l"))
                for h, lab in zip(hands, handed)
            ])
            X_parts.append(feats)
            y_parts.extend([letter] * len(feats))
            people_parts.extend([person_dir.name] * len(feats))
    if not X_parts:
        return None, None, None
    return np.concatenate(X_parts), np.array(y_parts), np.array(people_parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", help="person_id to hold out entirely for evaluation "
                     "(recommended with 2 people - the real generalization test)")
    ap.add_argument("--loo", action="store_true",
                     help="leave-one-person-out cross-validation across everyone collected "
                          "(recommended with 3+ people - see this script's docstring)")
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()
    if args.holdout and args.loo:
        raise SystemExit("--holdout and --loo are alternatives, pass one or the other")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
    except ImportError:
        raise SystemExit("needs scikit-learn + joblib: "
                          ".venv/bin/pip install scikit-learn joblib")

    X, y, people = load_dataset()
    if X is None:
        raise SystemExit(f"no data in {DATA_DIR} - run tools/collect_fingerspell.py first")

    unique_people = sorted(set(people.tolist()))
    unique_letters = sorted(set(y.tolist()))
    print(f"loaded {len(X)} samples, {len(unique_letters)} letters, "
          f"{len(unique_people)} people: {unique_people}")

    if len(unique_people) < 2:
        print("\n*** WARNING: only one person's data found. ***")
        print("A model trained and evaluated on one person's hand cannot show it ")
        print("generalizes to anyone else - that's the exact bug this phase exists ")
        print("to fix. The accuracy number below is real, but it is not the result ")
        print("that matters; collect at least one more person before trusting this ")
        print("model over the rule-based classifier. See docs/REWORK_ROADMAP.md.\n")

    eval_meta: dict = {}

    if args.loo:
        if len(unique_people) < 2:
            raise SystemExit("--loo needs at least 2 people's data")
        print(f"\nleave-one-person-out across {len(unique_people)} people:")
        per_person = {}
        for person in unique_people:
            train_mask = people != person
            test_mask = ~train_mask
            fold_clf = RandomForestClassifier(
                n_estimators=args.n_estimators, random_state=0, n_jobs=-1)
            fold_clf.fit(X[train_mask], y[train_mask])
            fold_acc = float(accuracy_score(y[test_mask], fold_clf.predict(X[test_mask])))
            per_person[person] = fold_acc
            print(f"  held out {person:20s} accuracy {fold_acc:.3f}  "
                  f"({int(test_mask.sum())} samples)")
        accs = np.array(list(per_person.values()))
        acc = float(accs.mean())
        print(f"\nmean accuracy: {acc:.3f}   std: {float(accs.std()):.3f}   "
              f"min: {float(accs.min()):.3f}   max: {float(accs.max()):.3f}")
        print("(a wide spread means the model is much better on some hands than "
              "others - worth more people, or a look at which person's accuracy "
              "is lowest and why)")
        eval_meta = {
            "eval_mode": "leave_one_person_out",
            "per_person_accuracy": per_person,
            "mean_accuracy": acc,
            "std_accuracy": float(accs.std()),
        }
    else:
        if args.holdout:
            if args.holdout not in unique_people:
                raise SystemExit(f"no data for person {args.holdout!r}; have {unique_people}")
            holdout_person = args.holdout
        elif len(unique_people) >= 2:
            holdout_person = unique_people[-1]
            print(f"(no --holdout/--loo given, using the last person alphabetically: "
                  f"{holdout_person} - with 3+ people, --loo is the better test)")
        else:
            holdout_person = None

        if holdout_person:
            train_mask = people != holdout_person
            test_mask = ~train_mask
            split_desc = f'held out "{holdout_person}" entirely (genuine generalization test)'
        else:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(X))
            split = int(len(X) * 0.8)
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[idx[:split]] = True
            test_mask = ~train_mask
            split_desc = "random 80/20 split WITHIN one person (not a generalization test)"

        clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0, n_jobs=-1)
        clf.fit(X[train_mask], y[train_mask])
        pred = clf.predict(X[test_mask])
        acc = float(accuracy_score(y[test_mask], pred))

        print(f"\nsplit: {split_desc}")
        print(f"accuracy: {acc:.3f}  ({int(test_mask.sum())} test samples)\n")
        print(classification_report(y[test_mask], pred, zero_division=0))
        eval_meta = {
            "eval_mode": "holdout",
            "holdout_person": holdout_person,
            "holdout_accuracy": acc,
            "generalization_test": holdout_person is not None and len(unique_people) >= 2,
        }

    # the deployed model is trained on everything collected so far
    clf_final = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0, n_jobs=-1)
    clf_final.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_final, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "feature_dim": int(X.shape[1]),
        "letters": unique_letters,
        "people": unique_people,
        "n_samples": int(len(X)),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **eval_meta,
    }, indent=2))
    print(f"\nsaved model -> {MODEL_PATH}")
    print(f"saved metadata -> {META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
