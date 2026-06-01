"""
train.py
========
Loads all CSVs, extracts time-series features from sliding windows,
trains XGBoost + Random Forest, evaluates on a held-out test set.

Split strategy: FILE-LEVEL 80/20 split (not window-level).
  Windows from the same recording stay together in train OR test —
  this prevents data leakage from overlapping sliding windows.

Usage:
    python train.py

Output:
    models/model.pkl    — best trained model (+ scaler inside pipeline)
    models/report.json  — full metrics: CV, test accuracy, confusion matrix
"""

import json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────
FS        = 60      # Hz
WIN_SEC   = 3.0
STEP_SEC  = 1.0
WIN_N     = int(WIN_SEC  * FS)   # 180 samples
STEP_N    = int(STEP_SEC * FS)   # 60 samples
TEST_SIZE = 0.20                 # 20% of FILES held out as test set
RANDOM_STATE = 42

LABEL_NAMES = ["Coffee", "Strawberry", "Water"]
CLASS_DIRS  = {
    "Coffee":     ("data/Coffee",     0),
    "Strawberry": ("data/Strawberry", 1),
    "Water":      ("data/Water",      2),
}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def window_features(seg: np.ndarray) -> np.ndarray:
    """55 features from a (WIN_N, 3) window — must match app.py exactly."""
    mag = np.sqrt((seg**2).sum(axis=1))
    feats = []
    for ch in [seg[:,0], seg[:,1], seg[:,2], mag]:
        feats += [
            ch.mean(), ch.std(), ch.min(), ch.max(),
            ch.max() - ch.min(),
            float(skew(ch)), float(kurtosis(ch)),
            np.percentile(ch, 25), np.percentile(ch, 75),
            np.sqrt(np.mean(ch**2)),
            np.mean(np.abs(np.diff(ch))),
        ]
        freqs, psd = welch(ch, fs=FS, nperseg=min(len(ch), WIN_N//2))
        feats.append(float(freqs[np.argmax(psd)]))
        feats.append(float(np.sum(psd)))
    feats.append(float(np.corrcoef(seg[:,0], seg[:,1])[0,1]))
    feats.append(float(np.corrcoef(seg[:,1], seg[:,2])[0,1]))
    feats.append(float(np.corrcoef(seg[:,0], seg[:,2])[0,1]))
    return np.array(feats, dtype=np.float32)


def file_to_windows(csv_path: Path, label: int):
    """Load one CSV → list of (feature_vector, label)."""
    df  = pd.read_csv(csv_path)
    if not all(c in df.columns for c in ["x","y","z"]):
        return []
    sig = df[["x","y","z"]].values.astype(np.float32)
    out = []
    start = 0
    while start + WIN_N <= len(sig):
        out.append((window_features(sig[start:start+WIN_N]), label))
        start += STEP_N
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FILE-LEVEL TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def file_level_split(class_dirs, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """
    Split at the FILE level per class so no recording spans both sets.
    Returns: train_files, test_files — each a list of (path, label) tuples.
    """
    rng = np.random.default_rng(seed)
    train_files, test_files = [], []

    for name, (folder, label) in class_dirs.items():
        files = sorted(Path(folder).glob("*.csv"))
        if not files:
            print(f"  ⚠  No files in {folder}")
            continue
        files = list(files)
        rng.shuffle(files)
        n_test  = max(1, round(len(files) * test_size))
        n_train = len(files) - n_test
        train_files += [(f, label) for f in files[:n_train]]
        test_files  += [(f, label) for f in files[n_train:]]
        print(f"  {name:12s}: {len(files)} files → "
              f"{n_train} train / {n_test} test")

    return train_files, test_files


def files_to_arrays(file_list):
    """Convert list of (path, label) → X, y arrays."""
    X, y = [], []
    for path, label in file_list:
        windows = file_to_windows(path, label)
        for feat, lbl in windows:
            X.append(feat); y.append(lbl)
    return np.array(X, dtype=np.float32), np.array(y)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n═══════════════════════════════════════════════════════════")
    print("  DrinkSense — Model Training")
    print("═══════════════════════════════════════════════════════════\n")

    # ── File-level train / test split ──────────────────────────────────────
    print("── File-level 80/20 split ──────────────────────────────────")
    train_files, test_files = file_level_split(CLASS_DIRS)

    print(f"\nBuilding feature arrays…")
    X_train, y_train = files_to_arrays(train_files)
    X_test,  y_test  = files_to_arrays(test_files)

    print(f"\n  Train: {len(X_train)} windows  "
          f"(Coffee={( y_train==0).sum()}, "
          f"Strawberry={(y_train==1).sum()}, "
          f"Water={(y_train==2).sum()})")
    print(f"  Test:  {len(X_test)} windows  "
          f"(Coffee={( y_test==0).sum()}, "
          f"Strawberry={(y_test==1).sum()}, "
          f"Water={(y_test==2).sum()})")

    # ── Models wrapped in scaler pipeline ──────────────────────────────────
    # Scaler is fit ONLY on X_train — no leakage into test set.
    def make_pipeline(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    candidates = {
        "RandomForest": make_pipeline(RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
        "GradientBoosting": make_pipeline(GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE
        )),
    }
    if XGB_AVAILABLE:
        candidates["XGBoost"] = make_pipeline(xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
        ))

    # ── Cross-validation on TRAIN set only ─────────────────────────────────
    print("\n── 5-fold CV on training set ───────────────────────────────")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_cv, best_pipe = None, 0.0, None

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)
        mu, sd = scores.mean(), scores.std()
        marker = " ◄ best" if mu > best_cv else ""
        print(f"  {name:20s}: {mu*100:.1f}% ± {sd*100:.1f}%{marker}")
        if mu > best_cv:
            best_cv, best_name, best_pipe = mu, name, pipe

    print(f"\n  ✓ Selected: {best_name}  (CV {best_cv*100:.1f}%)")

    # ── Retrain best model on full training set ─────────────────────────────
    print("\n── Training final model on full train set ──────────────────")
    best_pipe.fit(X_train, y_train)

    # ── Evaluate on HELD-OUT test set ──────────────────────────────────────
    print("\n── Test set evaluation (held-out, never seen during CV) ────")
    y_pred_test = best_pipe.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1  = f1_score(y_test, y_pred_test, average="macro")
    cm       = confusion_matrix(y_test, y_pred_test)

    print(f"\n  Test Accuracy : {test_acc*100:.1f}%")
    print(f"  Test F1 (macro): {test_f1*100:.1f}%")
    print(f"\n{classification_report(y_test, y_pred_test, target_names=LABEL_NAMES)}")
    print("  Confusion matrix (rows=actual, cols=predicted):")
    header = "           " + "  ".join(f"{n:>10}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:10s}" + "  ".join(f"{v:>10d}" for v in row))

    # Also show train accuracy to check for overfitting
    y_pred_train = best_pipe.predict(X_train)
    train_acc    = accuracy_score(y_train, y_pred_train)
    overfit_gap  = train_acc - test_acc
    print(f"\n  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Test  accuracy : {test_acc*100:.1f}%")
    if overfit_gap > 0.10:
        print(f"  ⚠  Overfit gap  : {overfit_gap*100:.1f}%  — consider more real data")
    else:
        print(f"  ✓  Overfit gap  : {overfit_gap*100:.1f}%  — looks healthy")

    # ── Save ───────────────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)

    # Save the full pipeline (scaler + model together — no separate scaler needed)
    joblib.dump(best_pipe, "models/model.pkl")

    # Also save scaler separately so app.py can use it independently
    joblib.dump(best_pipe.named_steps["scaler"], "models/scaler.pkl")

    cm_list = cm.tolist()
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        per_class[name] = {"precision": round(prec,4), "recall": round(rec,4)}

    report = {
        "model":          best_name,
        "cv_accuracy":    round(best_cv,   4),
        "test_accuracy":  round(test_acc,  4),
        "test_f1_macro":  round(test_f1,   4),
        "train_accuracy": round(train_acc, 4),
        "overfit_gap":    round(overfit_gap, 4),
        "per_class":      per_class,
        "confusion_matrix": cm_list,
        "labels":         LABEL_NAMES,
        "n_train_windows": int(len(X_train)),
        "n_test_windows":  int(len(X_test)),
        "feature_dim":    int(X_train.shape[1]),
        "win_sec":        WIN_SEC,
        "step_sec":       STEP_SEC,
        "fs":             FS,
        "win_n":          WIN_N,
        "step_n":         STEP_N,
        "split":          "file-level 80/20",
    }
    with open("models/report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Saved:")
    print(f"   models/model.pkl    ({best_name} pipeline: scaler + classifier)")
    print(f"   models/scaler.pkl   (standalone scaler)")
    print(f"   models/report.json  (CV: {best_cv*100:.1f}%  |  Test: {test_acc*100:.1f}%)")


if __name__ == "__main__":
    main()