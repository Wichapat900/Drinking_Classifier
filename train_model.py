"""
Train Drink Prediction Model
Processes real + synthetic accelerometer data → extracts features → trains classifier

Usage:
    python train_model.py

Output:
    model.pkl         - trained sklearn model
    scaler.pkl        - feature scaler
    feature_names.pkl - list of feature names (for debugging)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ─────────────────────────── Config ───────────────────────────
WINDOW_SIZE = 60       # samples per window (~1 second at 60Hz)
STEP_SIZE   = 30       # 50% overlap
LABEL_NAMES = {0: "Hot Coffee", 1: "Strawberry Smoothie", 2: "Bottle of Water"}

# ─────────────────────────── Feature extraction ───────────────
def extract_features(window: pd.DataFrame) -> dict:
    """Extract statistical features from a window of accelerometer data."""
    feats = {}
    for axis in ['x', 'y', 'z']:
        v = window[axis].values
        feats[f'{axis}_mean']   = np.mean(v)
        feats[f'{axis}_std']    = np.std(v)
        feats[f'{axis}_min']    = np.min(v)
        feats[f'{axis}_max']    = np.max(v)
        feats[f'{axis}_range']  = np.max(v) - np.min(v)
        feats[f'{axis}_iqr']    = np.percentile(v, 75) - np.percentile(v, 25)
        feats[f'{axis}_energy'] = np.sum(v**2) / len(v)
        feats[f'{axis}_mad']    = np.mean(np.abs(v - np.mean(v)))  # mean absolute deviation

    # Magnitude features
    mag = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    feats['mag_mean']  = np.mean(mag)
    feats['mag_std']   = np.std(mag)
    feats['mag_range'] = np.max(mag) - np.min(mag)
    feats['mag_energy']= np.sum(mag**2) / len(mag)

    # Cross-axis correlations
    feats['xy_corr'] = np.corrcoef(window['x'], window['y'])[0,1]
    feats['xz_corr'] = np.corrcoef(window['x'], window['z'])[0,1]
    feats['yz_corr'] = np.corrcoef(window['y'], window['z'])[0,1]

    # Jerk (rate of change) - captures sudden movements
    for axis in ['x', 'y', 'z']:
        jerk = np.diff(window[axis].values)
        feats[f'{axis}_jerk_std'] = np.std(jerk)
        feats[f'{axis}_jerk_max'] = np.max(np.abs(jerk))

    return feats


def session_to_windows(df: pd.DataFrame, label: int):
    """Slide a window across a session and extract features."""
    rows, labels = [], []
    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        rows.append(extract_features(window))
        labels.append(label)
    return rows, labels


# ─────────────────────────── Load real data ───────────────────
def load_real_data():
    real_files = [
        ("กาแฟ2.csv",   0),
        ("กาแฟ3.csv",   0),
        ("Str1.csv",    1),
        ("Str3.csv",    1),
        ("ด__ม3.csv",   2),
        ("ด__ม8.csv",   2),
    ]

    all_rows, all_labels = [], []
    loaded = 0
    for fname, label in real_files:
        path = Path(fname)
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        df = pd.read_csv(path)
        rows, labels = session_to_windows(df, label)
        all_rows.extend(rows)
        all_labels.extend(labels)
        loaded += 1
        print(f"  [ok] {fname} → {len(rows)} windows (label={LABEL_NAMES[label]})")

    return all_rows, all_labels, loaded


# ─────────────────────────── Load synthetic data ──────────────
def load_synthetic_data():
    path = Path("synthetic_data.csv")
    if not path.exists():
        print("  [skip] synthetic_data.csv not found — run generate_data.py first")
        return [], []

    df = pd.read_csv(path)
    all_rows, all_labels = [], []

    for session_id, group in df.groupby('session_id'):
        label = group['label'].iloc[0]
        rows, labels = session_to_windows(group[['x','y','z']], label)
        all_rows.extend(rows)
        all_labels.extend(labels)

    print(f"  [ok] synthetic_data.csv → {len(all_rows)} windows")
    return all_rows, all_labels


# ─────────────────────────── Main ─────────────────────────────
def main():
    print("=" * 50)
    print("  DRINK PREDICTOR — Model Training")
    print("=" * 50)

    print("\n[1] Loading real data...")
    real_rows, real_labels, n_real_files = load_real_data()

    print("\n[2] Loading synthetic data...")
    syn_rows, syn_labels = load_synthetic_data()

    all_rows   = real_rows   + syn_rows
    all_labels = real_labels + syn_labels

    if len(all_rows) == 0:
        print("\nERROR: No data found. Run generate_data.py first or place CSV files here.")
        return

    X = pd.DataFrame(all_rows).fillna(0).values
    y = np.array(all_labels)

    print(f"\n[3] Dataset summary:")
    print(f"   Total windows: {len(X)}")
    for cls, name in LABEL_NAMES.items():
        count = np.sum(y == cls)
        print(f"   Class {cls} ({name}): {count} windows")

    # Save feature names for inspection
    feature_names = list(pd.DataFrame(all_rows).columns)
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\n[4] Training Random Forest classifier ({X.shape[1]} features)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("\n[5] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"   CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    print("\n[6] Fitting final model on all data...")
    clf.fit(X_scaled, y)

    # Full dataset report
    y_pred = clf.predict(X_scaled)
    print("\n[7] Classification Report (train set):")
    print(classification_report(y, y_pred, target_names=list(LABEL_NAMES.values())))

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    print("\nTop 10 most important features:")
    print(importances.nlargest(10).to_string())

    print("\n[8] Saving model...")
    with open("model.pkl",  "wb") as f: pickle.dump(clf,    f)
    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    print("\n✓ Saved: model.pkl, scaler.pkl, feature_names.pkl")
    print("✓ Training complete! Run the web app next.")


if __name__ == "__main__":
    main()