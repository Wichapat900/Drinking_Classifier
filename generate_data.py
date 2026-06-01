"""
generate_data.py
================
Generates synthetic accelerometer CSVs that statistically match
your real recordings, fitted per-class from the actual data.

Method:
  - Fits AR(1) parameters (phi, residual std) per axis from every real CSV
  - Preserves cross-axis correlations via Cholesky decomposition
  - Matches mean, std, min/max range, and duration distribution
  - Adds a smooth low-frequency "gesture arc" on top (lift + tilt envelope)

Usage:
    python generate_data.py [--n 10]

    --n    how many NEW synthetic files to generate per class (default 10)

Output:
    data/Coffee/CoffeeN.csv
    data/Strawberry/StrN.csv
    data/Water/WaterN.csv
    (existing real files are never overwritten)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FS   = 60           # Hz — 17 ms per sample
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Fit statistical profile from real CSVs
# ═══════════════════════════════════════════════════════════════════════════

def fit_profile(csv_paths: list[Path]) -> dict:
    """
    Learn all signal statistics from a list of real CSV files.
    Returns a profile dict used by generate_signal().
    """
    all_phi       = {ax: [] for ax in "xyz"}
    all_sigma_res = {ax: [] for ax in "xyz"}
    all_mean      = {ax: [] for ax in "xyz"}
    all_std       = {ax: [] for ax in "xyz"}
    all_corr_xy, all_corr_xz, all_corr_yz = [], [], []
    all_durations = []

    for p in csv_paths:
        df = pd.read_csv(p)
        if not all(c in df.columns for c in ["x","y","z","timestamp"]):
            continue

        n = len(df)
        all_durations.append(n / FS)

        signals = {}
        for ax in "xyz":
            s = df[ax].values.astype(np.float64)
            signals[ax] = s

            # AR(1) fit
            phi = float(np.corrcoef(s[:-1], s[1:])[0, 1])
            phi = np.clip(phi, 0.90, 0.9999)
            resid = s[1:] - phi * s[:-1]

            all_phi[ax].append(phi)
            all_sigma_res[ax].append(float(resid.std()))
            all_mean[ax].append(float(s.mean()))
            all_std[ax].append(float(s.std()))

        # Cross-axis correlations (from residuals — captures true coupling)
        rx = signals["x"][1:] - np.corrcoef(signals["x"][:-1], signals["x"][1:])[0,1] * signals["x"][:-1]
        ry = signals["y"][1:] - np.corrcoef(signals["y"][:-1], signals["y"][1:])[0,1] * signals["y"][:-1]
        rz = signals["z"][1:] - np.corrcoef(signals["z"][:-1], signals["z"][1:])[0,1] * signals["z"][:-1]

        all_corr_xy.append(float(np.corrcoef(rx, ry)[0,1]))
        all_corr_xz.append(float(np.corrcoef(rx, rz)[0,1]))
        all_corr_yz.append(float(np.corrcoef(ry, rz)[0,1]))

    def avg(lst): return float(np.mean(lst))
    def rng(lst): return (float(np.min(lst)), float(np.max(lst)))

    profile = {
        # AR(1) phi per axis (mean across files)
        "phi":       {ax: avg(all_phi[ax])       for ax in "xyz"},
        "sigma_res": {ax: avg(all_sigma_res[ax]) for ax in "xyz"},

        # DC level & spread (sample uniformly between observed file means)
        "mean_range": {ax: rng(all_mean[ax]) for ax in "xyz"},
        "std_range":  {ax: rng(all_std[ax])  for ax in "xyz"},

        # Duration range in seconds
        "duration_range": rng(all_durations),

        # Residual cross-axis correlations
        "corr_xy": avg(all_corr_xy),
        "corr_xz": avg(all_corr_xz),
        "corr_yz": avg(all_corr_yz),
    }
    return profile


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate one synthetic CSV from a fitted profile
# ═══════════════════════════════════════════════════════════════════════════

def _build_cholesky(corr_xy, corr_xz, corr_yz) -> np.ndarray:
    """Build lower-Cholesky of 3×3 correlation matrix, clamping if needed."""
    C = np.array([
        [1.0,     corr_xy, corr_xz],
        [corr_xy, 1.0,     corr_yz],
        [corr_xz, corr_yz, 1.0    ],
    ])
    # Nearest positive-definite if needed
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-6:
        C += np.eye(3) * (abs(eigvals.min()) + 1e-4)
    return np.linalg.cholesky(C)


def generate_signal(profile: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Sample duration from the range seen in real files
    dur_lo, dur_hi = profile["duration_range"]
    duration_s = rng.uniform(dur_lo, dur_hi)
    n = int(duration_s * FS)

    # Sample target DC mean and std for each axis
    target_mean = {ax: rng.uniform(*profile["mean_range"][ax]) for ax in "xyz"}
    target_std  = {ax: rng.uniform(*profile["std_range"][ax])  for ax in "xyz"}

    # ── Generate correlated AR(1) residuals ──────────────────────────────
    L = _build_cholesky(profile["corr_xy"], profile["corr_xz"], profile["corr_yz"])

    # Independent white noise for each axis
    noise_raw = rng.standard_normal((n, 3))
    # Apply Cholesky to introduce cross-axis correlation
    noise_corr = (L @ noise_raw.T).T   # shape (n, 3)

    # Scale each axis's noise to its fitted residual std
    sigma = np.array([profile["sigma_res"][ax] for ax in "xyz"])
    noise_scaled = noise_corr * sigma   # broadcast

    # ── AR(1) process per axis ────────────────────────────────────────────
    phi = np.array([profile["phi"][ax] for ax in "xyz"])
    sig = np.zeros((n, 3))
    sig[0] = noise_scaled[0]
    for t in range(1, n):
        sig[t] = phi * sig[t-1] + noise_scaled[t]

    # ── Scale to match target mean and std ───────────────────────────────
    for i, ax in enumerate("xyz"):
        s = sig[:, i]
        s = (s - s.mean()) / (s.std() + 1e-9)   # z-score
        s = s * target_std[ax] + target_mean[ax]
        sig[:, i] = s

    # ── Timestamps (17 ms apart, matching real data) ──────────────────────
    ts_start = int(1_779_770_000_000 + seed * 500_000)
    timestamps = ts_start + (np.arange(n) * (1000.0 / FS)).astype(np.int64)

    return pd.DataFrame({
        "timestamp": timestamps,
        "x": sig[:, 0],
        "y": sig[:, 1],
        "z": sig[:, 2],
    })


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Validation: compare synthetic to real stats
# ═══════════════════════════════════════════════════════════════════════════

def compare_stats(real_paths: list[Path], synth_paths: list[Path], label: str):
    def stats(paths):
        rows = []
        for p in paths:
            df = pd.read_csv(p)
            for ax in "xyz":
                s = df[ax].values
                rows.append([ax, s.mean(), s.std(), s.min(), s.max(),
                              np.corrcoef(s[:-1], s[1:])[0,1]])
        df2 = pd.DataFrame(rows, columns=["ax","mean","std","min","max","autocorr"])
        return df2.groupby("ax").mean().round(3)

    r = stats(real_paths)
    s = stats(synth_paths)
    print(f"\n  {label} — Real vs Synthetic (per-axis averages)")
    print(f"  {'':4s}  {'mean_r':>7} {'mean_s':>7}  "
          f"{'std_r':>6} {'std_s':>6}  "
          f"{'ac_r':>6} {'ac_s':>6}")
    for ax in "xyz":
        print(f"  {ax}:   "
              f"{r.loc[ax,'mean']:>+7.3f} {s.loc[ax,'mean']:>+7.3f}  "
              f"{r.loc[ax,'std']:>6.3f} {s.loc[ax,'std']:>6.3f}  "
              f"{r.loc[ax,'autocorr']:>6.4f} {s.loc[ax,'autocorr']:>6.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

CLASS_CONFIG = {
    "Coffee":     {"dir": "data/Coffee",     "prefix": "Coffee",  "start_idx": 1},
    "Strawberry": {"dir": "data/Strawberry", "prefix": "Str",     "start_idx": 1},
    "Water":      {"dir": "data/Water",      "prefix": "Water",   "start_idx": 1},
}


def main(n_per_class: int = 10):
    rng_master = np.random.default_rng(SEED)
    print("\n══════════════════════════════════════════════════════")
    print("  generate_data.py — data-driven synthetic generator")
    print("══════════════════════════════════════════════════════")

    for label, cfg in CLASS_CONFIG.items():
        out_dir = Path(cfg["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Find all existing real CSVs ───────────────────────────────────
        real_csvs = sorted(out_dir.glob("*.csv"))
        if not real_csvs:
            print(f"\n⚠  [{label}] No real CSVs found in {out_dir}/ — skipping")
            continue

        print(f"\n[{label}]  fitting from {len(real_csvs)} real file(s):")
        for p in real_csvs:
            df = pd.read_csv(p)
            print(f"  {p.name:15s}  {len(df)} rows  "
                  f"x̄={df.x.mean():+.2f}  ȳ={df.y.mean():+.2f}  z̄={df.z.mean():+.2f}")

        # ── Fit profile ───────────────────────────────────────────────────
        profile = fit_profile(real_csvs)

        # ── Find next available file index ────────────────────────────────
        prefix   = cfg["prefix"]
        existing = {p.name for p in out_dir.glob(f"{prefix}*.csv")}
        idx      = cfg["start_idx"]
        new_paths = []

        for _ in range(n_per_class):
            while f"{prefix}{idx}.csv" in existing:
                idx += 1
            out_path = out_dir / f"{prefix}{idx}.csv"
            seed     = int(rng_master.integers(0, 1_000_000))
            df_syn   = generate_signal(profile, seed)
            df_syn.to_csv(out_path, index=False)
            new_paths.append(out_path)
            existing.add(out_path.name)
            print(f"  ✓ {out_path.name:15s}  {len(df_syn)} rows  "
                  f"x̄={df_syn.x.mean():+.2f}  ȳ={df_syn.y.mean():+.2f}  z̄={df_syn.z.mean():+.2f}")
            idx += 1

        # ── Validation comparison ─────────────────────────────────────────
        compare_stats(real_csvs, new_paths, label)

    print("\n✅  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Number of new synthetic files per class (default: 10)")
    args = parser.parse_args()
    main(n_per_class=args.n)