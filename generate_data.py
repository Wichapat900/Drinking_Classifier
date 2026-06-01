"""
Synthetic Data Generator for Drink Prediction
Generates realistic accelerometer data based on patterns from real recordings.

Usage: python generate_data.py
Output: synthetic_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_coffee_session(n_samples=600, session_id=0):
    """
    Hot Coffee: ~45-degree tilt, steady sipping motion.
    - Phone tilted: Y around -7, X around -5, Z around -3
    - Gentle periodic lift-and-sip movement
    - Low magnitude variance (smooth motion)
    """
    t = np.linspace(0, n_samples / 60, n_samples)  # ~60Hz

    # Base orientation: phone tilted ~45 degrees
    x_base = -5.0 + np.random.normal(0, 0.3)
    y_base = -7.0 + np.random.normal(0, 0.3)
    z_base = -3.0 + np.random.normal(0, 0.2)

    # Slow sipping oscillation (lift cup up, drink, put down)
    sip_freq = 0.15 + np.random.uniform(-0.05, 0.05)  # ~every 6-7 seconds
    sip_amp_x = 2.5 + np.random.uniform(-0.5, 0.5)
    sip_amp_y = 1.5 + np.random.uniform(-0.3, 0.3)

    x = x_base + sip_amp_x * np.sin(2 * np.pi * sip_freq * t) + np.random.normal(0, 0.25, n_samples)
    y = y_base + sip_amp_y * np.cos(2 * np.pi * sip_freq * t) + np.random.normal(0, 0.25, n_samples)
    z = z_base + 0.3 * np.sin(2 * np.pi * sip_freq * t + 0.5) + np.random.normal(0, 0.15, n_samples)

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': 0})  # 0 = Coffee


def generate_smoothie_session(n_samples=420, session_id=0):
    """
    Strawberry Smoothie: upright/vertical hold with straw sipping.
    - Phone nearly vertical: Y very negative ~-9, X moderate, Z near 0
    - Very steady hold (drinking through straw = minimal tilt)
    - Lowest magnitude variance of all three
    """
    t = np.linspace(0, n_samples / 60, n_samples)

    x_base = -3.5 + np.random.normal(0, 0.3)
    y_base = -9.0 + np.random.normal(0, 0.2)
    z_base = 0.0 + np.random.normal(0, 0.2)

    # Very gentle, minimal movement (straw drinking = cup barely moves)
    micro_freq = 0.08 + np.random.uniform(-0.02, 0.02)
    x = x_base + 0.6 * np.sin(2 * np.pi * micro_freq * t) + np.random.normal(0, 0.15, n_samples)
    y = y_base + 0.4 * np.cos(2 * np.pi * micro_freq * t) + np.random.normal(0, 0.12, n_samples)
    z = z_base + 0.3 * np.sin(2 * np.pi * micro_freq * t + 1.0) + np.random.normal(0, 0.12, n_samples)

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': 1})  # 1 = Smoothie


def generate_water_session(n_samples=500, session_id=0):
    """
    Bottle of Water: most dynamic movement.
    - Wide range of orientations (tilting bottle high to drink)
    - High variance on all axes, large magnitude spikes
    - Picking up heavy bottle = more inertial forces
    """
    t = np.linspace(0, n_samples / 60, n_samples)

    x_base = -4.8 + np.random.normal(0, 0.5)
    y_base = -6.5 + np.random.normal(0, 0.5)
    z_base = -0.5 + np.random.normal(0, 0.3)

    # More dynamic: higher amplitude, more chaotic
    drink_freq = 0.12 + np.random.uniform(-0.04, 0.04)
    amp_x = 3.5 + np.random.uniform(-0.5, 1.0)
    amp_y = 3.0 + np.random.uniform(-0.5, 1.0)

    x = x_base + amp_x * np.sin(2 * np.pi * drink_freq * t) + np.random.normal(0, 0.8, n_samples)
    y = y_base + amp_y * np.cos(2 * np.pi * drink_freq * t) + np.random.normal(0, 0.8, n_samples)
    z = z_base + 1.2 * np.sin(2 * np.pi * drink_freq * t + 0.8) + np.random.normal(0, 0.5, n_samples)

    # Add occasional sharp motion bursts (picking up / putting down bottle)
    n_bursts = np.random.randint(2, 5)
    for _ in range(n_bursts):
        burst_pos = np.random.randint(20, n_samples - 20)
        burst_len = np.random.randint(5, 15)
        x[burst_pos:burst_pos+burst_len] += np.random.normal(0, 2.0, burst_len)
        y[burst_pos:burst_pos+burst_len] += np.random.normal(0, 2.0, burst_len)
        z[burst_pos:burst_pos+burst_len] += np.random.normal(0, 1.5, burst_len)

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': 2})  # 2 = Water


def generate_dataset(n_sessions_per_class=50):
    """Generate a full synthetic dataset."""
    all_data = []

    generators = [generate_coffee_session, generate_smoothie_session, generate_water_session]
    names = ['Coffee', 'Smoothie', 'Water']

    print("Generating synthetic sessions...")
    for cls_idx, (gen_fn, name) in enumerate(zip(generators, names)):
        for i in range(n_sessions_per_class):
            n_samples = np.random.randint(300, 700)
            df = gen_fn(n_samples=n_samples, session_id=i)
            df['session_id'] = f"{name}_{i:03d}"
            all_data.append(df)
        print(f"  {name}: {n_sessions_per_class} sessions done")

    dataset = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(dataset)}")
    print(f"Label distribution:\n{dataset['label'].value_counts().sort_index()}")
    print(f"  0=Coffee, 1=Smoothie, 2=Water")
    return dataset


if __name__ == "__main__":
    df = generate_dataset(n_sessions_per_class=60)
    df.to_csv("synthetic_data.csv", index=False)
    print("\nSaved to synthetic_data.csv")