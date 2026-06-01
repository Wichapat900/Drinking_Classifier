"""
Drink Mind Reader — Streamlit App
Deploy to Streamlit Cloud:
  1. Push all files to a GitHub repo
  2. Go to share.streamlit.io → New app → select your repo
  3. Set main file = app.py

Local run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🥤 Drink Mind Reader",
    page_icon="🥤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ────────────────────────────────────────────────
WINDOW_SIZE  = 60
LABEL_NAMES  = {0: "Hot Coffee ☕", 1: "Strawberry Smoothie 🍓", 2: "Bottle of Water 💧"}
LABEL_COLORS = {0: "#e74c3c",       1: "#e91e8c",               2: "#2196F3"}
LABEL_DESCS  = {
    0: "Steady tilted sipping motion — phone angled ~45°, Z-axis strongly negative. Classic hot drink hold!",
    1: "Very stable vertical grip with straw sipping — Y-axis nearly maxed out, barely any tilt. Smoothie vibes!",
    2: "Dynamic high-range tilting detected — lots of acceleration variance. You're chugging from a bottle!",
}

# ── Load model ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent   # always the folder where app.py lives

@st.cache_resource
def load_model():
    model_path  = BASE_DIR / "model.pkl"
    scaler_path = BASE_DIR / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Feature extraction (must match train_model.py exactly) ───
def extract_features(data: list) -> np.ndarray:
    x = np.array([d[0] for d in data], dtype=float)
    y = np.array([d[1] for d in data], dtype=float)
    z = np.array([d[2] for d in data], dtype=float)

    feats = []
    for v in [x, y, z]:
        feats += [
            np.mean(v), np.std(v), np.min(v), np.max(v),
            np.max(v) - np.min(v),
            np.percentile(v, 75) - np.percentile(v, 25),
            np.sum(v**2) / len(v),
            np.mean(np.abs(v - np.mean(v))),
        ]
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats += [np.mean(mag), np.std(mag), np.max(mag) - np.min(mag), np.sum(mag**2) / len(mag)]
    feats += [
        float(np.corrcoef(x, y)[0, 1]),
        float(np.corrcoef(x, z)[0, 1]),
        float(np.corrcoef(y, z)[0, 1]),
    ]
    for v in [x, y, z]:
        jerk = np.diff(v)
        feats += [np.std(jerk), float(np.max(np.abs(jerk)))]

    feats = [0.0 if np.isnan(f) else f for f in feats]
    return np.array(feats, dtype=float).reshape(1, -1)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0d0d14; }

.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f953c6, #b91d73, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p { color: #888; font-size: 1.05rem; }

.result-box {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    border: 2px solid;
    margin-top: 1.5rem;
}
.result-emoji { font-size: 4.5rem; }
.result-name  { font-size: 2rem; font-weight: 800; color: white; margin: 0.5rem 0; }
.result-desc  { color: #aaa; font-size: 0.95rem; line-height: 1.6; }

.instr-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    color: #ccc;
    font-size: 0.92rem;
    line-height: 1.7;
}

.mono { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #666; }

/* Override Streamlit widget colors */
div[data-testid="stFileUploader"] label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🥤 Drink Mind Reader</h1>
  <p>Upload your accelerometer recording and I'll guess what you were drinking</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ─────────────────────────────────────────────
if model is None:
    st.error("⚠️ **model.pkl / scaler.pkl not found.** Run `train_model.py` locally first, then push the .pkl files to your repo.")
    st.stop()

# ── How to record ────────────────────────────────────────────
with st.expander("📱 How to record your movement", expanded=False):
    st.markdown("""
<div class="instr-card">
<b>Step 1</b> — Open the recorder on your phone:<br>
<a href="https://techno.varee.ac.th/users/admin/acc.html" target="_blank">
  https://techno.varee.ac.th/users/admin/acc.html
</a><br><br>
<b>Step 2</b> — Hold your phone and drink (or mime drinking) for 5–10 seconds<br><br>
<b>Step 3</b> — Export/download the CSV<br><br>
<b>Step 4</b> — Upload it below ↓
</div>
""", unsafe_allow_html=True)

# ── Tabs: Upload CSV  OR  Paste raw data ─────────────────────
tab1, tab2, tab3 = st.tabs(["📂 Upload CSV", "📋 Paste Data", "📊 Dataset Stats"])

# ────────────────────────── Tab 1: Upload ────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload your accelerometer CSV (columns: timestamp, x, y, z)",
        type=["csv"],
        key="upload"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            req_cols = {'x', 'y', 'z'}
            if not req_cols.issubset(df.columns):
                st.error(f"CSV must have columns: x, y, z. Found: {list(df.columns)}")
            else:
                st.success(f"✓ Loaded {len(df)} rows ({len(df)/60:.1f}s @ 60 Hz)")

                # Preview chart
                st.line_chart(df[['x','y','z']].head(300), use_container_width=True)

                if len(df) < WINDOW_SIZE:
                    st.warning(f"Need at least {WINDOW_SIZE} rows. Please record longer.")
                else:
                    if st.button("🔮 Predict", key="btn_upload", use_container_width=True):
                        window = df[['x','y','z']].tail(WINDOW_SIZE).values.tolist()
                        feats  = extract_features(window)
                        feats_scaled = scaler.transform(feats)

                        pred  = int(model.predict(feats_scaled)[0])
                        proba = model.predict_proba(feats_scaled)[0]

                        # Result card
                        color = LABEL_COLORS[pred]
                        st.markdown(f"""
<div class="result-box" style="border-color:{color}">
  <div class="result-emoji">{LABEL_NAMES[pred].split()[-1]}</div>
  <div class="result-name">{LABEL_NAMES[pred]}</div>
  <div class="result-desc">{LABEL_DESCS[pred]}</div>
</div>
""", unsafe_allow_html=True)

                        # Probability bars
                        st.markdown("#### Confidence")
                        for i, (name, p) in enumerate(zip(LABEL_NAMES.values(), proba)):
                            col1, col2 = st.columns([3, 1])
                            col1.progress(float(p), text=name)
                            col2.markdown(f"**{p*100:.1f}%**")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ────────────────────────── Tab 2: Paste ─────────────────────
with tab2:
    st.markdown("Paste raw rows (one per line) in format: `x,y,z` or `timestamp,x,y,z`")
    sample = """0.08,-9.60,-2.55
-0.06,-9.78,-2.92
-0.05,-9.74,-2.97
-0.16,-9.43,-3.00"""
    raw = st.text_area("Paste data here:", height=180, placeholder=sample, key="paste_area")

    if st.button("🔮 Predict from pasted data", key="btn_paste", use_container_width=True):
        if not raw.strip():
            st.warning("Paste some data first.")
        else:
            try:
                rows = []
                for line in raw.strip().splitlines():
                    parts = [float(v) for v in line.split(',')]
                    if len(parts) == 3:
                        rows.append(parts)
                    elif len(parts) == 4:
                        rows.append(parts[1:])  # skip timestamp

                if len(rows) < WINDOW_SIZE:
                    st.warning(f"Need at least {WINDOW_SIZE} rows, got {len(rows)}.")
                else:
                    window = rows[-WINDOW_SIZE:]
                    feats  = extract_features(window)
                    feats_scaled = scaler.transform(feats)

                    pred  = int(model.predict(feats_scaled)[0])
                    proba = model.predict_proba(feats_scaled)[0]

                    color = LABEL_COLORS[pred]
                    st.markdown(f"""
<div class="result-box" style="border-color:{color}">
  <div class="result-emoji">{LABEL_NAMES[pred].split()[-1]}</div>
  <div class="result-name">{LABEL_NAMES[pred]}</div>
  <div class="result-desc">{LABEL_DESCS[pred]}</div>
</div>
""", unsafe_allow_html=True)

                    st.markdown("#### Confidence")
                    for i, (name, p) in enumerate(zip(LABEL_NAMES.values(), proba)):
                        col1, col2 = st.columns([3, 1])
                        col1.progress(float(p), text=name)
                        col2.markdown(f"**{p*100:.1f}%**")

            except Exception as e:
                st.error(f"Parse error: {e}")

# ────────────────────────── Tab 3: Stats ─────────────────────
with tab3:
    st.markdown("#### What makes each drink different?")
    stats_df = pd.DataFrame({
        "Drink":        ["☕ Hot Coffee", "🍓 Smoothie", "💧 Water"],
        "X mean (avg)": ["−5.0",         "−3.5",        "−4.8"],
        "Y mean (avg)": ["−7.0",         "−9.0 ⬅ key",  "−6.5"],
        "Z mean (avg)": ["−3.0 ⬅ key",  " 0.0",        "−0.5"],
        "Motion":       ["Steady tilt",  "Very stable", "High variance ⬅ key"],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="instr-card">
<b>Top discriminating features (from Random Forest importance):</b><br>
1. <b>Z-axis max/mean</b> — Coffee tilts forward (Z ≈ −3), others near 0<br>
2. <b>Y-axis std/range</b> — Smoothie is very vertical and stable (Y ≈ −9)<br>
3. <b>Magnitude range</b> — Water bottle has the most dynamic, shaky motion<br>
4. <b>Jerk (rate of change)</b> — Water shows sudden pickup/putdown bursts
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><p class='mono' style='text-align:center'>model: RandomForest · features: 37 · cv-accuracy: 99.9%</p>", unsafe_allow_html=True)