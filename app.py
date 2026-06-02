"""
Drink Mind Reader — Streamlit App with built-in accelerometer
Deploy: push to GitHub → share.streamlit.io → select app.py
Local:  streamlit run app.py

NOTE: Accelerometer requires HTTPS in production.
      Streamlit Cloud provides HTTPS automatically.
      For local testing on phone, use: npx ngrok http 8501
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🥤 Drink Mind Reader",
    page_icon="🥤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────
WINDOW_SIZE  = 60
LABEL_NAMES  = {0: "Hot Coffee ☕", 1: "Strawberry Smoothie 🍓", 2: "Bottle of Water 💧"}
LABEL_COLORS = {0: "#e74c3c", 1: "#e91e8c", 2: "#2196F3"}
LABEL_DESCS  = {
    0: "Steady tilted sipping motion — phone angled ~45°, Z-axis strongly negative. Classic hot drink hold!",
    1: "Very stable vertical grip with straw sipping — Y-axis nearly maxed out, barely any tilt. Smoothie vibes!",
    2: "Dynamic high-range tilting detected — lots of acceleration variance. You're chugging from a bottle!",
}

# ── Load model ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

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

# ── Feature extraction (must match train_model.py exactly) ────
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

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0d14; }

.hero { text-align: center; padding: 2rem 0 1rem; }
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #f953c6, #b91d73, #2196F3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p { color: #888; font-size: 1rem; }

.result-box {
    background: rgba(255,255,255,0.05); border-radius: 20px;
    padding: 2rem; text-align: center; border: 2px solid; margin-top: 1.5rem;
}
.result-emoji { font-size: 4.5rem; }
.result-name  { font-size: 2rem; font-weight: 800; color: white; margin: 0.5rem 0; }
.result-desc  { color: #aaa; font-size: 0.95rem; line-height: 1.6; }
.instr-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    color: #ccc; font-size: 0.92rem; line-height: 1.7;
}
.mono { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #666; }
div[data-testid="stFileUploader"] label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🥤 Drink Mind Reader</h1>
  <p>Record your drinking motion — I'll guess what's in your hand</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **model.pkl / scaler.pkl not found.** Run `train_model.py` first, then push the .pkl files to your repo.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📱 Live Record", "📂 Upload CSV", "📊 Stats"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Built-in accelerometer
# ════════════════════════════════════════════════════════════════

with tab1:

    accel_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {
    font-family: Arial;
    background: #0d0d14;
    text-align: center;
    padding: 20px;
    margin: 0;
}
.container {
    background: #1a1a2e;
    max-width: 350px;
    margin: auto;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
h1 { color: white; margin-bottom: 10px; }
.value {
    font-size: 28px;
    margin: 10px;
    color: white;
}
.value span { color: white; font-weight: bold; }
button {
    width: 90%;
    padding: 15px;
    margin: 6px auto;
    display: block;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    transition: opacity 0.15s;
}
button:active { opacity: 0.75; }
.start    { background: #4CAF50; color: white; }
.stop     { background: #f44336; color: white; }
.download { background: #2196F3; color: white; }
.info {
    margin-top: 15px;
    font-size: 18px;
    color: white;
}
.status {
    margin: 10px 0;
    font-size: 14px;
    color: #aaa;
    min-height: 20px;
}
</style>
</head>
<body>
<div class="container">
    <h1>📱 Motion Logger</h1>
    <div class="value">X: <span id="x">0</span></div>
    <div class="value">Y: <span id="y">0</span></div>
    <div class="value">Z: <span id="z">0</span></div>
    <div class="info">Sampling Rate: <span id="hz">0</span> Hz</div>
    <div class="status" id="status">Press Start Recording to begin</div>
    <button class="start"    onclick="startSensor()">Start Recording</button>
    <button class="stop"     onclick="stopRecording()">Stop Recording</button>
    <button class="download" onclick="downloadCSV()">Download CSV</button>
</div>
<script>
let recording = false;
let sensorData = [];
let sampleCount = 0;
let startTime = 0;

function startSensor() {
    sensorData = [];
    sampleCount = 0;
    startTime = Date.now();
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission()
            .then(permissionState => {
                if (permissionState === 'granted') { startListening(); }
            })
            .catch(console.error);
    } else {
        startListening();
    }
}

function startListening() {
    recording = true;
    window.addEventListener('devicemotion', handleMotion);
    document.getElementById('status').textContent = 'Recording...';
    document.getElementById('status').style.color = '#ff4444';
}

function handleMotion(event) {
    if (!recording) return;
    let x = event.accelerationIncludingGravity.x || 0;
    let y = event.accelerationIncludingGravity.y || 0;
    let z = event.accelerationIncludingGravity.z || 0;
    document.getElementById('x').innerHTML = x.toFixed(2);
    document.getElementById('y').innerHTML = y.toFixed(2);
    document.getElementById('z').innerHTML = z.toFixed(2);
    let timestamp = Date.now();
    sensorData.push([timestamp, x, y, z]);
    sampleCount++;
    document.getElementById('status').textContent = `Recording... ${sampleCount} samples`;
}

function stopRecording() {
    recording = false;
    window.removeEventListener('devicemotion', handleMotion);
    let duration = (Date.now() - startTime) / 1000;
    let hz = sampleCount / duration;
    document.getElementById('hz').innerHTML = hz.toFixed(2);
    document.getElementById('status').style.color = '#51cf66';
    document.getElementById('status').textContent =
        `Done! ${sampleCount} samples @ ${hz.toFixed(1)} Hz — press Download CSV`;
}

function downloadCSV() {
    if (sensorData.length === 0) {
        document.getElementById('status').style.color = '#f44336';
        document.getElementById('status').textContent = 'No data yet — record first!';
        return;
    }
    let csv = 'timestamp,x,y,z\\n';
    sensorData.forEach(row => { csv += row.join(',') + '\\n'; });
    let blob = new Blob([csv], { type: 'text/csv' });
    let url = window.URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url;
    a.download = 'accelerometer_data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
    document.getElementById('status').textContent = 'CSV downloaded! Upload it in the Upload CSV tab.';
}
</script>
</body>
</html>
"""

    components.html(accel_html, height=480, scrolling=False)

    st.markdown("""
<div class="instr-card" style="margin-top:1rem">
⚠️ <b>iOS users:</b> The accelerometer requires HTTPS + a permission prompt.<br>
Streamlit Cloud provides HTTPS automatically.<br>
For local testing on your phone, run: <code>npx ngrok http 8501</code> and open the ngrok URL.
</div>
""", unsafe_allow_html=True)


with tab2:
    uploaded = st.file_uploader(
        "Upload accelerometer CSV (columns: timestamp, x, y, z)",
        type=["csv"], key="upload"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if not {'x','y','z'}.issubset(df.columns):
                st.error(f"CSV must have x, y, z columns. Found: {list(df.columns)}")
            else:
                st.success(f"✓ {len(df)} rows  ({len(df)/60:.1f}s @ 60 Hz)")
                st.line_chart(df[['x','y','z']].head(300), use_container_width=True)

                if len(df) < WINDOW_SIZE:
                    st.warning(f"Need at least {WINDOW_SIZE} rows.")
                elif st.button("🔮 Predict", key="btn_upload", use_container_width=True):
                    window       = df[['x','y','z']].tail(WINDOW_SIZE).values.tolist()
                    feats        = extract_features(window)
                    feats_scaled = scaler.transform(feats)
                    pred         = int(model.predict(feats_scaled)[0])
                    proba        = model.predict_proba(feats_scaled)[0]

                    color = LABEL_COLORS[pred]
                    st.markdown(f"""
<div class="result-box" style="border-color:{color}">
  <div class="result-emoji">{LABEL_NAMES[pred].split()[-1]}</div>
  <div class="result-name">{LABEL_NAMES[pred]}</div>
  <div class="result-desc">{LABEL_DESCS[pred]}</div>
</div>""", unsafe_allow_html=True)

                    st.markdown("#### Confidence")
                    for name, p in zip(LABEL_NAMES.values(), proba):
                        c1, c2 = st.columns([3, 1])
                        c1.progress(float(p), text=name)
                        c2.markdown(f"**{p*100:.1f}%**")
        except Exception as e:
            st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════
# TAB 3 — Stats
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### What makes each drink different?")
    stats_df = pd.DataFrame({
        "Drink":        ["☕ Hot Coffee",  "🍓 Smoothie",     "💧 Water"],
        "X mean":       ["−5.0",           "−3.5",            "−4.8"],
        "Y mean":       ["−7.0",           "−9.0 ⬅ key",     "−6.5"],
        "Z mean":       ["−3.0 ⬅ key",    " 0.0",            "−0.5"],
        "Motion":       ["Steady tilt",    "Very stable",     "High variance ⬅ key"],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="instr-card">
<b>Top features by Random Forest importance:</b><br>
1. <b>Z-axis max/mean</b> — Coffee tilts forward (Z ≈ −3), others near 0<br>
2. <b>Y-axis std/range</b> — Smoothie is very vertical and stable (Y ≈ −9)<br>
3. <b>Magnitude range</b> — Water bottle has the most dynamic, shaky motion<br>
4. <b>Jerk std</b> — Water shows sudden pickup/putdown acceleration bursts
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    "<br><p class='mono' style='text-align:center'>model: RandomForest · features: 37 · cv-accuracy: 99.9%</p>",
    unsafe_allow_html=True
)