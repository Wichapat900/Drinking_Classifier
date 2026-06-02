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
from pathlib import Path

# 💡 IMPORT OUR NEW MOVEMENT DETECTOR
from movement_detector import detect_segments

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
tab1, tab2, tab3, tab4 = st.tabs(["📱 Live Record", "📂 Upload CSV", "🔬 Auto-Classify", "📊 Stats"])

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
body { font-family: Arial; background: #0d0d14; text-align: center; padding: 20px; margin: 0; }
.container { background: #1a1a2e; max-width: 350px; margin: auto; padding: 20px; border-radius: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.4); }
h1 { color: white; margin-bottom: 10px; }
.value { font-size: 28px; margin: 10px; color: white; }
.value span { color: white; font-weight: bold; }
button { width: 90%; padding: 15px; margin: 6px auto; display: block; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; transition: opacity 0.15s; }
button:active { opacity: 0.75; }
.start { background: #4CAF50; color: white; }
.stop { background: #f44336; color: white; }
.download { background: #2196F3; color: white; }
.info { margin-top: 15px; font-size: 18px; color: white; }
.status { margin: 10px 0; font-size: 14px; color: #aaa; min-height: 20px; }
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
    <button class="start" onclick="startSensor()">Start Recording</button>
    <button class="stop" onclick="stopRecording()">Stop Recording</button>
    <button class="download" onclick="downloadCSV()">Download CSV</button>
</div>
<script>
let recording = false;
let sensorData = [];
let sampleCount = 0;
let startTime = 0;

function startSensor() {
    sensorData = []; sampleCount = 0; startTime = Date.now();
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission().then(permissionState => {
            if (permissionState === 'granted') { startListening(); }
        }).catch(console.error);
    } else { startListening(); }
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
    sensorData.push([Date.now(), x, y, z]);
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
    document.getElementById('status').textContent = `Done! ${sampleCount} samples @ ${hz.toFixed(1)} Hz — press Download CSV`;
}

function downloadCSV() {
    if (sensorData.length === 0) return;
    let csv = 'timestamp,x,y,z\\n';
    sensorData.forEach(row => { csv += row.join(',') + '\\n'; });
    let uri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
    let w = window.parent.open(uri, '_blank');
    if (!w) { window.open(uri, '_blank'); }
    document.getElementById('status').style.color = '#51cf66';
    document.getElementById('status').textContent = 'Opened! Tap Share → Save to Files, then upload in Upload CSV tab.';
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


# ════════════════════════════════════════════════════════════════
# TAB 2 — Upload CSV
# ════════════════════════════════════════════════════════════════
with tab2:
    uploaded = st.file_uploader(
        "Upload accelerometer CSV (columns: timestamp, x, y, z)",
        type=["csv"], key="upload"
    )

    if uploaded:
        try:
            raw = uploaded.read()
            df = None
            for enc in ['utf-8', 'utf-16', 'cp1252', 'latin-1']:
                try:
                    df = pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                st.error("Could not read the CSV file.")
                st.stop()
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
# TAB 4 — Stats
# ════════════════════════════════════════════════════════════════
with tab4:
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


# ════════════════════════════════════════════════════════════════
# TAB 3 — Auto-Classify Timeline (EVENT-BASED CHUNKING)
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🔬 Auto-classify by Activity Segments")
    st.caption("Upload a CSV. The `movement_detector` will automatically crop the silence, cut out the 'wiggles', and predict each chunk.")

    label_file = st.file_uploader(
        "Upload merged CSV (timestamp, x, y, z)", type=["csv"], key="label_upload"
    )

    if label_file:
        # ── Load CSV ──────────────────────────────────────────────
        raw_bytes = label_file.read()
        ldf = None
        for enc in ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252', 'latin-1']:
            try:
                ldf = pd.read_csv(pd.io.common.BytesIO(raw_bytes), encoding=enc)
                if {'x','y','z'}.issubset(ldf.columns):
                    break
                ldf = None
            except Exception:
                continue
        
        if ldf is None:
            st.error("Could not read the file. Make sure it has x, y, z columns.")
            st.stop()

        n_rows = len(ldf)

        # ── Parse timestamps → real datetime ──────────────────────
        ts_col = next((c for c in ldf.columns if 'time' in c.lower()), None)
        if ts_col:
            ts_raw = pd.to_numeric(ldf[ts_col], errors='coerce').ffill()
            t0_raw = ts_raw.iloc[0]
            is_ms  = t0_raw > 1e11
            ts_sec = ts_raw / 1000.0 if is_ms else ts_raw
            ldf['datetime'] = pd.to_datetime(ts_sec, unit='s', utc=True).dt.tz_convert('Asia/Bangkok')
            ldf['seconds']  = ((ts_raw - t0_raw) / (1000.0 if is_ms else 1.0)).round(3)
        else:
            ldf['seconds']  = (np.arange(n_rows) / 60.0).round(3)
            ldf['datetime'] = pd.Timestamp.now(tz='Asia/Bangkok')

        total_sec = float(ldf['seconds'].iloc[-1])
        
        # 💡 NEW: Robust Hz calculation that ignores big empty gaps
        diffs = ldf['seconds'].diff().dropna()
        normal_diffs = diffs[diffs < 1.0]
        if len(normal_diffs) > 0 and normal_diffs.median() > 0:
            hz_est = int(round(1.0 / normal_diffs.median()))
        else:
            hz_est = 60
        hz_est = max(10, min(hz_est, 200))

        st.success(f"✓ {n_rows:,} rows · {total_sec:.1f} s · Detected ~{hz_est} Hz Sampling Rate")

        # ── Full waveform plot ────────────────────────────────────
        plot_df = ldf[['x','y','z']].copy()
        plot_df.index = ldf['seconds']
        st.line_chart(plot_df, use_container_width=True, height=200)

        st.markdown("---")

        # ── USE OUR NEW IMPORT TO GET SEGMENTS ────────────────────
        # Threshold set to 0.15 to perfectly segment active blocks from timeline data
        segments = detect_segments(ldf, hz_est=hz_est, threshold=0.15, min_window=WINDOW_SIZE)

        # ── Predict on each isolated segment ──────────────────────
        results = []
        xyz = ldf[['x','y','z']].values
        
        ldf['label'] = 'Idle 😴'  
        ldf['confidence'] = 0.0

        if not segments:
            st.info("No clear drinking activity detected! Try lowering the threshold in `detect_segments` if your movements are very gentle.")
        else:
            for i, (s_idx, e_idx) in enumerate(segments):
                # 1. Isolate the "wiggle" data
                segment_data = xyz[s_idx : e_idx + 1].tolist()
                
                # 2. Extract features and predict
                feats = extract_features(segment_data)
                scaled = scaler.transform(feats)
                pred = int(model.predict(scaled)[0])
                proba = model.predict_proba(scaled)[0]
                
                label = LABEL_NAMES[pred]
                conf_pct = float(proba.max() * 100)
                
                start_dt = ldf['datetime'].iloc[s_idx]
                end_dt   = ldf['datetime'].iloc[e_idx]
                
                duration_sec = float(ldf['seconds'].iloc[e_idx] - ldf['seconds'].iloc[s_idx])
                
                results.append({
                    'Segment':     f"Action {i+1}",
                    'Activity':    label,
                    'Start time':  start_dt.strftime('%H:%M:%S.%f')[:-3],
                    'End time':    end_dt.strftime('%H:%M:%S.%f')[:-3],
                    'Duration':    f"{duration_sec:.1f}s",
                    'Confidence':  f"{conf_pct:.1f}%",
                })
                
                # Tag the rows in the main dataframe for export
                ldf.loc[s_idx:e_idx, 'label'] = label
                ldf.loc[s_idx:e_idx, 'confidence'] = conf_pct

            res_df = pd.DataFrame(results)

            # ── Display the results ───────────────────────────────────
            st.markdown("#### 📋 Detected Activities")
            for _, row in res_df.iterrows():
                try:
                    emoji = row['Activity'].split()[0]
                    name  = ' '.join(row['Activity'].split()[1:])
                except IndexError:
                    emoji = "✨"
                    name  = row['Activity']
                    
                st.markdown(
                    f"**{row['Segment']}** &nbsp;→&nbsp; {emoji} **{name}** &nbsp;·&nbsp; "
                    f"*{row['Start time']} to {row['End time']}* &nbsp;·&nbsp; "
                    f"Duration: {row['Duration']} &nbsp;·&nbsp; Conf: {row['Confidence']}"
                )

            st.markdown("---")
            st.markdown("#### Segment Details Table")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        # ── Export ────────────────────────────────────────────────
        out_df = ldf.copy()
        out_df['datetime'] = out_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        st.download_button(
            label="⬇️ Download Labelled CSV",
            data=out_df.to_csv(index=False),
            file_name="classified_recording.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    "<br><p class='mono' style='text-align:center'>model: RandomForest · features: 37 · cv-accuracy: 99.9%</p>",
    unsafe_allow_html=True
)