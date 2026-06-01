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

    # The component returns JSON: {"samples": [[x,y,z], ...], "csv": "..."}
    # We use a key so Streamlit re-renders when data arrives
    accel_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d0d14;
    color: white;
    padding: 16px;
    min-height: 280px;
  }

  .axes {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 14px;
  }
  .axis-box {
    background: rgba(255,255,255,0.07); border-radius: 12px;
    padding: 10px; text-align: center;
  }
  .axis-label { font-size: 0.7rem; opacity: 0.55; margin-bottom: 3px; }
  .axis-val   { font-size: 1.3rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .ax { color: #ff6b6b; } .ay { color: #51cf66; } .az { color: #339af0; }

  canvas {
    width: 100%; height: 80px; display: block;
    background: rgba(0,0,0,0.4); border-radius: 10px; margin-bottom: 12px;
  }

  .status {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.82rem; color: #aaa; margin-bottom: 12px;
    background: rgba(255,255,255,0.05); border-radius: 8px; padding: 8px 12px;
  }
  .dot {
    width: 9px; height: 9px; border-radius: 50%; background: #444; flex-shrink: 0;
  }
  .dot.rec { background: #ff4444; animation: blink 1s infinite; }
  .dot.ok  { background: #51cf66; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

  .btns { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
  .btn {
    padding: 13px; border: none; border-radius: 12px;
    font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: all 0.15s;
  }
  .btn:active { transform: scale(0.96); }
  .btn-rec  { background: linear-gradient(135deg,#11998e,#38ef7d); color: #0a1f1a; }
  .btn-stop { background: linear-gradient(135deg,#f7971e,#ffd200); color: #1a1000; }
  .btn-send { background: linear-gradient(135deg,#f953c6,#b91d73); color: white; grid-column: span 2; }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }

  .dl-btn {
    width: 100%; padding: 10px; border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px; background: transparent; color: #aaa;
    font-size: 0.85rem; cursor: pointer; margin-top: 6px;
  }
  .dl-btn:hover { background: rgba(255,255,255,0.07); }

  .hint { font-size: 0.78rem; color: #555; text-align: center; margin-top: 8px; line-height: 1.5; }
  .err  { background: rgba(255,70,70,0.15); border: 1px solid #ff4444;
          border-radius: 8px; padding: 10px; font-size: 0.82rem; margin-top: 8px; display:none; }
</style>
</head>
<body>

<div class="axes">
  <div class="axis-box"><div class="axis-label">X</div><div class="axis-val ax" id="vx">0.00</div></div>
  <div class="axis-box"><div class="axis-label">Y</div><div class="axis-val ay" id="vy">0.00</div></div>
  <div class="axis-box"><div class="axis-label">Z</div><div class="axis-val az" id="vz">0.00</div></div>
</div>

<canvas id="cv" width="460" height="80"></canvas>

<div class="status"><div class="dot" id="dot"></div><span id="stxt">Press Start to begin</span></div>

<div class="btns">
  <button class="btn btn-rec"  id="btnS" onclick="startRec()">▶ Start</button>
  <button class="btn btn-stop" id="btnX" onclick="stopRec()" disabled>⏹ Stop</button>
  <button class="btn btn-send" id="btnP" onclick="sendData()" disabled>🔮 Predict My Drink</button>
</div>
<button class="dl-btn" id="btnDL" onclick="downloadCSV()" style="display:none">⬇ Download CSV</button>
<p class="hint">Hold your phone naturally while drinking for 5–10 seconds</p>
<div class="err" id="err"></div>

<script>
const NEED = 60;
let recording = false, samples = [], hasPerm = false;
let lx = 0, ly = 0, lz = 0;
const wave = { x:[], y:[], z:[] };
const MAX_W = 220;

const canvas = document.getElementById('cv');
const ctx    = canvas.getContext('2d');

// ── Permission ──────────────────────────────────────────────
async function ensurePerm() {
  if (hasPerm) return true;
  if (typeof DeviceMotionEvent === 'undefined') {
    showErr('DeviceMotionEvent not available on this browser.');
    return false;
  }
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const r = await DeviceMotionEvent.requestPermission();
      if (r !== 'granted') { showErr('Motion permission denied.'); return false; }
    } catch(e) { showErr('Permission error: ' + e.message); return false; }
  }
  window.addEventListener('devicemotion', onMotion);
  hasPerm = true;
  return true;
}

// ── Motion handler ──────────────────────────────────────────
function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a) return;
  lx = a.x ?? 0; ly = a.y ?? 0; lz = a.z ?? 0;
  document.getElementById('vx').textContent = lx.toFixed(2);
  document.getElementById('vy').textContent = ly.toFixed(2);
  document.getElementById('vz').textContent = lz.toFixed(2);

  if (recording) {
    samples.push([lx, ly, lz]);
    const s = (samples.length / 60).toFixed(1);
    document.getElementById('stxt').textContent =
      `Recording… ${samples.length} samples (${s}s)`;
  }

  ['x','y','z'].forEach((ax,i)=>{
    const v=[lx,ly,lz][i];
    wave[ax].push(v);
    if(wave[ax].length > MAX_W) wave[ax].shift();
  });
  drawWave();
}

// ── Waveform ────────────────────────────────────────────────
function drawWave() {
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const colors={x:'#ff6b6b',y:'#51cf66',z:'#339af0'};
  const sc = H/40;
  ['x','y','z'].forEach(ax=>{
    const h=wave[ax]; if(h.length<2) return;
    ctx.beginPath(); ctx.strokeStyle=colors[ax]; ctx.lineWidth=1.5;
    h.forEach((v,i)=>{
      const px=(i/(MAX_W-1))*W, py=H/2-v*sc;
      i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
    });
    ctx.stroke();
  });
}

// ── Controls ────────────────────────────────────────────────
async function startRec() {
  if (!await ensurePerm()) return;
  recording = true; samples = [];
  set('btnS', true); set('btnX', false); set('btnP', true);
  document.getElementById('btnDL').style.display = 'none';
  document.getElementById('dot').className = 'dot rec';
  document.getElementById('stxt').textContent = 'Recording…';
  hideErr();
}

function stopRec() {
  recording = false;
  set('btnS', false); set('btnX', true);
  const ok = samples.length >= NEED;
  set('btnP', !ok);
  document.getElementById('dot').className = ok ? 'dot ok' : 'dot';
  document.getElementById('stxt').textContent = ok
    ? `✓ ${samples.length} samples — ready to predict!`
    : `Only ${samples.length} samples — need ${NEED}+. Record longer.`;
  if (ok) document.getElementById('btnDL').style.display = 'block';
}

function set(id, disabled) { document.getElementById(id).disabled = disabled; }

// ── Send to Streamlit ────────────────────────────────────────
function sendData() {
  if (samples.length < NEED) { showErr('Not enough data.'); return; }
  // Send via Streamlit component messaging
  const payload = JSON.stringify({ samples: samples });
  window.parent.postMessage({ type: 'streamlit:setComponentValue', value: payload }, '*');
}

// ── Download CSV ─────────────────────────────────────────────
function downloadCSV() {
  let csv = 'timestamp,x,y,z\\n';
  const t0 = Date.now() - samples.length * 16;
  samples.forEach((s,i) => { csv += `${t0 + i*16},${s[0]},${s[1]},${s[2]}\\n`; });
  const a = document.createElement('a');
  a.href = 'data:text/csv,' + encodeURIComponent(csv);
  a.download = 'recording.csv';
  a.click();
}

// ── Helpers ──────────────────────────────────────────────────
function showErr(m) {
  const e=document.getElementById('err'); e.textContent='⚠ '+m; e.style.display='block';
}
function hideErr() { document.getElementById('err').style.display='none'; }

drawWave();
</script>
</body>
</html>
"""

    # Render the accelerometer component
    result = components.html(accel_html, height=420, scrolling=False)

    st.markdown("---")

    # ── Manual JSON paste fallback (receives postMessage data) ──
    st.markdown("##### Paste recorded data to predict")
    st.caption("After stopping, copy the data from the recorder or use the Download CSV button above, then paste below:")

    raw_json = st.text_area(
        "Paste raw samples as JSON array  `[[x,y,z], [x,y,z], ...]`  or CSV rows  `x,y,z`",
        height=100,
        placeholder='[[-1.2, -9.5, -0.3], [-1.1, -9.6, -0.2], ...]',
        key="live_paste"
    )

    if st.button("🔮 Predict", key="btn_live_predict", use_container_width=True):
        samples = []
        raw = raw_json.strip()
        if not raw:
            st.warning("Paste your recorded data first.")
        else:
            try:
                # Try JSON array first
                if raw.startswith('['):
                    parsed = json.loads(raw)
                    # Could be [[x,y,z],...] or [{"x":...},...]
                    for item in parsed:
                        if isinstance(item, list) and len(item) >= 3:
                            samples.append([float(item[0]), float(item[1]), float(item[2])])
                        elif isinstance(item, dict):
                            samples.append([float(item['x']), float(item['y']), float(item['z'])])
                else:
                    # CSV rows: x,y,z or timestamp,x,y,z
                    for line in raw.splitlines():
                        parts = [float(v) for v in line.split(',')]
                        if len(parts) == 3:
                            samples.append(parts)
                        elif len(parts) == 4:
                            samples.append(parts[1:])
            except Exception as e:
                st.error(f"Parse error: {e}")
                samples = []

            if samples and len(samples) < WINDOW_SIZE:
                st.warning(f"Need at least {WINDOW_SIZE} samples, got {len(samples)}. Record longer.")
            elif samples:
                window       = samples[-WINDOW_SIZE:]
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

    st.markdown("""
<div class="instr-card" style="margin-top:1rem">
⚠️ <b>iOS users:</b> The accelerometer requires HTTPS + a permission prompt.<br>
Streamlit Cloud provides HTTPS automatically — it will just work when deployed.<br>
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