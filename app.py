"""
app.py — Drinking Gesture Classifier
=====================================
Streamlit app that:
  1. Accepts live X/Y/Z accelerometer data (paste from acc.html recorder)
  2. Segments it into windows, extracts features, runs the trained model
  3. Shows prediction: Water Bottle / Strawberry Smoothie / Hot Coffee

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json, joblib, io
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# ── Signal params (must match train.py) ────────────────────────────────────
FS      = 60
WIN_SEC = 3.0
WIN_N   = int(WIN_SEC * FS)   # 180 samples
STEP_N  = int(1.0     * FS)   # 60 samples

LABEL_NAMES  = ["Hot Coffee ☕", "Strawberry Smoothie 🍓", "Water Bottle 💧"]
LABEL_EMOJIS = ["☕", "🍓", "💧"]
LABEL_COLORS = ["#f4a124", "#e05a7a", "#2ab5b5"]

# ── CardioSense-style palette ───────────────────────────────────────────────
C = {
    "bg":       "#050b12", "panel":  "#080f18", "panel2": "#0c1620",
    "border":   "#1a2d3d", "text":   "#c8dde8", "text_mid":"#7a9bb8",
    "text_dim": "#3a5a78", "accent": "#2ab5b5", "success":"#1fcc7a",
    "danger":   "#f04060", "warn":   "#f4a124", "white":  "#ffffff",
}

# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Drink_Classifier", page_icon="🥤",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Sora:wght@600;700&display=swap');
* {{ box-sizing:border-box; }}
html,body,[class*="css"] {{ font-family:'Inter',sans-serif; }}
.stApp {{ background:{C['bg']}; color:{C['text']}; }}
.main .block-container {{ padding:0 !important; max-width:100% !important; }}
[data-testid="stSidebar"] {{ background:{C['panel']} !important; border-right:1px solid {C['border']} !important; }}
[data-testid="stSidebar"] * {{ color:{C['text']} !important; }}
[data-testid="stSidebar"] hr {{ border-color:{C['border']} !important; }}
.stTabs [data-baseweb="tab-list"] {{ background:{C['panel']}; border-bottom:1px solid {C['border']}; padding:0 1.5rem; }}
.stTabs [data-baseweb="tab"] {{ color:{C['text_mid']} !important; font-size:0.78rem !important; font-weight:500 !important;
    letter-spacing:0.07em !important; text-transform:uppercase !important;
    padding:0.9rem 1.4rem !important; border-bottom:2px solid transparent !important;
    margin-bottom:-1px !important; background:transparent !important; }}
.stTabs [aria-selected="true"] {{ color:{C['accent']} !important; border-bottom:2px solid {C['accent']} !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding:1.5rem 2rem !important; background:{C['bg']}; }}
[data-testid="metric-container"] {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:10px; padding:1rem !important; }}
[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace !important; font-size:1.5rem !important; color:{C['white']} !important; }}
[data-testid="stMetricLabel"] {{ font-size:0.65rem !important; font-weight:600 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:{C['text_dim']} !important; }}
.cs-card {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:12px; padding:1.4rem; margin-bottom:0.8rem; }}
.cs-label {{ font-size:0.62rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
    color:{C['text_dim']}; margin-bottom:0.5rem; padding-bottom:0.3rem; border-bottom:1px solid {C['border']}; }}
.stDownloadButton>button {{ background:linear-gradient(135deg,#1e6fa8,#2ab5b5) !important;
    color:white !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }}
[data-testid="stFileUploader"] {{ background:{C['panel']}; border:1.5px dashed #243d55; border-radius:10px; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{C['bg']}; }}
::-webkit-scrollbar-thumb {{ background:#243d55; border-radius:3px; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (must match train.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

def window_features(seg: np.ndarray) -> np.ndarray:
    mag = np.sqrt((seg**2).sum(axis=1))
    channels = [seg[:,0], seg[:,1], seg[:,2], mag]
    feats = []
    for ch in channels:
        feats += [
            ch.mean(), ch.std(), ch.min(), ch.max(),
            ch.max()-ch.min(),
            float(skew(ch)), float(kurtosis(ch)),
            np.percentile(ch,25), np.percentile(ch,75),
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


def extract_all_windows(sig: np.ndarray):
    """Slide window over signal → list of feature vectors."""
    Xf = []
    start = 0
    while start + WIN_N <= len(sig):
        Xf.append(window_features(sig[start:start+WIN_N]))
        start += STEP_N
    return np.array(Xf) if Xf else None


def majority_vote(predictions):
    counts = np.bincount(predictions, minlength=3)
    return int(np.argmax(counts)), counts


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    mp = Path("models/model.pkl");   sp = Path("models/scaler.pkl")
    rp = Path("models/report.json")
    if not mp.exists() or not sp.exists():
        return None, None, None
    model   = joblib.load(str(mp))
    scaler  = joblib.load(str(sp))
    report  = json.load(open(rp)) if rp.exists() else {}
    return model, scaler, report


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_axes(df, title="Accelerometer Signal"):
    t = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
    fig = go.Figure()
    for col, color, name in [("x","#f04060","X-axis"),
                              ("y","#2ab5b5","Y-axis"),
                              ("z","#f4a124","Z-axis")]:
        fig.add_trace(go.Scatter(x=t, y=df[col], mode="lines",
            line=dict(color=color, width=1.4), name=name,
            hovertemplate=f"t=%{{x:.2f}}s  {col}=%{{y:.3f}}<extra></extra>"))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter",size=12,color=C["text_mid"]), x=0.01),
        paper_bgcolor=C["panel"], plot_bgcolor=C["panel2"],
        xaxis=dict(title="Time (s)", color=C["text_mid"], gridcolor=C["border"],
                   tickfont=dict(family="JetBrains Mono",size=10)),
        yaxis=dict(title="m/s²", color=C["text_mid"], gridcolor=C["border"],
                   tickfont=dict(family="JetBrains Mono",size=10)),
        legend=dict(bgcolor="rgba(8,15,24,0.85)", bordercolor=C["border"],
                    font=dict(family="Inter",size=11,color=C["text"])),
        height=280, margin=dict(l=55,r=15,t=40,b=45),
    )
    return fig


def plot_magnitude(df):
    t   = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
    mag = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=mag, mode="lines", fill="tozeroy",
        line=dict(color=C["accent"], width=1.5),
        fillcolor="rgba(42,181,181,0.08)",
        hovertemplate="t=%{x:.2f}s  |mag|=%{y:.3f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="Signal Magnitude", font=dict(family="Inter",size=12,color=C["text_mid"]), x=0.01),
        paper_bgcolor=C["panel"], plot_bgcolor=C["panel2"],
        xaxis=dict(title="Time (s)",color=C["text_mid"],gridcolor=C["border"],
                   tickfont=dict(family="JetBrains Mono",size=10)),
        yaxis=dict(title="m/s²",color=C["text_mid"],gridcolor=C["border"],
                   tickfont=dict(family="JetBrains Mono",size=10)),
        height=220, margin=dict(l=55,r=15,t=40,b=45),
    )
    return fig


def plot_votes(counts):
    colors = LABEL_COLORS
    fig = go.Figure(go.Bar(
        x=LABEL_NAMES, y=counts,
        marker_color=colors,
        text=[str(c) for c in counts],
        textposition="outside",
        textfont=dict(family="JetBrains Mono",size=12,color=C["text"]),
    ))
    fig.update_layout(
        title=dict(text="Window Vote Distribution", font=dict(family="Inter",size=12,color=C["text_mid"])),
        paper_bgcolor=C["panel"], plot_bgcolor=C["panel2"],
        xaxis=dict(color=C["text_mid"], tickfont=dict(family="Inter",size=11)),
        yaxis=dict(color=C["text_mid"], gridcolor=C["border"],
                   tickfont=dict(family="JetBrains Mono",size=10)),
        height=240, margin=dict(l=55,r=15,t=45,b=40), showlegend=False,
    )
    return fig


def plot_confidence_gauge(prob, label_idx):
    color = LABEL_COLORS[label_idx]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        number=dict(suffix="%", font=dict(color=color, size=32, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(color=C["text_dim"], family="JetBrains Mono", size=9)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=C["panel2"], borderwidth=1, bordercolor=C["border"],
            steps=[dict(range=[0,50], color="rgba(42,181,181,0.05)"),
                   dict(range=[50,100], color="rgba(42,181,181,0.02)")],
        ),
    ))
    fig.update_layout(paper_bgcolor=C["panel"], height=180,
                      margin=dict(l=15,r=15,t=15,b=10),
                      font=dict(color=C["text"]))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_signal(source) -> pd.DataFrame | None:
    """Accept file-like or raw CSV text. Returns df with timestamp,x,y,z."""
    try:
        if hasattr(source, "read"):
            content = source.read().decode("utf-8", errors="ignore")
        else:
            content = str(source)
        df = pd.read_csv(io.StringIO(content))
        needed = ["timestamp","x","y","z"]
        # Try to auto-detect columns
        if not all(c in df.columns for c in needed):
            if len(df.columns) == 4:
                df.columns = needed
            elif len(df.columns) == 3:
                df.columns = ["x","y","z"]
                df.insert(0, "timestamp", np.arange(len(df)) * (1000/FS))
            else:
                return None
        return df[needed].dropna().reset_index(drop=True)
    except Exception:
        return None


def predict_signal(df, model, scaler):
    """Run full prediction pipeline on a parsed dataframe."""
    sig  = df[["x","y","z"]].values.astype(np.float32)
    Xf   = extract_all_windows(sig)
    if Xf is None or len(Xf) == 0:
        return None, None, None, None
    Xf_s = scaler.transform(Xf)
    preds = model.predict(Xf_s)
    probas = model.predict_proba(Xf_s)
    label_idx, counts = majority_vote(preds)
    confidence = float(probas[:, label_idx].mean())
    return label_idx, confidence, counts, probas


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def sidebar(model, report):
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1rem 0 0.8rem;'>
          <div style='font-size:1.8rem; margin-bottom:6px;'>🥤</div>
          <div style='font-family:"Sora",sans-serif; font-size:1.3rem; color:white; font-weight:700; line-height:1;'>
            DrinkClassifier
          </div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:0.55rem; color:{C["text_dim"]};
                      letter-spacing:0.12em; margin-top:4px;'>DRINKING GESTURE CLASSIFIER v1.0</div>
          <div style='font-size:0.7rem; color:{C["text_mid"]}; margin-top:6px; line-height:1.5;'>
            Detects what you're drinking<br>from wrist accelerometer motion
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a5a78;margin-bottom:0.5rem;padding-bottom:0.3rem;border-bottom:1px solid #1a2d3d;">Input Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Input Mode",
            ["Upload CSV file", "Paste CSV text", "Demo — Coffee ☕", "Demo — Smoothie 🍓", "Demo — Water 💧"],
            label_visibility="collapsed")

        st.divider()
        st.markdown('<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a5a78;margin-bottom:0.5rem;padding-bottom:0.3rem;border-bottom:1px solid #1a2d3d;">Model Status</div>', unsafe_allow_html=True)
        if model:
            cv = report.get("cv_accuracy", 0)
            mn = report.get("model", "Unknown")
            st.markdown(f"""
            <div style='display:inline-flex;align-items:center;gap:5px;background:#0c1620;
                border:1px solid #1fcc7a;border-radius:16px;padding:3px 10px;
                font-size:0.72rem;font-family:JetBrains Mono,monospace;color:#1fcc7a;margin:2px 0;'>
              🟢 {mn} loaded — CV: {cv*100:.1f}%
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:inline-flex;align-items:center;gap:5px;background:#0c1620;
                border:1px solid #f04060;border-radius:16px;padding:3px 10px;
                font-size:0.72rem;font-family:JetBrains Mono,monospace;color:#f04060;margin:2px 0;'>
              🔴 No model — run train.py first
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"""
        <div style='font-size:0.62rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                    color:{C["text_dim"]};margin-bottom:0.5rem;padding-bottom:0.3rem;
                    border-bottom:1px solid {C["border"]};'>How to record</div>
        <div style='font-size:0.78rem; color:{C["text_mid"]}; line-height:1.7;'>
          1. Open the recorder:<br>
          <code style='font-size:0.72rem;color:{C["accent"]};'>techno.varee.ac.th/users/admin/acc.html</code><br>
          2. Hold your phone like you're drinking<br>
          3. Perform the drinking motion (~5-15 s)<br>
          4. Export → download CSV<br>
          5. Upload here ↑
        </div>""", unsafe_allow_html=True)

        return mode


# ═══════════════════════════════════════════════════════════════════════════
# DEMO SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

def make_demo(label: str, duration=8.0) -> pd.DataFrame:
    """Generate a synthetic demo signal for the given label."""
    rng = np.random.default_rng({"Coffee":0,"Strawberry":1,"Water":2}[label])
    n   = int(duration * FS)
    t   = np.arange(n) / FS
    env = np.minimum(np.clip(t/(duration*0.15),0,1),
                     np.clip((duration-t)/(duration*0.15),0,1))
    if label == "Coffee":
        x = -0.3 + 3.0*np.sin(2*np.pi*0.8*t)*env + 0.8*rng.standard_normal(n)
        y = -9.5 + 1.8*env*np.sin(2*np.pi*0.5*t) + 0.5*rng.standard_normal(n)
        z = -2.3 + 0.6*np.sin(2*np.pi*0.6*t)     + 0.3*rng.standard_normal(n)
    elif label == "Strawberry":
        x = -1.4 + 2.0*np.sin(2*np.pi*0.5*t)*env + 0.5*rng.standard_normal(n)
        y = -9.6 + 0.8*env*np.sin(2*np.pi*0.4*t) + 0.3*rng.standard_normal(n)
        z = -1.0 + 0.9*np.sin(2*np.pi*0.5*t)     + 0.4*rng.standard_normal(n)
    else:  # Water
        x = -2.0 + 4.0*np.sin(2*np.pi*0.6*t)*env + 1.2*rng.standard_normal(n)
        y = -9.2 + 3.5*env*np.sin(2*np.pi*0.5*t) + 1.0*rng.standard_normal(n)
        z = -2.0 + 1.5*np.sin(2*np.pi*0.7*t)     + 0.7*rng.standard_normal(n)
    ts = 1_779_770_000_000 + (np.arange(n) * (1000/FS)).astype(int)
    return pd.DataFrame({"timestamp":ts,"x":x,"y":y,"z":z})


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    model, scaler, report = load_model()
    mode = sidebar(model, report)

    # ── TOP BAR ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{C["panel"]}; border-bottom:1px solid {C["border"]};
                padding:0.85rem 2rem; display:flex; align-items:center; justify-content:space-between;'>
      <div style='display:flex; align-items:center; gap:12px;'>
        <span style='font-size:1.6rem;'>🥤</span>
        <div>
          <span style='font-family:"Sora",sans-serif; font-size:1.25rem; color:white; font-weight:700;'>
            DrinkClassifier
          </span>
          <span style='font-family:"Inter",sans-serif; font-size:0.72rem; color:{C["text_dim"]};
                       margin-left:10px; letter-spacing:0.08em; text-transform:uppercase;'>
            Drinking Gesture Classifier
          </span>
        </div>
      </div>
      <div style='display:flex; gap:12px; align-items:center;'>
        <span style='font-size:1.3rem;'>☕</span>
        <span style='font-size:1.3rem;'>🍓</span>
        <span style='font-size:1.3rem;'>💧</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown(f"""
        <div style='padding:60px 2rem;'>
          <div style='background:{C["panel"]}; border:1px solid {C["border"]}; border-left:4px solid {C["danger"]};
                      border-radius:10px; padding:1.4rem 1.6rem;'>
            <div style='font-family:"Sora",sans-serif; font-size:1rem; color:{C["danger"]}; font-weight:700; margin-bottom:6px;'>
              ⚠️ Model not found
            </div>
            <div style='font-size:0.85rem; color:{C["text_mid"]}; line-height:1.7;'>
              No trained model found in <code>models/</code>. Run the following commands first:
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.code("python generate_data.py   # generate synthetic training data\npython train.py           # train and save the model")
        return

    # ── LOAD SIGNAL ──────────────────────────────────────────────────────
    df = None; source_name = ""

    if mode == "Upload CSV file":
        uploaded = st.file_uploader(
            "Upload accelerometer CSV (timestamp, x, y, z)", type=["csv","txt"])
        if uploaded:
            df = parse_signal(uploaded)
            source_name = uploaded.name
            if df is None:
                st.error("Could not parse file. Make sure it has columns: timestamp, x, y, z")

    elif mode == "Paste CSV text":
        st.markdown(f'<div class="cs-label">Paste CSV data from acc.html recorder</div>',
                    unsafe_allow_html=True)
        raw = st.text_area("Paste CSV here", height=160,
                           placeholder="timestamp,x,y,z\n1779769989454,-0.045,...")
        if raw.strip():
            df = parse_signal(raw)
            source_name = "pasted data"
            if df is None:
                st.error("Could not parse CSV. Expected columns: timestamp, x, y, z")

    elif mode == "Demo — Coffee ☕":
        df = make_demo("Coffee"); source_name = "Demo Coffee signal"
    elif mode == "Demo — Smoothie 🍓":
        df = make_demo("Strawberry"); source_name = "Demo Smoothie signal"
    elif mode == "Demo — Water 💧":
        df = make_demo("Water"); source_name = "Demo Water signal"

    if df is None:
        st.markdown(f"""
        <div style='padding:80px 20px; text-align:center;'>
          <div style='font-size:3rem; margin-bottom:12px;'>🥤</div>
          <div style='font-family:"Inter",sans-serif; font-size:0.9rem;
                      color:{C["text_dim"]}; letter-spacing:0.06em;'>
            Select an input mode in the sidebar to begin
          </div>
          <div style='font-size:0.78rem; color:{C["text_dim"]}; margin-top:8px;'>
            Record motion data at
            <a href="https://techno.varee.ac.th/users/admin/acc.html" target="_blank"
               style="color:{C["accent"]};">techno.varee.ac.th/users/admin/acc.html</a>
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── PREDICT ──────────────────────────────────────────────────────────
    if len(df) < WIN_N:
        st.warning(f"Signal too short ({len(df)} samples = {len(df)/FS:.1f}s). "
                   f"Need at least {WIN_N} samples ({WIN_SEC:.0f}s).")
        return

    label_idx, confidence, counts, probas = predict_signal(df, model, scaler)
    if label_idx is None:
        st.error("Feature extraction failed.")
        return

    name  = LABEL_NAMES[label_idx]
    emoji = LABEL_EMOJIS[label_idx]
    color = LABEL_COLORS[label_idx]

    # ── RESULT BANNER ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:rgba(0,0,0,0.3); border:2px solid {color}; border-radius:14px;
                padding:1.6rem 2rem; margin:1.2rem 2rem; display:flex;
                align-items:center; justify-content:space-between;'>
      <div>
        <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.12em;
                    text-transform:uppercase; color:{color}; margin-bottom:4px;'>Detected Drink</div>
        <div style='font-family:"Sora",sans-serif; font-size:2.2rem; font-weight:700;
                    color:{color}; line-height:1.1;'>{emoji} {name}</div>
        <div style='font-size:0.82rem; color:{C["text_mid"]}; margin-top:6px;'>
          Confidence: <span style='color:{color}; font-family:JetBrains Mono,monospace;
          font-weight:600;'>{confidence*100:.1f}%</span>
          &nbsp;·&nbsp; Source: <span style='color:{C["text"]}; font-family:JetBrains Mono,monospace;'>
          {source_name}</span>
        </div>
      </div>
      <div style='font-size:4rem;'>{emoji}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── METRICS ──────────────────────────────────────────────────────────
    t_dur = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction", name.split()[0] + " " + emoji)
    m2.metric("Confidence", f"{confidence*100:.1f}%")
    m3.metric("Duration", f"{t_dur:.1f} s")
    m4.metric("Windows analysed", str(int(sum(counts))))

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Per-class probabilities ───────────────────────────────────────────
    prob_cols = st.columns(3)
    for i, (col, lname, lcolor) in enumerate(zip(prob_cols, LABEL_NAMES, LABEL_COLORS)):
        p = float(probas[:, i].mean())
        bw = int(p * 100)
        col.markdown(f"""
        <div class='cs-card' style='border-color:{lcolor if i==label_idx else C["border"]};'>
          <div style='font-size:0.72rem; color:{C["text_dim"]}; font-weight:700;
                      letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;'>{lname}</div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:1.4rem;
                      color:{lcolor}; font-weight:600;'>{p*100:.1f}%</div>
          <div style='background:{C["border"]}; border-radius:99px; height:6px; margin-top:8px; overflow:hidden;'>
            <div style='width:{bw}%; background:{lcolor}; height:100%; border-radius:99px;'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────
    tabs = st.tabs(["📡  Axes Signal","📊  Magnitude","🗳️  Vote Distribution","📋  Raw Data"])

    with tabs[0]:
        st.plotly_chart(plot_axes(df, title=f"Accelerometer · {source_name}"),
                        use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plot_magnitude(df), use_container_width=True)
        mag = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mean |mag|", f"{mag.mean():.3f}")
        mc2.metric("Max |mag|", f"{mag.max():.3f}")
        mc3.metric("Std |mag|", f"{mag.std():.3f}")

    with tabs[2]:
        st.plotly_chart(plot_votes(counts), use_container_width=True)
        st.markdown(f"""
        <div style='font-size:0.82rem; color:{C["text_mid"]}; padding:0 0.5rem;'>
          Each <strong style='color:{C["text"]};'>window</strong> = {WIN_SEC:.0f}s of motion,
          sliding every 1s. Majority vote across {int(sum(counts))} windows
          → <strong style='color:{color};'>{name}</strong>.
        </div>""", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown(f'<div class="cs-label">Raw Signal Data ({len(df)} rows)</div>',
                    unsafe_allow_html=True)
        disp = df.copy()
        disp["timestamp_s"] = (disp["timestamp"] - disp["timestamp"].iloc[0]) / 1000.0
        st.dataframe(disp[["timestamp_s","x","y","z"]].rename(columns={"timestamp_s":"time (s)"}),
                     use_container_width=True, height=320, hide_index=True)
        st.download_button("⬇  Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="drinkClassifier_signal.csv", mime="text/csv")


if __name__ == "__main__":
    main()