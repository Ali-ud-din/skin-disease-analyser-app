from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ------------------------------------------------------------------ config
HERE = Path(__file__).parent
MODEL_PATH = HERE / "skin_cancer_model_quant.tflite"
DATASET_URL = "https://drive.google.com/drive/folders/1lAgzbmSgU8sM3KqyoidOlLDa73Zc7adj?usp=sharing"
CLASS_NAMES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Light Diseases and Disorders of Pigmentation',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections',
]
ICONS = ["🧴", "⚠️", "🎨", "🍄", "🦠"]
TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(page_title="DermaScan · Spatial", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ spatial UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
section[data-testid="stSidebar"] {display:none;}

/* depth backdrop: soft glowing orbs behind everything */
.stApp {
  background:
    radial-gradient(900px 600px at 8% -5%, rgba(99,102,241,.45), transparent 60%),
    radial-gradient(800px 600px at 95% 5%, rgba(20,184,166,.35), transparent 60%),
    radial-gradient(900px 700px at 50% 110%, rgba(236,72,153,.28), transparent 60%),
    #0b1020;
  background-attachment: fixed;
}
.block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}

/* glass material */
.glass, div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, rgba(255,255,255,.10), rgba(255,255,255,.04)) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  border-radius: 28px !important;
  backdrop-filter: blur(26px) saturate(160%);
  -webkit-backdrop-filter: blur(26px) saturate(160%);
  box-shadow: 0 24px 60px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.22);
}
.glass {padding: 1.5rem 1.7rem; margin-bottom: 1.1rem; transition: transform .35s ease, box-shadow .35s ease;}
.glass:hover {transform: translateY(-4px) perspective(900px) rotateX(1.2deg);
              box-shadow: 0 34px 80px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.28);}
div[data-testid="stVerticalBlockBorderWrapper"] {padding: .6rem .8rem;}

/* floating nav */
.nav {display:flex; align-items:center; justify-content:space-between; gap:1rem;
      padding:.7rem 1.2rem; border-radius:999px; margin-bottom:1.4rem;}
.logo {font-weight:700; font-size:1.05rem; letter-spacing:.2px;}
.logo span {display:inline-block; width:30px; height:30px; line-height:30px; text-align:center;
            border-radius:10px; margin-right:.55rem; background:linear-gradient(135deg,#6366f1,#14b8a6);}
.navlinks {display:flex; gap:.5rem; flex-wrap:wrap;}
.chip {padding:.28rem .8rem; border-radius:999px; font-size:.78rem; font-weight:500;
       background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.14); opacity:.92;}

/* hero */
.hero h1 {font-size:2.7rem; font-weight:700; line-height:1.1; margin:0 0 .6rem;
          background:linear-gradient(90deg,#fff,#a5b4fc 55%,#5eead4); -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;}
.hero p {font-size:1.05rem; opacity:.8; max-width:640px; margin:0 0 1.1rem;}

/* result */
.ring {width:150px; height:150px; border-radius:50%; display:flex; align-items:center; justify-content:center;
       background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.10) 0);
       box-shadow: 0 0 40px var(--c); flex:none;}
.ring div {width:118px; height:118px; border-radius:50%; background:#0f1630; display:flex; flex-direction:column;
           align-items:center; justify-content:center; font-size:1.9rem; font-weight:700;}
.ring small {font-size:.7rem; font-weight:500; opacity:.65; margin-top:-2px;}
.res-row {display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap;}
.eyebrow {font-size:.75rem; text-transform:uppercase; letter-spacing:.14em; opacity:.6;}
.pred {font-size:1.5rem; font-weight:700; line-height:1.25; margin:.2rem 0 .5rem;}
.badge {display:inline-block; padding:.25rem .85rem; border-radius:999px; font-size:.8rem; font-weight:600;}
.b-hi {background:rgba(34,197,94,.2); color:#86efac; border:1px solid rgba(34,197,94,.5);}
.b-mid {background:rgba(245,158,11,.2); color:#fcd34d; border:1px solid rgba(245,158,11,.5);}
.b-lo {background:rgba(239,68,68,.2); color:#fca5a5; border:1px solid rgba(239,68,68,.5);}
.alert {border-radius:18px; padding:.8rem 1.1rem; margin-top:1rem; font-size:.92rem;
        background:rgba(239,68,68,.14); border:1px solid rgba(239,68,68,.45);}

/* probability bars */
.bar {margin:.7rem 0;}
.bar-top {display:flex; justify-content:space-between; gap:1rem; font-size:.88rem; margin-bottom:.3rem;}
.bar-top b {font-weight:600;}
.track {height:10px; border-radius:99px; background:rgba(255,255,255,.1); overflow:hidden;}
.fill {height:100%; border-radius:99px; background:linear-gradient(90deg,#6366f1,#14b8a6);}
.fill.top {background:linear-gradient(90deg,#f472b6,#fb923c);}

/* class tiles */
.tiles {display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:.9rem;}
.tile {padding:1rem 1.1rem; border-radius:22px; background:rgba(255,255,255,.07);
       border:1px solid rgba(255,255,255,.13); transition:transform .3s;}
.tile:hover {transform: translateY(-5px) scale(1.02);}
.tile .e {font-size:1.6rem;} .tile .t {font-size:.86rem; margin-top:.3rem; opacity:.9; line-height:1.35;}

/* widgets */
section[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.06) !important; border:1.5px dashed rgba(165,180,252,.55) !important;
  border-radius:22px !important;}
div[data-baseweb="tab-list"] {gap:.4rem; background:rgba(255,255,255,.07); padding:.35rem; border-radius:999px;
  width:fit-content; border:1px solid rgba(255,255,255,.12); backdrop-filter: blur(20px);}
button[data-baseweb="tab"] {border-radius:999px !important; padding:.4rem 1.2rem !important; height:auto !important;}
button[data-baseweb="tab"][aria-selected="true"] {background:linear-gradient(135deg,#6366f1,#14b8a6) !important; color:#fff !important;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none;}
.stLinkButton a, .stDownloadButton button {border-radius:999px !important; font-weight:600 !important;
  background:linear-gradient(135deg,#6366f1,#14b8a6) !important; color:#fff !important; border:none !important;
  box-shadow:0 10px 30px rgba(99,102,241,.4);}
.stImage img {border-radius:20px;}
.foot {text-align:center; opacity:.55; font-size:.82rem; margin-top:2rem;}
@media (max-width:700px){ .hero h1{font-size:2rem;} .nav{border-radius:24px; flex-direction:column; align-items:flex-start;} }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model…")
def load_interpreter():
    it = Interpreter(model_path=str(MODEL_PATH))
    it.allocate_tensors()
    return it


def predict(it, img: Image.Image) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    img = ImageOps.exif_transpose(img).convert("RGB").resize((w, h))
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)   # raw 0-255
    if inp["dtype"] != np.float32:
        scale, zero = inp["quantization"]
        x = (x / scale + zero).astype(inp["dtype"])
    it.set_tensor(inp["index"], x)
    it.invoke()
    p = it.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] != np.float32:
        scale, zero = out["quantization"]
        p = (p - zero) * scale
    return p / p.sum()


try:
    it = load_interpreter()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH.name}`: {e}")
    st.stop()
if int(it.get_output_details()[0]["shape"][-1]) != len(CLASS_NAMES):
    st.error("Model output size does not match the class list in app.py.")
    st.stop()
SIZE = int(it.get_input_details()[0]["shape"][1])

# ------------------------------------------------------------------ nav + hero
st.markdown(f"""
<div class="glass nav">
  <div class="logo"><span>🩺</span>DermaScan</div>
  <div class="navlinks"><div class="chip">{len(CLASS_NAMES)} classes</div>
  <div class="chip">{SIZE}×{SIZE} input</div><div class="chip">EfficientNet · TFLite</div></div>
</div>
<div class="glass hero">
  <h1>Skin condition analysis,<br>in a glance.</h1>
  <p>Drop in a photo and the model ranks which of five skin-condition categories it most resembles —
  with a confidence reading and a full probability breakdown.</p>
</div>
""", unsafe_allow_html=True)

tab_a, tab_b, tab_c = st.tabs(["Analyze", "Batch", "About & Dataset"])

# ------------------------------------------------------------------ analyze
with tab_a:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("##### Upload image")
            up = st.file_uploader("Skin image", type=TYPES, label_visibility="collapsed")
            if up:
                image = Image.open(up)
                st.image(image, caption=up.name, use_container_width=True)
            else:
                st.caption("JPG, PNG or WEBP · a clear, well-lit close-up works best.")
    with right:
        if not up:
            st.markdown('<div class="glass"><div class="eyebrow">Result</div>'
                        '<div class="pred">Waiting for an image…</div>'
                        '<span class="badge b-mid">Upload a photo to begin</span></div>',
                        unsafe_allow_html=True)
        else:
            with st.spinner("Analysing…"):
                probs = predict(it, image)
            order = np.argsort(probs)[::-1]
            best, conf = int(order[0]), float(probs[order[0]]) * 100
            cls, color, label = (("b-hi", "#22c55e", "High confidence") if conf >= 70 else
                                 ("b-mid", "#f59e0b", "Moderate confidence") if conf >= 45 else
                                 ("b-lo", "#ef4444", "Low confidence"))
            extra = ""
            if "Malignant" in CLASS_NAMES[best]:
                extra += ('<div class="alert">⚠️ This category includes potentially malignant lesions. '
                          'Please consult a dermatologist.</div>')
            if conf < 45:
                extra += '<div class="alert">The model is unsure. Try a clearer, closer, well-lit photo.</div>'
            st.markdown(f"""
            <div class="glass"><div class="res-row">
              <div class="ring" style="--p:{conf:.1f};--c:{color}"><div>{conf:.0f}%<small>confidence</small></div></div>
              <div style="flex:1;min-width:220px">
                <div class="eyebrow">Most likely category</div>
                <div class="pred">{ICONS[best]} {CLASS_NAMES[best]}</div>
                <span class="badge {cls}">{label}</span>
              </div></div>{extra}</div>""", unsafe_allow_html=True)

            bars = ""
            for rank, i in enumerate(order):
                pc = float(probs[i]) * 100
                bars += (f'<div class="bar"><div class="bar-top"><span>{ICONS[i]} {CLASS_NAMES[i]}</span>'
                         f'<b>{pc:.1f}%</b></div><div class="track"><div class="fill {"top" if rank == 0 else ""}" '
                         f'style="width:{max(pc, 0.8):.1f}%"></div></div></div>')
            st.markdown(f'<div class="glass"><div class="eyebrow">Probability breakdown</div>{bars}</div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ batch
with tab_b:
    with st.container(border=True):
        st.markdown("##### Classify many images at once")
        files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True,
                                 key="batch", label_visibility="collapsed")
        if files:
            rows, bar = [], st.progress(0.0)
            for i, f in enumerate(files):
                try:
                    p = predict(it, Image.open(f)); k = int(np.argmax(p))
                    rows.append({"File": f.name, "Prediction": CLASS_NAMES[k],
                                 "Confidence (%)": round(float(p[k]) * 100, 2)})
                except Exception as e:
                    rows.append({"File": f.name, "Prediction": f"Error: {e}", "Confidence (%)": None})
                bar.progress((i + 1) / len(files))
            res = pd.DataFrame(rows)
            st.dataframe(res, hide_index=True, use_container_width=True)
            st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ------------------------------------------------------------------ about
with tab_c:
    tiles = "".join(f'<div class="tile"><div class="e">{ICONS[i]}</div><div class="t">{c}</div></div>'
                    for i, c in enumerate(CLASS_NAMES))
    st.markdown(f'<div class="glass"><div class="eyebrow">Categories the model can recognise</div>'
                f'<div class="tiles" style="margin-top:.8rem">{tiles}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass"><div class="eyebrow">Dataset</div>'
                '<div class="pred">Get the images</div>'
                '<p style="opacity:.8">Download the dataset, then upload any image from it to test the model.</p></div>',
                unsafe_allow_html=True)
    st.link_button("⬇️ Open dataset (Google Drive)", DATASET_URL)

st.markdown('<div class="foot">⚠️ Educational project only — not a medical device and not a diagnosis. '
            'Always consult a qualified doctor.</div>', unsafe_allow_html=True)
