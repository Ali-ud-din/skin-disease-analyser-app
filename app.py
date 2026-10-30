import base64
import io
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
SHORT = ["Acne & Rosacea", "Actinic Keratosis / Malignant", "Pigmentation Disorders",
         "Fungal Infections", "Warts & Viral Infections"]
ICONS = ["🧴", "⚠️", "🎨", "🍄", "🦠"]
TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(page_title="DermaScan · Spatial", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ themes
THEMES = {
    "dark": dict(
        SCENE="radial-gradient(900px 600px at 15% 10%, rgba(47,85,255,.35), transparent 60%),"
              "radial-gradient(900px 700px at 90% 90%, rgba(120,60,200,.30), transparent 60%),"
              "radial-gradient(700px 500px at 60% 0%, rgba(30,160,200,.20), transparent 60%), #070b16",
        WIN="linear-gradient(180deg, rgba(40,48,68,.62) 0%, rgba(30,36,54,.58) 70%, rgba(120,130,150,.28) 100%)",
        WINBD="rgba(255,255,255,.14)", WINSH="rgba(0,0,0,.55)",
        CARD="linear-gradient(160deg, rgba(255,255,255,.075), rgba(255,255,255,.025))",
        CARDBD="rgba(255,255,255,.12)", TEXT="#f1f4ff", MUTED="rgba(241,244,255,.62)",
        ACC="#2f55ff", ACC2="#6d8bff", GLOW="rgba(80,120,255,.55)", TRACK="rgba(255,255,255,.10)",
        PILL="rgba(255,255,255,.09)", PILLON="rgba(255,255,255,.22)", INNER="rgba(8,12,24,.55)",
        HI="#4ade80", MID="#fbbf24", LO="#f87171", ALERT="rgba(248,113,113,.14)", ALERTFG="#fecaca"),
    "light": dict(
        SCENE="radial-gradient(900px 600px at 15% 10%, rgba(99,130,255,.35), transparent 60%),"
              "radial-gradient(900px 700px at 90% 90%, rgba(244,114,182,.28), transparent 60%),"
              "radial-gradient(700px 500px at 60% 0%, rgba(45,212,191,.25), transparent 60%), #eef2fb",
        WIN="linear-gradient(180deg, rgba(255,255,255,.78) 0%, rgba(255,255,255,.62) 100%)",
        WINBD="rgba(255,255,255,.95)", WINSH="rgba(60,80,160,.25)",
        CARD="linear-gradient(160deg, rgba(255,255,255,.85), rgba(255,255,255,.5))",
        CARDBD="rgba(80,100,200,.14)", TEXT="#14193a", MUTED="rgba(20,25,58,.60)",
        ACC="#2f55ff", ACC2="#5b7bff", GLOW="rgba(47,85,255,.35)", TRACK="rgba(47,85,255,.12)",
        PILL="rgba(47,85,255,.08)", PILLON="rgba(47,85,255,.20)", INNER="rgba(255,255,255,.8)",
        HI="#15803d", MID="#b45309", LO="#b91c1c", ALERT="rgba(239,68,68,.10)", ALERTFG="#7f1d1d"),
}
DEFAULT_MODE = "dark"                                             # "dark" or "light"
_mode = str(st.query_params.get("theme", DEFAULT_MODE)).lower()    # or try  ?theme=light
T = THEMES.get(_mode, THEMES["dark"])

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] {font-family:'Inter',sans-serif;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}
section[data-testid="stSidebar"] {display:none;}
.stApp {background: __SCENE__; background-attachment: fixed; color: __TEXT__;}

/* the big frosted "window" */
.block-container {
  max-width: 1280px; margin: 1.4rem auto 2rem; padding: 1.6rem 2.2rem 2rem !important;
  background: __WIN__; border: 1px solid __WINBD__; border-radius: 40px;
  backdrop-filter: blur(40px) saturate(150%); -webkit-backdrop-filter: blur(40px) saturate(150%);
  box-shadow: 0 40px 100px __WINSH__, inset 0 1px 0 rgba(255,255,255,.25);
}
.stApp p, .stApp label, .stApp span, .stApp div {color: inherit;}

/* top bar */
.logo {width:44px; height:44px; border-radius:15px; display:flex; align-items:center; justify-content:center;
       font-weight:800; font-size:1.2rem; color:#fff; background:linear-gradient(135deg,__ACC__,__ACC2__);
       box-shadow:0 10px 26px __GLOW__;}
.cta-top {display:inline-flex; align-items:center; gap:.5rem; padding:.62rem 1.2rem; border-radius:999px;
          background:__ACC__; color:#fff !important; font-weight:600; font-size:.9rem; text-decoration:none;
          box-shadow:0 10px 28px __GLOW__; float:right; transition:transform .2s;}
.cta-top:hover {transform:translateY(-2px);}
button[data-variant="pills"] {
  border-radius:999px !important; background:__PILL__ !important; border:1px solid __CARDBD__ !important;
  padding:.4rem 1.4rem !important; transition:all .2s;}
button[data-variant="pills"] p {color:__TEXT__ !important; font-weight:500; font-size:.92rem;}
button[data-variant="pills"][data-selected="true"] {background:__PILLON__ !important; border-color:__ACC2__ !important;
  box-shadow:0 0 0 1px __ACC2__, 0 8px 22px __GLOW__;}
button[data-variant="pills"][data-selected="true"] p {font-weight:700;}
a.cta, a.cta-top, a.cta span, a.cta-top span {text-decoration:none !important; color:#fff !important;}

/* headings */
.big {font-size:3.6rem; font-weight:700; line-height:1.02; letter-spacing:-.02em; margin:.6rem 0 .5rem; color:__TEXT__;}
.sub {color:__MUTED__; font-size:.98rem; max-width:420px; margin-bottom:1rem;}
.sec {display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:.98rem; margin:.2rem 0 .8rem;}
.sec i {width:8px; height:8px; border-radius:50%; background:__ACC__; display:inline-block; box-shadow:0 0 10px __ACC__;}

/* glass cards */
.card {background:__CARD__; border:1px solid __CARDBD__; border-radius:26px; padding:1rem 1.15rem;
       backdrop-filter: blur(18px); box-shadow: inset 0 1px 0 rgba(255,255,255,.12);
       transition: transform .3s ease, box-shadow .3s ease;}
.card:hover {transform: translateY(-4px); box-shadow: 0 22px 50px __WINSH__, inset 0 1px 0 rgba(255,255,255,.18);}
.grid2 {display:grid; grid-template-columns:1fr 1fr; gap:.9rem;}
.lbl {font-size:.78rem; color:__MUTED__; display:flex; align-items:center; gap:.4rem; font-weight:500;}
.lbl b {font-weight:600; color:__TEXT__; font-size:.8rem;}
.val {font-size:1.45rem; font-weight:700; margin-top:.15rem; line-height:1.15;}
.val small {font-size:.75rem; font-weight:500; color:__MUTED__;}
.tilerow {display:flex; align-items:flex-end; justify-content:space-between; gap:.6rem; margin-top:.7rem; min-height:70px;}
.mbars {display:flex; align-items:flex-end; gap:7px; height:68px; flex:1;}
.mbars span {flex:1; border-radius:6px; background:__TRACK__; min-height:6px;}
.mbars span.on {background:linear-gradient(180deg,__ACC2__,__ACC__);}
.cap {background:__ACC__; color:#fff; border-radius:18px; padding:.75rem .7rem; text-align:center;
      font-weight:700; line-height:1.1; box-shadow:0 10px 24px __GLOW__; min-width:62px;}
.cap small {display:block; font-size:.62rem; font-weight:500; opacity:.85;}
.capg {background:__PILL__; color:__TEXT__; box-shadow:none; border:1px solid __CARDBD__;}
.wave {width:100%; height:68px;}

/* dataset card */
.rowi {display:flex; align-items:center; gap:.7rem; padding:.5rem .55rem; border-radius:16px;
       background:__PILL__; margin-top:.5rem; font-size:.84rem; font-weight:500;}
.rowi .e {width:30px; height:30px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:__INNER__;}
.cta {display:flex; justify-content:space-between; align-items:center; margin-top:.8rem; padding:.75rem 1.1rem;
      border-radius:16px; background:__ACC__; color:#fff !important; text-decoration:none; font-weight:600; font-size:.9rem;
      box-shadow:0 12px 28px __GLOW__; transition:transform .2s;}
.cta:hover {transform:translateY(-2px);}

/* hero stage */
.stage {position:relative; height:400px; display:flex; align-items:center; justify-content:center; margin-top:.4rem;}
.ring {position:absolute; border-radius:50%; border:1px solid __CARDBD__;}
.r1 {width:250px; height:250px; background:radial-gradient(circle, __GLOW__ 0%, transparent 70%); border-color:rgba(255,255,255,.25);}
.r2 {width:320px; height:320px; animation:pulse 5s ease-in-out infinite;}
.r3 {width:390px; height:390px; animation:pulse 5s ease-in-out infinite .8s; opacity:.6;}
@keyframes pulse {0%,100%{transform:scale(1);opacity:.55} 50%{transform:scale(1.04);opacity:.95}}
@keyframes floaty {0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)}}
.orb {position:relative; width:230px; height:230px; border-radius:50%; overflow:hidden; background:__INNER__;
      border:2px solid rgba(255,255,255,.55); box-shadow:0 0 70px __GLOW__, 0 30px 60px __WINSH__;
      display:flex; align-items:center; justify-content:center; animation:floaty 6s ease-in-out infinite;}
.orb img {width:100%; height:100%; object-fit:cover;}
.orb .ph {text-align:center; font-size:.85rem; color:__MUTED__; padding:1rem; line-height:1.4;}
.orb .ph b {display:block; font-size:2.4rem; margin-bottom:.3rem;}
.dot {position:absolute; width:22px; height:22px; border-radius:50%; background:rgba(255,255,255,.9);
      box-shadow:0 0 0 7px rgba(255,255,255,.22), 0 0 22px rgba(255,255,255,.8);}
.d1{top:46px; left:56%;} .d2{top:42%; left:21%;} .d3{bottom:78px; left:30%;} .d4{top:57%; right:20%;}
.float {position:absolute; left:2%; bottom:14px; width:190px; border-radius:22px; padding:.8rem .95rem;
        background:__CARD__; border:1px solid __CARDBD__; backdrop-filter:blur(22px);
        box-shadow:0 18px 40px __WINSH__; animation:floaty 7s ease-in-out infinite .5s;}
.float .t {font-size:.72rem; color:__MUTED__; font-weight:500;}
.float .n {font-size:.9rem; font-weight:700; margin:.1rem 0 .2rem; line-height:1.2;}
.float svg {width:100%; height:34px;}

/* bottom class cards */
.cls {display:grid; grid-template-columns:repeat(5,1fr); gap:.9rem;}
.cc {position:relative; text-align:center; padding:1rem .7rem .9rem;}
.cc .ic {width:64px; height:64px; margin:0 auto .6rem; border-radius:50%; display:flex; align-items:center; justify-content:center;
         font-size:1.9rem; background:__INNER__; box-shadow:inset 0 0 0 1px __CARDBD__;}
.cc .nm {font-size:.8rem; font-weight:600; line-height:1.25; min-height:2.1em;}
.cc .pc {font-size:1.25rem; font-weight:700; margin:.35rem 0 .45rem;}
.cc .tr {height:6px; border-radius:99px; background:__TRACK__; overflow:hidden;}
.cc .tr div {height:100%; border-radius:99px; background:linear-gradient(90deg,__ACC__,__ACC2__);}
.cc.top {background:linear-gradient(160deg, __ACC__, __ACC2__); border-color:transparent; color:#fff; box-shadow:0 18px 44px __GLOW__;}
.cc.top .tr {background:rgba(255,255,255,.3);} .cc.top .tr div {background:#fff;}
.cc.top .ic {background:rgba(255,255,255,.2);}
.tag {position:absolute; top:10px; right:12px; font-size:.62rem; font-weight:700; background:#fff; color:__ACC__ !important;
      border-radius:999px; padding:.12rem .55rem;}

.alert {border-radius:18px; padding:.7rem 1rem; margin:.6rem 0; font-size:.88rem; background:__ALERT__;
        border:1px solid rgba(248,113,113,.5); color:__ALERTFG__;}
.foot {text-align:center; color:__MUTED__; font-size:.8rem; margin-top:1.6rem;}

/* widgets */
div[data-testid="stFileUploaderDropzone"] {background:__CARD__ !important; border:1.5px dashed __ACC2__ !important;
  border-radius:22px !important;}
.stDownloadButton button {border-radius:999px !important; background:__ACC__ !important; color:#fff !important; border:none !important;}
div[data-testid="stVerticalBlockBorderWrapper"] {border:none !important; background:transparent !important;}

@media (max-width: 900px){ .big{font-size:2.5rem;} .cls{grid-template-columns:repeat(2,1fr);} .grid2{grid-template-columns:1fr;}
  .block-container{padding:1rem !important; border-radius:26px;} .stage{height:340px;} }
</style>
"""
for _k, _v in T.items():
    CSS = CSS.replace(f"__{_k}__", _v)
st.markdown(CSS, unsafe_allow_html=True)


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
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)    # raw 0-255
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


def b64_thumb(img: Image.Image, size=460) -> str:
    im = ImageOps.exif_transpose(img).convert("RGB")
    im = ImageOps.fit(im, (size, size))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def wave(vals, w=200, h=60):
    xs = np.linspace(4, w - 4, len(vals)); m = max(max(vals), 1e-6)
    ys = [h - 6 - (v / m) * (h - 14) for v in vals]
    d = f"M{xs[0]:.1f},{ys[0]:.1f}"
    for i in range(1, len(xs)):
        cx = (xs[i - 1] + xs[i]) / 2
        d += f" C{cx:.1f},{ys[i-1]:.1f} {cx:.1f},{ys[i]:.1f} {xs[i]:.1f},{ys[i]:.1f}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><path d="{d}" fill="none" '
            f'stroke="{T["ACC2"]}" stroke-width="2.5" stroke-linecap="round"/></svg>')


try:
    it = load_interpreter()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH.name}`: {e}")
    st.stop()
if int(it.get_output_details()[0]["shape"][-1]) != len(CLASS_NAMES):
    st.error("Model output size does not match the class list in app.py.")
    st.stop()
SIZE = int(it.get_input_details()[0]["shape"][1])

# ------------------------------------------------------------------ top bar
c1, c2, c3 = st.columns([0.5, 3, 1.5], vertical_alignment="center")
c1.markdown('<div class="logo">D</div>', unsafe_allow_html=True)
with c2:
    view = st.pills("View", ["Analyze", "Batch"], default="Analyze", label_visibility="collapsed") or "Analyze"
c3.markdown(f'<a class="cta-top" href="{DATASET_URL}" target="_blank">⬇ Dataset</a>', unsafe_allow_html=True)

# ------------------------------------------------------------------ analyze
if view == "Analyze":
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="big">Skin<br>Analysis</div>'
                    '<div class="sub">Upload a photo and the model estimates which of five skin-condition '
                    'categories it most resembles.</div>', unsafe_allow_html=True)
        up = st.file_uploader("Skin image", type=TYPES, label_visibility="collapsed")

        probs = None
        if up:
            image = Image.open(up)
            probs = predict(it, image)
            order = np.argsort(probs)[::-1]
            best, conf = int(order[0]), float(probs[order[0]]) * 100
            orb = f'<img src="data:image/jpeg;base64,{b64_thumb(image)}">'
            fl_t, fl_n, fl_w = "Top match", SHORT[best], wave(list(probs))
        else:
            orb = '<div class="ph"><b>＋</b>Upload a skin image<br>to begin</div>'
            fl_t, fl_n, fl_w = "Waiting", "No image yet", wave([.2] * 5)
        alert = ""
        if probs is not None:
            if "Malignant" in CLASS_NAMES[best]:
                alert += '<div class="alert">⚠️ This category includes potentially malignant lesions. Please consult a dermatologist.</div>'
            if conf < 45:
                alert += '<div class="alert">The model is unsure — try a clearer, closer, well-lit photo.</div>'
        st.markdown(f"""{alert}
        <div class="stage"><div class="ring r3"></div><div class="ring r2"></div><div class="ring r1"></div>
          <div class="orb">{orb}</div>
          <span class="dot d1"></span><span class="dot d2"></span><span class="dot d3"></span><span class="dot d4"></span>
          <div class="float"><div class="t">{fl_t}</div><div class="n">{fl_n}</div>{fl_w}</div>
        </div>""", unsafe_allow_html=True)

    with right:
        if probs is not None:
            lvl, col = (("High", T["HI"]) if conf >= 70 else ("Moderate", T["MID"]) if conf >= 45 else ("Low", T["LO"]))
            sec = int(order[1]); mx = float(probs.max())
            bars = "".join(f'<span class="{"on" if i == best else ""}" style="height:{max(probs[i]/mx*100, 8):.0f}%"></span>'
                           for i in range(len(probs)))
            conf_v, cap_v = f"{conf:.1f}<small> %</small>", f"{conf:.0f}<small>% top</small>"
            cert_v, cert_c = lvl, col
            run_v = f"{SHORT[sec]}"; run_p = f"{probs[sec]*100:.1f}%"
        else:
            bars = "".join('<span style="height:30%"></span>' for _ in CLASS_NAMES)
            conf_v, cap_v, cert_v, cert_c, run_v, run_p = "—", "—", "—", T["MUTED"], "—", "—"

        wv = wave(list(probs) if probs is not None else [.2] * 5, 220, 66)
        rows = "".join(f'<div class="rowi"><div class="e">{ICONS[i]}</div>{SHORT[i]}</div>' for i in range(5))
        r1, r2 = st.columns([1.55, 1], gap="medium")
        with r1:
            st.markdown(f"""
            <div class="sec"><i></i>Prediction</div>
            <div class="grid2">
              <div class="card"><div class="lbl">📈 <b>Confidence</b></div><div class="val">{conf_v}</div>
                <div class="tilerow"><div class="mbars">{bars}</div><div class="cap">{cap_v}</div></div></div>
              <div class="card"><div class="lbl">🎯 <b>Certainty</b></div><div class="val" style="color:{cert_c}">{cert_v}</div>
                <div class="tilerow">{wv}</div></div>
              <div class="card"><div class="lbl">🥈 <b>Runner-up</b></div><div class="val" style="font-size:1.05rem">{run_v}</div>
                <div class="tilerow" style="min-height:40px"><div class="cap capg" style="margin-left:auto">{run_p}</div></div></div>
              <div class="card"><div class="lbl">🖼️ <b>Model input</b></div><div class="val">{SIZE}×{SIZE}<small> RGB</small></div>
                <div class="tilerow" style="min-height:40px"><div class="lbl">EfficientNet · TFLite</div></div></div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="sec"><i></i>Dataset</div>
            <div class="card"><div class="lbl"><b>5 categories</b></div>{rows}
              <a class="cta" href="{DATASET_URL}" target="_blank"><span>Open dataset</span><span>↗</span></a></div>""",
                        unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:1.1rem"><i></i>Class probabilities</div>', unsafe_allow_html=True)
        cards = ""
        for i in range(5):
            pc = float(probs[i]) * 100 if probs is not None else 0
            top = probs is not None and i == best
            cards += (f'<div class="card cc {"top" if top else ""}">{"<span class=tag>TOP</span>" if top else ""}'
                      f'<div class="ic">{ICONS[i]}</div><div class="nm">{SHORT[i]}</div>'
                      f'<div class="pc">{pc:.1f}%</div><div class="tr"><div style="width:{max(pc, 1.5):.1f}%"></div></div></div>')
        st.markdown(f'<div class="cls">{cards}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ batch
else:
    st.markdown('<div class="big" style="font-size:2.6rem">Batch<br>Classification</div>'
                '<div class="sub">Upload many images (for example from the dataset) and download the results.</div>',
                unsafe_allow_html=True)
    files = st.file_uploader("Images", type=TYPES, accept_multiple_files=True, key="batch", label_visibility="collapsed")
    if files:
        rows, bar = [], st.progress(0.0)
        for i, f in enumerate(files):
            try:
                p = predict(it, Image.open(f)); k = int(np.argmax(p))
                rows.append({"File": f.name, "Prediction": CLASS_NAMES[k], "Confidence (%)": round(float(p[k]) * 100, 2)})
            except Exception as e:
                rows.append({"File": f.name, "Prediction": f"Error: {e}", "Confidence (%)": None})
            bar.progress((i + 1) / len(files))
        res = pd.DataFrame(rows)
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(), "predictions.csv", "text/csv")

st.markdown('<div class="foot">⚠️ Educational project only — not a medical device and not a diagnosis. '
            'Always consult a qualified doctor.</div>', unsafe_allow_html=True)
