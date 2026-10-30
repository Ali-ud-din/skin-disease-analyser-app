from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:                                   # light runtime (recommended)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                    # fallback if full TensorFlow is installed
    from tensorflow.lite.python.interpreter import Interpreter

# ---------------------------------------------------------------- config
HERE = Path(__file__).parent
MODEL_PATH = HERE / "skin_cancer_model_quant.tflite"
CLASS_NAMES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Light Diseases and Disorders of Pigmentation',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections',
]
DATASET_URL = "https://drive.google.com/drive/folders/1lAgzbmSgU8sM3KqyoidOlLDa73Zc7adj?usp=sharing"

st.set_page_config(page_title="DermaScan AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg,#0f766e 0%,#2563eb 100%);
       padding: 2rem 2.2rem; border-radius: 20px; color:#fff; margin-bottom:1.2rem;}
.hero h1 {margin:0; font-size:2.2rem; color:#fff;}
.hero p {margin:.4rem 0 0; opacity:.92; font-size:1.05rem;}
.card {background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.25);
       border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1rem;}
.pred-label {font-size:1.6rem; font-weight:700;}
.pill {display:inline-block; padding:.2rem .8rem; border-radius:999px;
       font-size:.85rem; font-weight:600; color:#fff;}
.pill.hi{background:#16a34a;} .pill.mid{background:#d97706;} .pill.lo{background:#dc2626;}
.footer {text-align:center; opacity:.65; font-size:.85rem; margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_interpreter():
    it = Interpreter(model_path=str(MODEL_PATH))
    it.allocate_tensors()
    return it


def predict(it, img: Image.Image) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    img = ImageOps.exif_transpose(img).convert("RGB").resize((w, h))
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)   # raw 0-255, model normalises itself
    if inp["dtype"] != np.float32:                              # quantised-input safety
        scale, zero = inp["quantization"]
        x = (x / scale + zero).astype(inp["dtype"])
    it.set_tensor(inp["index"], x)
    it.invoke()
    p = it.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] != np.float32:
        scale, zero = out["quantization"]
        p = (p - zero) * scale
    return p / p.sum()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📂 Dataset")
    st.write("Download the dataset, then upload any image from it to test the model.")
    st.link_button("⬇️ Open Dataset (Google Drive)", DATASET_URL,
                   use_container_width=True, type="primary")
    st.divider()

try:
    it = load_interpreter()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH.name}`: {e}")
    st.stop()

n_classes = int(it.get_output_details()[0]["shape"][-1])
if n_classes != len(CLASS_NAMES):
    st.error(f"Model outputs {n_classes} classes but {len(CLASS_NAMES)} names are set in app.py.")
    st.stop()

with st.sidebar:
    st.header("ℹ️ About")
    size = it.get_input_details()[0]["shape"][1]
    st.write(f"**Classes:** {n_classes}  \n**Input size:** {size}×{size} RGB  \n"
             "**Model:** EfficientNet (TFLite)")
    with st.expander("All classes"):
        for c in CLASS_NAMES:
            st.markdown(f"- {c}")

st.markdown("""
<div class="hero"><h1>🩺 DermaScan AI</h1>
<p>Upload a skin image and the model estimates which category it most resembles.</p></div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Single image", "🗂️ Batch from dataset"])
TYPES = ["jpg", "jpeg", "png", "webp"]

with tab1:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        up = st.file_uploader("Upload a skin image", type=TYPES)
        if up:
            image = Image.open(up)
            st.image(image, caption="Uploaded image", use_container_width=True)
        else:
            st.markdown('<div class="card">👆 Upload a JPG/PNG to begin, or use the '
                        '<b>Dataset</b> button in the sidebar to get sample images.</div>',
                        unsafe_allow_html=True)
    with right:
        if up:
            with st.spinner("Analysing…"):
                probs = predict(it, image)
            order = np.argsort(probs)[::-1]
            best, conf = int(order[0]), float(probs[order[0]]) * 100
            lvl, txt = (("hi", "High confidence") if conf >= 70 else
                        ("mid", "Moderate confidence") if conf >= 45 else
                        ("lo", "Low confidence"))
            st.markdown(f"""<div class="card"><div style="opacity:.7">Most likely category</div>
            <div class="pred-label">{CLASS_NAMES[best]}</div><br>
            <span class="pill {lvl}">{txt} · {conf:.1f}%</span></div>""", unsafe_allow_html=True)
            if "Malignant" in CLASS_NAMES[best]:
                st.error("⚠️ This category includes potentially malignant lesions. "
                         "Please consult a dermatologist.")
            if conf < 45:
                st.warning("The model is unsure. Try a clearer, closer, well-lit photo.")
            df = pd.DataFrame({"Condition": [CLASS_NAMES[i] for i in order],
                               "Probability (%)": [round(float(probs[i]) * 100, 2) for i in order]})
            st.subheader("All probabilities")
            st.bar_chart(df, x="Condition", y="Probability (%)", horizontal=True)
            st.dataframe(df, hide_index=True, use_container_width=True)

with tab2:
    files = st.file_uploader("Upload multiple images", type=TYPES,
                             accept_multiple_files=True, key="batch")
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

st.markdown('<div class="footer">⚠️ Educational project only — not a medical device. '
            'Not a diagnosis; consult a qualified doctor.</div>', unsafe_allow_html=True)
