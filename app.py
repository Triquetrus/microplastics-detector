# app.py
"""
Microplastic Detector — Final Presentation-Ready Streamlit App
Features included:
 - Dark / Light (sky-blue) theme toggle without losing detection state
 - Centered original image + "Detect" flow: original initially centered; after detect it moves left and annotated appears right
 - KPIs displayed cleanly; detection table placed below KPIs to avoid overlap
 - Visualizations (Donut, Confidence bar, Histogram) inside expanders to reduce clutter
 - Session history with thumbnails (last 5 runs)
 - Download buttons (annotated image + CSV)
 - Clear / Reset controls
 - Robust handling of missing model, no-detections, and various ultralytics versions
 - Uses use_container_width=True (no deprecated use_column_width)
 - Theme-specific color palettes for charts and UI
Instructions:
 - Update DEFAULT_MODEL_PATH to point to your YOLOv8 weights (absolute path)
 - Run: streamlit run app.py
"""

import os
import io
import time
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import streamlit as st

# Attempt to import ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    YOLO_IMPORT_ERROR = str(e)


# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Microplastic Detector", layout="wide", page_icon="🧫")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

header_html = f"""
<div class="card" style="text-align:center; padding:20px;">
  <div style="font-size:32px;">🔬 <b>MicroDetect</b></div>
  <div style="margin-top:6px; font-size:15px;">
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Default: change this to your absolute path if different
DEFAULT_MODEL_PATH = r"D:\microplastic_project\runs\detect\microplastic_train2\weights\best.pt"

APP_TITLE = "⚛Microplastic Detector"
APP_SUBTITLE = "📸Upload an image, detect microplastics using your YOLOv8 model, and present easy-to-understand visualizations."

# ----------------------------- UTILITIES -----------------------------
def color_map_for_values(n: int, cmap_name: str = "viridis") -> List[str]:
    """Return n hex colors from a matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    return [cm.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

def safe_filename(f) -> str:
    try:
        return getattr(f, "name", "uploaded_image")
    except Exception:
        return "uploaded_image"

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ----------------------------- SESSION STATE INIT -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # internal key, maps to sidebar radio

if "model_path" not in st.session_state:
    st.session_state.model_path = DEFAULT_MODEL_PATH

if "image" not in st.session_state:
    st.session_state.image = None  # PIL Image

if "annotated" not in st.session_state:
    st.session_state.annotated = None  # PIL Image

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()  # detections DataFrame

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {filename,timestamp,df,annotated,elapsed}

if "model_loaded_for" not in st.session_state:
    st.session_state.model_loaded_for = None  # model path loaded

    # ---------------- SESSION STATE STAGE ----------------
if "stage" not in st.session_state:
    st.session_state.stage = "initial"  # possible values: initial, uploaded, detected


# ----------------------------- SIDEBAR: SETTINGS & THEME -----------------------------
st.sidebar.markdown("## ⚙️ Settings")

# Theme toggle (use radio with keys so state persists after reruns)
theme_choice = st.sidebar.radio("Theme", ["🌑 Dark", "🌕 Light (Sky Blue)"], index=0, key="theme_radio")
# Map to internal theme label
st.session_state.theme = "dark" if theme_choice.startswith("🌑") else "light"

# Model path (absolute)
model_path_input = st.sidebar.text_input("Model path (absolute)", value=st.session_state.model_path, key="model_path_input")
st.session_state.model_path = model_path_input.strip()

device_choice = st.sidebar.selectbox("Device", ["GPU", "CPU"], index=0, key="device_choice")
device = 0 if str(device_choice).startswith("GPU") else "CPU"

conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01, key="conf_threshold")
max_display = st.sidebar.slider("Max detections shown in confidence plot", 5, 200, 60, key="max_display")
box_thickness = st.sidebar.slider("Box thickness (px)", 1, 8, 2, key="box_thickness")

st.sidebar.markdown("---")
st.sidebar.markdown("Tips:")
st.sidebar.markdown("- Use a clear close-up image of the sample.")
st.sidebar.markdown("- Increase confidence threshold to reduce false positives.")
st.sidebar.markdown("- Use GPU if available for faster inference.")

# ----------------------------- THEME STYLING -----------------------------
# Theme palettes
if st.session_state.theme == "dark":
    PAGE_BG = "linear-gradient(135deg, #0f1724 0%, #2b2143 40%, #6e5b7b 100%)"
    CARD_BG = "rgba(255,255,255,0.04)"
    TEXT_COLOR = "#e6eef6"
    PRIMARY_COLOR = "#00f0ea"
    SECONDARY_COLOR = "#ff33cc"
    HIST_COLOR = "#ff33cc"
    PIE_CMAP = "viridis"
    CONF_CMAP = "plasma"
    MT_FIG_FACE = "#0f1724"
else:
    PAGE_BG = "linear-gradient(135deg, #e6f7ff 0%, #c7f0ff 50%, #a0e8ff 100%)"
    CARD_BG = "rgba(255,255,255,0.85)"
    TEXT_COLOR = "#07263b"
    PRIMARY_COLOR = "#0288d1"
    SECONDARY_COLOR = "#ff7ab6"
    HIST_COLOR = "#0077be"
    PIE_CMAP = "Spectral"
    CONF_CMAP = "coolwarm"
    MT_FIG_FACE = "#e6f7ff"

# Inject CSS
# Extra widget theming so Streamlit UI matches our dark/light toggle
if st.session_state.theme == "dark":
    WIDGET_BG = "#1e1e2f"
    WIDGET_TEXT = "#f5f5f5"
    WIDGET_ACCENT = PRIMARY_COLOR
else:
    WIDGET_BG = "#ffffff"
    WIDGET_TEXT = "#07263b"
    WIDGET_ACCENT = PRIMARY_COLOR

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAGE_BG}; color: {TEXT_COLOR}; }}
      .stButton>button, .stDownloadButton>button {{
          background-color: {WIDGET_BG};
          color: {WIDGET_TEXT};
          border: 1px solid {WIDGET_ACCENT};
          border-radius: 8px;
          padding: 0.4em 1em;
      }}
      .stButton>button:hover, .stDownloadButton>button:hover {{
          background-color: {WIDGET_ACCENT};
          color: white;
      }}
      div[data-baseweb="select"] > div {{
          background-color: {WIDGET_BG};
          color: {WIDGET_TEXT};
      }}
      .stSlider > div {{
          color: {WIDGET_TEXT};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Matplotlib config for theme
plt.rcParams.update({
    "figure.facecolor": MT_FIG_FACE,
    "axes.facecolor": MT_FIG_FACE,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": "#2b2143" if st.session_state.theme == "dark" else "#c7f0ff",
})

# ----------------------------- MODEL LOADING (cached) -----------------------------

@st.cache_resource
def _cached_load_model(path: str):
    if YOLO is None:
        raise ImportError("ultralytics YOLO not available")
    return YOLO(path)

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Rh85Qh47pdL763DkvnFsja_skovUU7eB"
MODEL_PATH = "best.pt"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv8 model..."):
        st.info("Downloading model from Google Drive...")
        os.system(f"wget {MODEL_URL} -O {MODEL_PATH}")

# Load the model using cached function
model = None
model_error = None
try:
    with st.spinner("Loading YOLO model..."):
        model = _cached_load_model(MODEL_PATH)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    model_error = str(e)
    st.sidebar.error(f"Failed to load model ❌: {model_error}")

"""
if using model from directly your machine use this otherwise use the above code
@st.cache_resource
def _cached_load_model(path: str):
    if YOLO is None:
        raise ImportError(f"ultralytics YOLO not available: {YOLO_IMPORT_ERROR}")
    return YOLO(path)

model = None
model_error = None
if st.session_state.model_path:
    if os.path.exists(st.session_state.model_path):
        try:
            # Only reload if path changed
            if st.session_state.model_loaded_for != st.session_state.model_path:
                with st.spinner("Loading YOLO model..."):
                    model = _cached_load_model(st.session_state.model_path)
                st.session_state.model_loaded_for = st.session_state.model_path
            else:
                # model already cached by load function; try to get from cache by calling
                model = _cached_load_model(st.session_state.model_path)
            st.sidebar.success("Model loaded ✅")
        except Exception as e:
            model_error = str(e)
            st.sidebar.error("Failed to load model ❌")
    else:
        model_error = f"Model file not found at: {st.session_state.model_path}"
        st.sidebar.warning("Model file path not found. Paste correct absolute path.")
else:
    st.sidebar.info("Paste the absolute path to your YOLO weights (best.pt) above.")"""

# ----------------------------- APP HEADER -----------------------------
st.markdown(f"<div class='card'><div class='title'>{APP_TITLE}</div><div class='muted'>{APP_SUBTITLE}</div></div>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
# Inject CSS for file uploader button
if theme_choice == "Light":
    st.markdown(
        """
        <style>
        div.stFileUploader button {
            background-color: #e0e0e0 !important;
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        div.stFileUploader button {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# File uploader with 10 MB limit
if st.session_state.stage in ["initial", "uploaded"]:
    uploaded = st.file_uploader("",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        if uploaded.size > 10 * 1024 * 1024:  # 10 MB limit
            st.error("File too large! Maximum allowed size is 10 MB.")
        else:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
                filename = safe_filename(uploaded)

                # If a previous detection exists, push to history
                if st.session_state.annotated is not None and not st.session_state.df.empty:
                    st.session_state.history.append({
                        "filename": st.session_state.history[-1]["filename"] if st.session_state.history else filename,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "df": st.session_state.df,
                        "annotated": st.session_state.annotated,
                        "elapsed": st.session_state.history[-1]["elapsed"] if st.session_state.history else 0,
                    })
                    st.session_state.history = st.session_state.history[-5:]  # keep last 5

                # Reset state for new image
                st.session_state.image = pil_img
                st.session_state.annotated = None
                st.session_state.df = pd.DataFrame()
                st.session_state.stage = "uploaded"  # move to uploaded stage

            except Exception as e:
                st.error(f"Failed to read uploaded image: {e}")



# Helper: run inference robustly
def run_yolo_inference(pil_img: Image.Image, model_obj: Any, conf: float, device_choice: Any, box_thickness_px: int = 2) -> Tuple[Image.Image, pd.DataFrame]:
    """
    Runs model on PIL image and returns annotated PIL image and detections DataFrame.
    DF columns: class, confidence, x1,y1,x2,y2,width,height,area
    """
    # Inference
    results = model_obj(pil_img, imgsz=640, conf=conf, device=device_choice)
    r = results[0]

    # Annotated image: ultralytics .plot() returns numpy array usually
    ann = r.plot()
    if isinstance(ann, np.ndarray):
        annotated_img = Image.fromarray(ann)
    else:
        # fallback if some versions return PIL
        annotated_img = ann if isinstance(ann, Image.Image) else Image.fromarray(np.array(ann))

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        df = pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2", "width", "height", "area"])
        return annotated_img, df

    # Extract coordinates robustly; ultralytics has tensors .xyxy .conf .cls
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        # fallback: iterate boxes
        xyxy = np.array([b.xyxy.cpu().numpy().reshape(-1) for b in boxes])
        confs = np.array([float(b.conf) for b in boxes])
        clses = np.array([int(b.cls) for b in boxes])

    # class names
    names = []
    for c in clses:
        try:
            names.append(model_obj.names[int(c)])
        except Exception:
            names.append(str(int(c)))

    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights

    df = pd.DataFrame({
        "class": names,
        "confidence": np.round(confs, 4),
        "x1": np.round(xyxy[:, 0], 1),
        "y1": np.round(xyxy[:, 1], 1),
        "x2": np.round(xyxy[:, 2], 1),
        "y2": np.round(xyxy[:, 3], 1),
        "width": np.round(widths, 1),
        "height": np.round(heights, 1),
        "area": np.round(areas, 1),
    })

    return annotated_img, df

# ---------------- DETECT BUTTON ----------------
if st.session_state.stage == "uploaded" and st.session_state.image is not None:
    st.markdown("<div class='card'><b>Preview (Original)</b></div>", unsafe_allow_html=True)
    st.image(st.session_state.image, use_container_width=True)

    detect_col1, detect_col2, detect_col3 = st.columns([1, 1, 1])
    with detect_col2:
        if st.button("🚀 Detect Microplastics"):
            if model is None:
                st.error("Model not loaded — fix the model path in the sidebar.")
            else:
                with st.spinner("Running YOLO inference..."):
                    t0 = time.time()
                    try:
                        annotated_img, df = run_yolo_inference(
                            st.session_state.image, model, conf_threshold, device, box_thickness
                        )
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        annotated_img, df = None, pd.DataFrame()
                    elapsed = time.time() - t0

                st.session_state.annotated = annotated_img
                st.session_state.df = df
                st.session_state.stage = "detected"

                # add to history
                st.session_state.history.append({
                    "filename": safe_filename(st.session_state.image) if st.session_state.image else "uploaded_image",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "df": df.copy(),
                    "annotated": annotated_img,
                    "elapsed": elapsed,
                })
                st.session_state.history = st.session_state.history[-5:]
                st.rerun()


# ---------------- TOP CONTROL BUTTONS ----------------
if st.session_state.stage == "detected":
    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        if st.button("🔄 Upload a New Image"):
            # Reset for new upload
            st.session_state.image = None
            st.session_state.annotated = None
            st.session_state.df = pd.DataFrame()
            st.session_state.stage = "initial"
            st.rerun()
    with top_col2:
        if st.button("🗑️ Clear All"):
            st.session_state.image = None
            st.session_state.annotated = None
            st.session_state.df = pd.DataFrame()
            st.session_state.history = []
            st.session_state.stage = "initial"
            st.rerun()
    
# ----------------------------- AFTER DETECTION: SIDE-BY-SIDE, KPIs, TABLE & VISUALS -----------------------------
if st.session_state.annotated is not None and st.session_state.image is not None:
    # Side-by-side original (left) and annotated (right)
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("<div class='card'><b>Original</b></div>", unsafe_allow_html=True)
        st.image(st.session_state.image, use_container_width=True)
    with right_col:
        st.markdown("<div class='card'><b>Annotated (YOLO Detection)</b></div>", unsafe_allow_html=True)
        st.image(st.session_state.annotated, use_container_width=True)

    # KPIs arranged neatly beneath images
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)  # spacing
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    df = st.session_state.df if isinstance(st.session_state.df, pd.DataFrame) else pd.DataFrame()
    with k1:
        st.markdown("<div class='card'><b>Detections</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pill'>{len(df)}</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='card'><b>Top Type</b></div>", unsafe_allow_html=True)
        top_type = df["class"].mode()[0] if not df.empty else "—"
        st.markdown(f"<div class='pill'>{top_type}</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='card'><b>Avg Confidence</b></div>", unsafe_allow_html=True)
        avg_conf = float(df["confidence"].mean()) if not df.empty else 0.0
        st.markdown(f"<div class='pill'>{avg_conf:.3f}</div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div class='card'><b>Last Run Time</b></div>", unsafe_allow_html=True)
        last_elapsed = st.session_state.history[-1]["elapsed"] if st.session_state.history else 0.0
        st.markdown(f"<div class='pill'>{last_elapsed:.2f}s</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)  # spacing before table

    # Detection table placed below KPIs to avoid overlap
    st.markdown("<div class='card'><b>Detections Table</b></div>", unsafe_allow_html=True)
    if df.empty:
        st.info("No detections to show. Try lowering the confidence threshold or providing a clearer image.")
    else:
        # show table with coordinates and sizes
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=260)

   # ---------------- VISUALIZATIONS ----------------
    with st.expander("📊 Type Distribution "):
        fig, ax = plt.subplots(figsize=(5, 4))
        if df.empty:
            ax.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        else:
            counts = df["class"].value_counts()
            labels = counts.index.tolist()
            values = counts.values.tolist()
            colors = color_map_for_values(len(labels), PIE_CMAP)
            if len(labels) == 1:
                ax.pie([1.0], labels=[labels[0]], colors=[colors[0]], startangle=90,
                       wedgeprops=dict(width=0.45, edgecolor=MT_FIG_FACE))
                ax.text(0, 0, "100%", ha="center", va="center", fontsize=18,
                        color=TEXT_COLOR, fontweight="bold")
            else:
                wedges, _, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
                    pctdistance=0.75,
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(width=0.45, edgecolor=MT_FIG_FACE)
                )
                ax.legend(wedges, [f"{l}: {v}" for l, v in zip(labels, values)],
                          bbox_to_anchor=(1.02, 0.6), loc="center left", frameon=False)
        ax.set_aspect("equal")
        st.pyplot(fig)
        st.caption("Donut shows what % of detected objects belong to each microplastic type.")

    with st.expander("📈 Confidence of Detections"):
        fig2_height = max(3, 0.25 * min(len(df), max_display) + 1.2)
        fig2, ax2 = plt.subplots(figsize=(8, fig2_height))
        if df.empty:
            ax2.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        else:
            df_sorted = df.sort_values("confidence", ascending=False).reset_index(drop=True)
            df_plot = df_sorted.head(max_display)
            colors = color_map_for_values(len(df_plot), CONF_CMAP)[::-1]
            y_pos = np.arange(len(df_plot))
            ax2.barh(y_pos, df_plot["confidence"], color=colors, edgecolor="#0b0b0b", height=0.6)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"{i+1}: {c}" for i, c in enumerate(df_plot["class"].tolist())], fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlim(0, 1.02)
            ax2.set_xlabel("Confidence (0–1)")
            ax2.set_title(f"Top {len(df_plot)} Detections by Confidence")
            for i, v in enumerate(df_plot["confidence"]):
                ax2.text(v + 0.01, i, f"{v:.3f}", va="center", color=TEXT_COLOR, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("Each bar shows how confident the model was for a detection. Higher = more confident.")

    with st.expander("📉 Size Distribution (Histogram)"):
        fig3, ax3 = plt.subplots(figsize=(9, 3))
        if df.empty:
            ax3.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        else:
            widths = df["width"]
            bins = min(12, max(4, int(math.sqrt(len(widths)) * 2)))
            n_vals, bins_edges, patches = ax3.hist(widths, bins=bins, color=HIST_COLOR,
                                                  edgecolor=MT_FIG_FACE, alpha=0.95)
            ax3.set_xlabel("Width (pixels)")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Histogram of Detected Object Widths")
            for rect in patches:
                h = rect.get_height()
                if h > 0:
                    ax3.text(rect.get_x() + rect.get_width()/2.0, h+0.05, f"{int(h)}",
                             ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)
        st.caption("Shows how many detected plastics fall into each size range (in pixels).")

    # Downloads & auxiliary controls
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
    with dl_col1:
        if st.session_state.annotated is not None:
            annotated_bytes = pil_to_bytes(st.session_state.annotated, fmt="PNG")
            st.download_button("⬇️ Download annotated image", data=annotated_bytes, file_name="microplastic_annotated.png", mime="image/png")
    with dl_col2:
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download detections CSV", data=csv_bytes, file_name="detections.csv", mime="text/csv")
    with dl_col3:
        if st.button("Clear last detection"):
            st.session_state.annotated = None
            st.session_state.df = pd.DataFrame()
            # keep history but remove last entry
            if st.session_state.history:
                st.session_state.history.pop()
            st.rerun()
# ----------------------------- SESSION HISTORY (last 5) -----------------------------
if st.session_state.history:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>🗂️ Recent detections (this session)</b></div>", unsafe_allow_html=True)
        recent_entries = st.session_state.history[-5:][::-1]
        for entry in recent_entries:
            with st.expander(f"{entry['filename']} • {entry['timestamp']}"):
                hk1, hk2, hk3, hk4 = st.columns(4)
                hk1.metric("Detections", len(entry["df"]))
                hk2.metric("Top Type", entry["df"]["class"].mode()[0] if not entry["df"].empty else "—")
                hk3.metric("Avg Confidence", f"{entry['df']['confidence'].mean():.3f}" if not entry["df"].empty else "0.0")
                hk4.metric("Last Run Time", f"{entry['elapsed']:.2f}s")
                hcol1, hcol2 = st.columns([1, 1])
                with hcol1:
                    if entry.get("annotated") is not None:
                        st.image(entry["annotated"], use_container_width=True)
                with hcol2:
                    if not entry["df"].empty:
                        st.dataframe(entry["df"].reset_index(drop=True), use_container_width=True, height=220)
    

# ----------------------------- FOOTER -----------------------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

footer_html = f"""
<div class="card" style="text-align:center; padding:20px;">
  <div style="font-size:22px;">֎ <b>Advanced AI-powered Microplastic Detection</b></div>
  <div style="margin-top:6px; font-size:15px; color:{TEXT_COLOR};">
    For environmental research, monitoring, and marine protection.
  </div>
  <hr style="margin:12px 0; border: none; border-top: 1px solid rgba(200,200,200,0.2);" />
  <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:12px; font-size:14px;">
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🧠 YOLOv8 Deep Learning</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🌐 Computer Vision</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🌍 Environmental Analysis</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">📊 Research</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🌱 Environmental Impact</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🐟 Marine Biology</span>
    <span style="padding:4px 10px; border-radius:20px; background:rgba(255,255,255,0.08);">🚨 Pollution Monitoring</span>
  </div>
  <div style="margin-top:14px; font-size:13px; opacity:0.8;">
    ©2025 Microplastic Detection AI. Advancing environmental science through artificial intelligence.
  </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
