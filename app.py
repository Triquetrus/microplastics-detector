"""MicroDetect: single-class YOLO screening and geometric cluster analysis."""
import hashlib
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
FEATURES = ["width", "height", "area", "aspect_ratio"]
COLUMNS = ["detection", "confidence", "x1", "y1", "x2", "y2", *FEATURES]


@st.cache_resource
def load_model(path, signature):
    from ultralytics import YOLO
    model = YOLO(path, task="detect")
    if model.task != "detect" or model.names != {0: "microplastic"}:
        raise ValueError("Expected the single-class microplastic detection model.")
    return model


def extract_detection_features(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=COLUMNS, dtype=float)
    xy = boxes.xyxy.cpu().numpy().astype(float)
    width = np.maximum(xy[:, 2] - xy[:, 0], 0)
    height = np.maximum(xy[:, 3] - xy[:, 1], 0)
    return pd.DataFrame(dict(detection=np.arange(1, len(xy) + 1),
        confidence=boxes.conf.cpu().numpy(), x1=xy[:, 0], y1=xy[:, 1],
        x2=xy[:, 2], y2=xy[:, 3], width=width, height=height,
        area=width * height, aspect_ratio=np.divide(width, height,
            out=np.zeros_like(width), where=height > 0)))


def run_inference(image, model, confidence, device):
    start = time.perf_counter()
    result = model.predict(image, conf=confidence, device=device, verbose=False)[0]
    elapsed = time.perf_counter() - start
    return extract_detection_features(result), elapsed


def annotate_image(image, df, thickness):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for row in df.itertuples():
        draw.rectangle((row.x1, row.y1, row.x2, row.y2), outline="#14b8a6", width=thickness)
        draw.text((row.x1 + 2, max(0, row.y1 - 12)),
                  f"microplastic {row.confidence:.0%}", fill="#14b8a6", stroke_width=1, stroke_fill="#102030")
    return annotated


def image_statistics(df, image):
    image_area = image.width * image.height
    total_area = float(df.area.sum())
    return {"count": len(df), "average_confidence": float(df.confidence.mean()) if len(df) else 0,
            "highest_confidence": float(df.confidence.max()) if len(df) else 0,
            "average_area": float(df.area.mean()) if len(df) else 0,
            "total_area": total_area, "image_area": image_area,
            "area_percentage": 100 * total_area / image_area}


def calculate_cluster_metrics(scaled, labels):
    groups = len(np.unique(labels))
    silhouette = float(silhouette_score(scaled, labels)) if 1 < groups < len(scaled) else None
    upper = min(5, len(scaled), len(np.unique(scaled, axis=0)))
    elbow = [{"K": k, "WCSS": float(KMeans(n_clusters=k, random_state=42,
               n_init="auto").fit(scaled).inertia_)} for k in range(1, upper + 1)]
    return silhouette, pd.DataFrame(elbow)


@st.cache_data(show_spinner=False, max_entries=64)
def run_kmeans(df, requested_k):
    output = df.copy()
    output["cluster"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    values = df[FEATURES].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values[:, 0] > 0) & (values[:, 1] > 0)
    usable = values[valid]
    unique = len(np.unique(usable, axis=0))
    k = min(int(requested_k), len(usable), unique)
    if k < 2:
        return output, None, pd.DataFrame(), None, "At least two distinct, positive-size detections are needed for K-Means."
    scaler = StandardScaler()
    scaled = scaler.fit_transform(usable)
    estimator = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(scaled)
    output.loc[valid, "cluster"] = estimator.labels_ + 1
    silhouette, elbow = calculate_cluster_metrics(scaled, estimator.labels_)
    centers = pd.DataFrame(scaler.inverse_transform(estimator.cluster_centers_), columns=FEATURES)
    message = f"K = {k}; {len(usable)} particles. Features: width, height, area, aspect ratio."
    if k != requested_k:
        message += " K was reduced to the number of distinct usable detections."
    if not valid.all():
        message += " Degenerate or nonfinite boxes were excluded."
    return output, silhouette, elbow, centers, message


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@st.cache_data(show_spinner=False)
def load_training_metrics(model_path, signature, inventory):
    """Only associate artifacts with a byte-identical checkpoint, never by run name."""
    try:
        digest = file_digest(model_path)
        for csv_name, _, _ in inventory:
            csv_path = Path(csv_name)
            checkpoint = csv_path.parent / "weights" / "best.pt"
            if not checkpoint.is_file() or file_digest(checkpoint) != digest:
                continue
            history = pd.read_csv(csv_path)
            history.columns = history.columns.str.strip()
            history = history.apply(pd.to_numeric, errors="coerce")
            candidates = {"precision": "metrics/precision", "recall": "metrics/recall",
                          "map50": "metrics/mAP50", "map95": "metrics/mAP50-95"}
            columns = {key: next((c for c in history if c in (prefix, prefix + "(B)")), None)
                       for key, prefix in candidates.items()}
            if columns["map95"] is None or history[columns["map95"]].dropna().empty:
                return csv_path.parent, history, columns, None
            best = history.loc[history[columns["map95"]].idxmax()]
            return csv_path.parent, history, columns, best
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    return None, pd.DataFrame(), {}, None


def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def plot_training_history(history, columns):
    if "epoch" not in history:
        st.info("Epoch column unavailable; training curves cannot be plotted.")
        return
    for loss in ("box_loss", "cls_loss", "dfl_loss"):
        available = [c for c in (f"train/{loss}", f"val/{loss}") if c in history]
        if available:
            fig, ax = plt.subplots(figsize=(8, 2.8))
            for col in available:
                ax.plot(history.epoch, history[col], label=col)
            ax.set(xlabel="Epoch", ylabel="Loss", title=loss.replace("_", " ").title())
            ax.legend()
            show_plot(fig)
    available = [c for c in columns.values() if c]
    if available:
        st.line_chart(history.set_index("epoch")[available], x_label="Epoch", y_label="Validation metric")


def render_performance():
    inventory = []
    for csv in ROOT.glob("**/runs/**/results.csv"):
        checkpoint = csv.parent / "weights" / "best.pt"
        if checkpoint.exists():
            inventory.append((str(csv), csv.stat().st_mtime_ns, checkpoint.stat().st_mtime_ns))
    signature = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else None
    run, history, columns, best = load_training_metrics(str(MODEL_PATH), signature, tuple(sorted(inventory)))
    if run is None:
        st.info("Training metrics unavailable for the currently loaded model.")
        return
    st.caption(f"Validation metrics · {run.relative_to(ROOT)}", help="Training artifacts are linked by an identical SHA-256 checkpoint.")
    if best is not None:
        values = {k: float(best[c]) if c and pd.notna(best[c]) else None for k, c in columns.items()}
        p, r = values["precision"], values["recall"]
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0 if p == r == 0 else None)
        for col, label, value in zip(st.columns(5), ["Precision", "Recall", "F1", "mAP50", "mAP50–95"],
                                    [p, r, f1, values["map50"], values["map95"]]):
            col.metric(label, f"{value:.3f}" if value is not None else "Unavailable")
        st.caption(f"Best epoch · {int(best['epoch']) if 'epoch' in best and pd.notna(best['epoch']) else 'Unavailable'}", help="Selected by maximum validation mAP50–95. F1 is calculated from that epoch’s precision and recall.")
    else:
        st.info("Best-epoch validation metrics unavailable in this results.csv.")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("#### Training / Validation Performance")
        plot_training_history(history, columns)
    with right:
        matrix = next((run / n for n in ["confusion_matrix_normalized.png", "confusion_matrix.png"] if (run / n).is_file()), None)
        st.markdown("#### Confusion Matrix")
        st.caption("Single-class detections and background errors", help="Saved validation artifact; its operating threshold may differ from the current image settings.")
        if matrix:
            try:
                st.image(Image.open(matrix), width="stretch")
            except (OSError, ValueError):
                st.info("The saved confusion matrix could not be read.")
        else:
            st.info("No saved confusion matrix exists for this verified run.")


def render_concepts():
    st.markdown("#### Supervised learning")
    st.write("YOLO learns localization and classification from labelled images. This detector has one class: microplastic.")
    st.caption("Precision measures correctness; recall measures coverage. F1 balances both. mAP evaluates detection across IoU thresholds (0.50, or 0.50–0.95).")
    st.code("Precision = TP / (TP + FP)    Recall = TP / (TP + FN)    F1 = 2PR / (P + R)", language=None)
    st.markdown("#### Generalization")
    st.write("Training loss measures error on training images; validation loss measures error on unseen validation images.")
    st.caption("Poor performance on both may indicate underfitting. Improving training loss with deteriorating validation loss may indicate overfitting. Inspect the real curves before drawing conclusions.")
    st.markdown("#### Unsupervised learning")
    st.write("StandardScaler balances feature scales. K-Means groups particles by geometric similarity without predefined labels.")
    st.caption("Silhouette evaluates cluster separation; the elbow method compares WCSS across K. Clusters do not identify polymers or improve YOLO accuracy.")


def pil_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_results(entry, enabled, k, thickness):
    df = entry["df"].copy()
    cluster_data = None
    if enabled:
        try:
            cluster_data = run_kmeans(df, k)
            df = cluster_data[0]
        except (ValueError, RuntimeError) as exc:
            st.warning(f"Clustering unavailable: {exc}")
    if "cluster" not in df:
        df["cluster"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    annotated = annotate_image(entry["image"], df, thickness)
    overview, detections, clustering, performance, concepts = st.tabs(["Overview", "Detections", "Clustering", "Model Performance", "ML Concepts"])
    with overview:
        stats = image_statistics(df, entry["image"])
        for col, label, value in zip(st.columns(4), ["Detected", "Avg. Confidence", "Highest Confidence", "Inference"],
                [stats["count"], f"{stats['average_confidence']:.1%}", f"{stats['highest_confidence']:.1%}", f"{entry['elapsed']:.2f} s"]):
            col.metric(label, value)
        left, right = st.columns(2)
        for column, title, image in [(left, "Original", entry["image"]), (right, "Detection Result", annotated)]:
            with column:
                st.caption(title)
                # Display-only letterboxing preserves the full image and aligns both panels.
                preview = ImageOps.pad(image, (900, 480), color=plt.rcParams["axes.facecolor"])
                st.image(preview, width="stretch")
        if not len(df):
            st.info("No particles detected above the selected confidence threshold.")
        st.caption(f"Processed at confidence ≥ {entry['confidence']:.0%} on {entry['device']}. Change the threshold and process again to update predictions.")
        with st.expander("Image measurements", expanded=False):
            st.write(f"Approximate detected-area percentage: **{stats['area_percentage']:.2f}%** · Mean box area: **{stats['average_area']:,.1f} px²** · Total box area: **{stats['total_area']:,.1f} px²** · Image area: **{stats['image_area']:,} px²**")
            st.caption("Image-based measurement: sum of bounding-box areas / image area. Boxes include surrounding pixels and may overlap, so the percentage can exceed 100%. This is not environmental concentration.")
        if len(df):
            for col, feature, title, label in zip(st.columns(2), ["confidence", "area"], ["Confidence distribution", "Size distribution"], ["Confidence", "Bounding-box area (px²)"]):
                with col:
                    fig, ax = plt.subplots(figsize=(5, 2.6))
                    ax.hist(df[feature], bins=min(15, max(3, int(np.sqrt(len(df))))), color="#14b8a6", edgecolor="#102030")
                    ax.set(title=title, xlabel=label, ylabel="Particles")
                    show_plot(fig)
    with detections:
        st.markdown(f"#### {len(df)} detections")
        st.caption(f"Average confidence: {stats['average_confidence']:.1%}")
        display = df.copy()
        display["confidence"] = display.confidence * 100
        display["cluster"] = display.cluster.map(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "—")
        display = display.rename(columns={c: c.replace("_", " ").title() for c in display})
        st.dataframe(display.round(2), hide_index=True, width="stretch", column_config={"Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%")})
        st.caption("Coordinates and dimensions are pixels; area is px². CSV confidence is a fraction (0–1); cluster IDs start at 1.")
        a, b = st.columns(2)
        stem = Path(entry["filename"]).stem
        a.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{stem}_detections.csv", mime="text/csv")
        b.download_button("Download annotated image", pil_to_bytes(annotated), file_name=f"{stem}_annotated.png", mime="image/png")
    with clustering:
        st.markdown("#### Particle Clustering")
        st.caption("K-Means groups detected particles using geometric features such as area and aspect ratio.")
        if not enabled:
            st.info("Enable K-Means in the sidebar to analyze particle geometry.")
        elif cluster_data:
            _, silhouette, elbow, centers, message = cluster_data
            if centers is None:
                st.info(message)
            if centers is not None:
                a, b, c = st.columns(3)
                a.metric("Clusters", len(centers), help=message)
                b.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "Not defined",
                         help="Closer to 1 = separated clusters; near 0 = overlap. Requires 2 through n−1 clusters.")
                c.metric("Particles Analyzed", int(df.cluster.notna().sum()))
                left, right = st.columns([1.1, 1], gap="large")
                with left:
                    fig, ax = plt.subplots(figsize=(7, 3.4))
                    cmap = matplotlib.colormaps.get_cmap("winter")
                    for cluster, group in df.dropna(subset=["cluster"]).groupby("cluster"):
                        ax.scatter(group.area, group.aspect_ratio, color=cmap((int(cluster)-1)/max(len(centers)-1, 1)), label=f"Cluster {cluster}", alpha=.8)
                    ax.scatter(centers.area, centers.aspect_ratio, marker="X", s=150, c="#cbd5e1", edgecolors="black", label="Centers")
                    ax.set(xlabel="Area (px²)", ylabel="Aspect ratio", title="K-Means Particle Clusters")
                    ax.legend()
                    show_plot(fig)
                with right:
                    st.caption("Cluster summary")
                    summary = df.dropna(subset=["cluster"]).groupby("cluster").agg(**{"Particle Count": ("detection", "count"), "Mean Width": ("width", "mean"), "Mean Height": ("height", "mean"), "Mean Area": ("area", "mean"), "Mean Aspect Ratio": ("aspect_ratio", "mean"), "Mean Confidence": ("confidence", "mean")}).reset_index()
                    summary["cluster"] = summary.cluster.map(lambda x: f"Cluster {x}")
                    st.dataframe(summary.rename(columns={"cluster": "Cluster"}).round(3), hide_index=True, width="stretch")
                st.markdown("#### Elbow Method")
                st.line_chart(elbow.set_index("K"), x_label="K", y_label="WCSS / inertia")
                st.caption("Clusters represent geometric similarity, not polymer type.", help="An elbow suggests diminishing WCSS reductions. With very few particles, it is not reliable. The scatter is a two-feature projection of four-dimensional clustering.")
    with performance:
        render_performance()
    with concepts:
        render_concepts()


def main():
    st.set_page_config(page_title="MicroDetect", layout="wide")
    st.sidebar.markdown("## MicroDetect")
    st.sidebar.caption("Settings")
    model_status = None
    model = None
    device = "cpu"
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
        model_status = "Ready"
    except Exception as exc:
        model_status = f"Model unavailable: {exc}"
    st.sidebar.markdown("#### Detection")
    confidence = st.sidebar.slider("Confidence threshold", .01, 1., .25, .01)
    thickness = st.sidebar.slider("Box thickness", 1, 8, 2)
    st.sidebar.markdown("#### Clustering")
    enabled = st.sidebar.toggle("Enable clustering", value=True)
    k = st.sidebar.slider("Number of clusters", 2, 5, 3, disabled=not enabled)
    st.sidebar.markdown("#### Compute")
    st.sidebar.caption("GPU" if device == 0 else "CPU")
    st.sidebar.markdown("#### Appearance")
    theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    dark = theme == "Dark"
    bg, card, text, border = ("#0b0f17", "#121923", "#e8edf3", "#26303c") if dark else ("#f7f8fa", "#ffffff", "#1b2533", "#dce1e7")
    muted = "#98a4b3" if dark else "#647184"
    st.sidebar.divider()
    st.sidebar.caption("Model")
    st.sidebar.markdown("best.pt")
    if model is not None:
        st.sidebar.caption("● Ready")
    else:
        st.sidebar.error(model_status)
    st.markdown(f"""<style>
    .stApp {{background:{bg}; color:{text}; color-scheme:{'dark' if dark else 'light'};
        --text-color:{text}; --background-color:{bg}; --secondary-background-color:{card}; --primary-color:#369d9b;}}
    [data-testid="stHeader"] {{background:transparent;}}
    [data-testid="stToolbar"], #MainMenu, footer {{display:none;}}
    [data-testid="stSidebar"] {{background:{card}; border-right:1px solid {border}; min-width:240px; max-width:280px;}}
    [data-testid="stSidebarUserContent"] {{padding:0.5rem 1.25rem 1rem;}}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:0.4rem;}}
    [data-testid="stSidebar"] h4 {{padding-top:0.35rem;}}
    .block-container {{padding:2rem 2.25rem 2rem; max-width:1440px;}}
    [data-testid="stVerticalBlock"] {{gap:0.8rem;}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,.stApp p {{color:{text};}}
    .stApp h1 {{font-size:2rem; letter-spacing:-0.045em; font-weight:650; padding:0;}}
    .stApp h3 {{font-size:1.1rem; font-weight:600; padding:0;}}
    .stApp h4 {{font-size:0.95rem; font-weight:550; padding:0.65rem 0 0.2rem;}}
    [data-testid="stCaptionContainer"] p {{color:{muted}; font-size:0.8rem; line-height:1.5;}}
    [data-testid="stMetric"] {{background:{card}; border:1px solid {border}; border-radius:8px; padding:14px 16px;}}
    [data-testid="stMetricLabel"] p {{color:{muted}; font-size:0.78rem;}}
    [data-testid="stMetricValue"] {{font-size:1.8rem; font-weight:550; letter-spacing:-0.035em;}}
    .stButton button,.stDownloadButton button {{background:{card}; color:{text}; border:1px solid {border}; border-radius:8px; min-height:2.3rem;}}
    .stButton button:hover,.stDownloadButton button:hover {{border-color:#369d9b; color:{text};}}
    .stButton button[kind="primary"] {{background:#287e7c; border-color:#287e7c; color:white;}}
    .stButton button:disabled {{opacity:0.45;}}
    [data-baseweb="select"]>div,[data-testid="stFileUploaderDropzone"] {{background:{card}; color:{text}; border-radius:8px;}}
    [data-testid="stFileUploaderDropzone"] {{border:1px dashed {border}; padding:0.7rem 1rem;}}
    [data-testid="stFileUploaderDropzone"] button {{background:{card}; color:{text}; border:1px solid {border}; border-radius:6px;}}
    [data-baseweb="tab-list"] {{gap:1.5rem; border-bottom:1px solid {border};}}
    [data-baseweb="tab"] {{color:{muted}; padding:0 0 10px; font-size:0.85rem;}}
    [data-baseweb="tab"][aria-selected="true"] {{color:{text};}}
    [data-baseweb="tab-highlight"] {{background:#369d9b; height:2px;}}
    [data-testid="stDataFrame"] {{border:1px solid {border}; border-radius:8px; overflow:hidden;}}
    [data-testid="stExpander"] details {{border-color:{border}; border-radius:8px;}}
    .product-subtitle {{font-size:0.95rem; color:{text}; margin:0.25rem 0;}}
    @media (max-width:900px) {{.block-container {{padding:1.5rem 1rem;}} [data-baseweb="tab-list"] {{gap:0.8rem;}}}}
    </style>""", unsafe_allow_html=True)
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "text.color": text,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.edgecolor": border, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "normal", "axes.titlesize": 11, "font.size": 9,
                         "legend.frameon": False, "grid.color": border, "grid.alpha": .35,
                         "axes.prop_cycle": matplotlib.cycler(color=["#369d9b", "#a8b6c7", "#688797", "#c2cbd4"])})
    st.title("MicroDetect")
    st.markdown('<div class="product-subtitle">Microplastic Detection &amp; Analysis</div>', unsafe_allow_html=True)
    st.caption("Computer vision detection with geometric clustering and model evaluation.")
    st.session_state.setdefault("samples", [])
    st.session_state.setdefault("upload_version", 0)
    with st.expander("Analyze Sample", expanded=not bool(st.session_state.samples)):
        st.caption("Upload microscopy/sample images for microplastic detection.")
        uploads = st.file_uploader("Sample images · JPG, JPEG, PNG · up to 10 MB each", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"uploads_{st.session_state.upload_version}")
        a, b = st.columns([1, 1, 3])[:2]
        process = a.button("Process Images", type="primary", disabled=not uploads or model is None)
        if b.button("Clear session", disabled=not st.session_state.samples and not uploads):
            st.session_state.samples = []
            st.session_state.upload_version += 1
            st.rerun()
    if process:
        with st.spinner("Processing sample images…"):
            for upload in uploads:
                try:
                    if upload.size > 10 * 1024 * 1024:
                        raise ValueError("Maximum file size is 10 MB.")
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert("RGB")
                    df, elapsed = run_inference(image, model, confidence, device)
                    st.session_state.samples.append(dict(filename=upload.name, image=image, df=df, elapsed=elapsed,
                        confidence=confidence, device="GPU" if device == 0 else "CPU", timestamp=time.strftime("%H:%M:%S")))
                except Exception as exc:
                    st.error(f"Could not process {upload.name}: {exc}")
    if st.session_state.samples:
        samples = st.session_state.samples
        selected = st.selectbox("Session images", range(len(samples)), index=len(samples)-1,
            format_func=lambda i: f"{i+1}. {samples[i]['filename']} · {samples[i]['timestamp']} · {len(samples[i]['df'])} detections")
        render_results(samples[selected], enabled, k, thickness)
        with st.expander("Recent session images"):
            for col, entry in zip(st.columns(min(5, len(samples))), samples[-5:]):
                col.image(entry["image"], caption=entry["filename"], width=100)
    else:
        performance, concepts = st.tabs(["Model Performance", "ML Concepts"])
        with performance:
            render_performance()
        with concepts:
            render_concepts()
    st.caption("Results are image-based screening outputs and should not be interpreted as laboratory-confirmed concentration or toxicity. Chemical identification may require FTIR or Raman spectroscopy.")


if __name__ == "__main__":
    main()
