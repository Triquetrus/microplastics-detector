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
    st.caption(f"Verified by identical SHA-256 checkpoint • {run.relative_to(ROOT)} • VALIDATION metrics")
    if best is not None:
        values = {k: float(best[c]) if c and pd.notna(best[c]) else None for k, c in columns.items()}
        p, r = values["precision"], values["recall"]
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0 if p == r == 0 else None)
        for col, label, value in zip(st.columns(5), ["Precision", "Recall", "F1 Score", "mAP@50", "mAP@50-95"],
                                    [p, r, f1, values["map50"], values["map95"]]):
            col.metric(label, f"{value:.3f}" if value is not None else "Unavailable")
        st.caption(f"Best Epoch: {int(best['epoch']) if 'epoch' in best and pd.notna(best['epoch']) else 'Unavailable'} • maximum validation mAP@50-95. F1 is calculated from that epoch’s precision and recall.")
    else:
        st.info("Best-epoch validation metrics unavailable in this results.csv.")
    st.markdown("#### Training and validation")
    st.caption("Training loss = model error on training data. Validation loss = model error on unseen validation data. Both poor may suggest underfitting; training improves while validation deteriorates may suggest overfitting; both improve or stabilize suggests healthier generalization. These are diagnostic patterns, not a diagnosis of this run.")
    plot_training_history(history, columns)
    matrix = next((run / n for n in ["confusion_matrix_normalized.png", "confusion_matrix.png"] if (run / n).is_file()), None)
    st.markdown("#### Validation Confusion Matrix")
    st.caption("For a single-class detector, this primarily helps visualize correct microplastic detections versus background-related errors. Run-level exported artifacts may use a different operating threshold from the current image settings.")
    if matrix:
        try:
            st.image(Image.open(matrix), width=680)
        except (OSError, ValueError):
            st.info("The saved confusion matrix could not be read.")
    else:
        st.info("No saved confusion matrix exists for this verified run.")


def render_concepts():
    cards = [
        ("Supervised Learning", "In YOLO training, labelled images teach the model to locate microplastic particles. The trained model predicts their locations in unseen images. Implementation: YOLO object detection."),
        ("Classification / Object Detection", "Object detection combines localization (where is the microplastic?) and classification (is it a microplastic?). This model has one class: microplastic."),
        ("Precision", "Of the detections predicted as microplastic, how many were correct? Precision = TP / (TP + FP)."),
        ("Recall", "Of the actual microplastics present, how many did the model detect? Recall = TP / (TP + FN)."),
        ("F1 Score", "Balances precision and recall. F1 = 2PR / (P + R). Confidence on uploaded images is not an accuracy metric."),
        ("mAP", "Mean Average Precision evaluates object detection. mAP50 uses IoU 0.50; mAP50-95 averages IoU thresholds from 0.50 to 0.95. IoU measures box overlap with ground truth."),
        ("Overfitting / Underfitting", "Compare real training and validation curves in Model Performance. Improving training loss with deteriorating validation loss may indicate overfitting. Poor performance on both may indicate underfitting."),
        ("Unsupervised Learning", "K-Means groups detected particles by geometric similarity without predefined cluster labels. It does not identify polymer types or improve YOLO accuracy."),
        ("Feature Scaling", "StandardScaler centers each feature and scales its variance, preventing large numeric ranges such as area from dominating K-Means distances."),
        ("Cluster Validation", "Silhouette measures separation and cohesion: closer to 1 is better separated, near 0 means overlap, negative values suggest poor assignment. An elbow in WCSS can guide K; neither proves scientific categories.")]
    cols = st.columns(2)
    for i, (title, body) in enumerate(cards):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(body)


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
        left, right = st.columns(2)
        for column, title, image in [(left, "ORIGINAL", entry["image"]), (right, "YOLO DETECTION", annotated)]:
            with column:
                st.caption(title)
                preview = image.copy()
                preview.thumbnail((720, 420))
                st.image(preview, width="content")
        stats = image_statistics(df, entry["image"])
        for col, label, value in zip(st.columns(4), ["DETECTED PARTICLES", "AVG CONFIDENCE", "HIGHEST CONFIDENCE", "INFERENCE TIME"],
                [stats["count"], f"{stats['average_confidence']:.1%}", f"{stats['highest_confidence']:.1%}", f"{entry['elapsed']:.2f} s"]):
            col.metric(label, value)
        st.markdown("#### Detection Summary")
        st.write(f"{len(df)} candidate microplastic particles were detected in this image." if len(df) else "No particles detected above the selected confidence threshold.")
        st.caption(f"Processed at confidence ≥ {entry['confidence']:.0%} on {entry['device']}. Change the threshold and process again to update predictions.")
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
        display = df.copy()
        display["confidence"] = display.confidence * 100
        display["cluster"] = display.cluster.map(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "—")
        display = display.rename(columns={c: c.replace("_", " ").title() for c in display})
        st.dataframe(display.round(2), hide_index=True, width="stretch", column_config={"Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%")})
        st.caption("Coordinates and dimensions are pixels; area is px². CSV confidence is a fraction (0–1); cluster IDs start at 1.")
        a, b = st.columns(2)
        stem = Path(entry["filename"]).stem
        a.download_button("Download detections CSV", df.to_csv(index=False).encode(), file_name=f"{stem}_detections.csv", mime="text/csv")
        b.download_button("Download annotated image", pil_to_bytes(annotated), file_name=f"{stem}_annotated.png", mime="image/png")
    with clustering:
        st.markdown("#### K-Means Particle Clusters")
        st.caption("Detected particles are grouped according to geometric similarity. These clusters are data-driven and do not represent polymer chemistry.")
        st.caption("YOLO detections → geometric features → StandardScaler → K-Means → particle clusters")
        if not enabled:
            st.info("Enable K-Means in the sidebar to analyze particle geometry.")
        elif cluster_data:
            _, silhouette, elbow, centers, message = cluster_data
            st.info(message)
            if centers is not None:
                fig, ax = plt.subplots(figsize=(7, 3.4))
                cmap = matplotlib.colormaps.get_cmap("viridis")
                for cluster, group in df.dropna(subset=["cluster"]).groupby("cluster"):
                    ax.scatter(group.area, group.aspect_ratio, color=cmap((int(cluster)-1)/max(len(centers)-1, 1)), label=f"Cluster {cluster}", alpha=.8)
                ax.scatter(centers.area, centers.aspect_ratio, marker="X", s=150, c="#f59e0b", edgecolors="black", label="Centers")
                ax.set(xlabel="Area (px²)", ylabel="Aspect ratio", title="K-Means Particle Clusters")
                ax.legend()
                show_plot(fig)
                st.caption("This is a two-feature projection of clustering in four scaled dimensions; centers are converted back to original units.")
                summary = df.dropna(subset=["cluster"]).groupby("cluster").agg(**{"Particle Count": ("detection", "count"), "Mean Width": ("width", "mean"), "Mean Height": ("height", "mean"), "Mean Area": ("area", "mean"), "Mean Aspect Ratio": ("aspect_ratio", "mean"), "Mean Confidence": ("confidence", "mean")}).reset_index()
                summary["cluster"] = summary.cluster.map(lambda x: f"Cluster {x}")
                st.dataframe(summary.rename(columns={"cluster": "Cluster"}).round(3), hide_index=True, width="stretch")
                st.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "Not defined")
                st.caption("Closer to 1 = better separated clusters. Near 0 = overlapping clusters. Requires 2 through n−1 distinct clusters; K = n is not valid for silhouette.")
                st.line_chart(elbow.set_index("K"), x_label="K", y_label="WCSS / inertia")
                st.caption("Elbow Method: look for diminishing reductions in within-cluster sum of squares. With very few particles, an elbow is not reliable.")
    with performance:
        render_performance()
    with concepts:
        render_concepts()


def main():
    st.set_page_config(page_title="MicroDetect", page_icon="🔬", layout="wide")
    st.sidebar.markdown("## MicroDetect")
    st.sidebar.markdown("#### MODEL")
    model_status = st.sidebar.empty()
    model = None
    device = "cpu"
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
        model_status.success("● Model Ready")
    except Exception as exc:
        model_status.error(f"Model unavailable: {exc}")
    st.sidebar.caption(f"Model: best.pt · Device: {'GPU' if device == 0 else 'CPU'}")
    st.sidebar.markdown("#### DETECTION SETTINGS")
    confidence = st.sidebar.slider("Confidence threshold", .01, 1., .25, .01)
    thickness = st.sidebar.slider("Bounding-box thickness", 1, 8, 2)
    st.sidebar.markdown("#### CLUSTERING")
    enabled = st.sidebar.toggle("Enable K-Means", value=True)
    k = st.sidebar.slider("Number of clusters", 2, 5, 3, disabled=not enabled)
    st.sidebar.markdown("#### DISPLAY")
    theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    dark = theme == "Dark"
    bg, card, text, border = ("#0b1220", "#132033", "#e6edf5", "#27394c") if dark else ("#f4f8fb", "#ffffff", "#10283e", "#cbd9e5")
    st.markdown(f"""<style>
    .stApp {{background:{bg}; color:{text}; color-scheme:{'dark' if dark else 'light'}; --text-color:{text}; --background-color:{bg}; --secondary-background-color:{card};}}
    [data-testid="stSidebar"], [data-testid="stHeader"] {{background:{card};}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp label,.stApp p {{color:{text};}}
    .block-container {{padding-top:2rem; max-width:1280px;}}
    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {{background:{card}; border:1px solid {border}; border-radius:12px; padding:12px;}}
    .stButton button,.stDownloadButton button {{background:{card}; color:{text}; border:1px solid #14b8a6; border-radius:8px;}}
    [data-baseweb="select"]>div,[data-testid="stFileUploaderDropzone"] {{background:{card}; color:{text};}}
    .badge {{display:inline-block; border:1px solid {border}; border-radius:20px; padding:4px 12px; margin:0 6px 8px 0; color:#14b8a6; font-size:12px;}}
    </style>""", unsafe_allow_html=True)
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "text.color": text,
                         "axes.labelcolor": text, "xtick.color": text, "ytick.color": text, "axes.edgecolor": border})
    st.title("MicroDetect")
    st.markdown("**AI-Powered Microplastic Detection & Analysis**")
    st.markdown(''.join(f'<span class="badge">{b}</span>' for b in ["YOLO Detection", "Supervised ML", "K-Means Clustering", "Unsupervised ML"]), unsafe_allow_html=True)
    st.write("Detect microplastics from sample images and analyze their geometric patterns using supervised and unsupervised machine learning.")
    st.session_state.setdefault("samples", [])
    st.session_state.setdefault("upload_version", 0)
    st.markdown("### Upload Sample Images")
    uploads = st.file_uploader("Sample images · JPG, JPEG, PNG · up to 10 MB each", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"uploads_{st.session_state.upload_version}")
    a, b = st.columns([1, 4])
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
        st.info("Upload sample images and select Process Images to begin.")
        performance, concepts = st.tabs(["Model Performance", "ML Concepts"])
        with performance:
            render_performance()
        with concepts:
            render_concepts()
    st.caption("Results are image-based screening outputs and should not be interpreted as laboratory-confirmed concentration or toxicity. Chemical identification may require FTIR or Raman spectroscopy.")


if __name__ == "__main__":
    main()
