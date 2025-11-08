import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
import streamlit as st
import requests, time, pandas as pd

# ---------- CONFIG ----------
API_BASE = "http://localhost:8000"
UPLOAD_EP = f"{API_BASE}/process-video/"
STREAM_EP = f"{API_BASE}/process-stream/"
PLATES_EP = f"{API_BASE}/plates/"
HEALTH_EP = f"{API_BASE}/health"

st.set_page_config(page_title="LPR Dashboard", page_icon="car", layout="wide")

# ---------- HELPERS ----------
def health_ok():
    try:
        return requests.get(HEALTH_EP, timeout=3).ok
    except:
        return False

@st.cache_data(ttl=4)
def get_plates():
    try:
        r = requests.get(PLATES_EP)
        return pd.DataFrame(r.json()) if r.ok else pd.DataFrame()
    except:
        return pd.DataFrame()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Control Panel")
    if not health_ok():
        st.error("Backend unreachable")
        st.stop()
    st.success("API ready")

    st.subheader("Source")
    source_type = st.radio(
        "Select source",
        ["Upload video file", "Webcam / RTSP / HTTP URL"],
        index=0
    )

    if source_type == "Upload video file":
        uploaded = st.file_uploader(
            "Choose video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported: MP4, AVI, MOV, MKV"
        )
        start_btn = st.button("Start Processing (Upload)", type="primary")
        if start_btn and uploaded:
            with st.spinner("Uploading and starting..."):
                files = {"file": (uploaded.name, uploaded, uploaded.type)}
                r = requests.post(UPLOAD_EP, files=files)
                if r.ok:
                    src = r.json().get("video_source", uploaded.name)
                    st.success(f"Processing: **{src}**")
                    st.balloons()
                else:
                    st.error(f"Upload failed: {r.text}")

    else:  # Webcam / RTSP / HTTP URL
        url = st.text_input(
            "Stream URL",
            placeholder="rtsp://admin:12345@192.168.1.50:554/stream1",
            help="RTSP, HTTP MJPEG, or '0' for local webcam"
        )
        name = st.text_input(
            "Friendly name (optional)",
            placeholder="Front Gate",
            help="Used in results table"
        )
        start_btn = st.button("Start Live Stream", type="primary")
        if start_btn and url.strip():
            payload = {
                "url": url.strip(),
                "name": name.strip() or None
            }
            with st.spinner("Connecting to stream..."):
                r = requests.post(STREAM_EP, json=payload)
                if r.ok:
                    src = r.json().get("video_source", url.strip())
                    st.success(f"Live stream started: **{src}**")
                    st.balloons()
                else:
                    st.error(f"Stream failed: {r.text}")

    st.markdown("---")
    st.caption("Results auto-refresh every 4 seconds")

# ---------- MAIN ----------
st.title("License Plate Recognition")
st.markdown("YOLOv8 + EasyOCR + SORT – **Batch & Live**")

plates = get_plates()

c1, c2 = st.columns([3, 1])
with c1:
    st.subheader("Detected Plates")
    if not plates.empty:
        plates["timestamp"] = pd.to_datetime(plates["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        plates = plates.sort_values(["score", "timestamp"], ascending=[False, False])
        st.dataframe(
            plates[["car_id", "license_number", "score", "video_source", "timestamp"]],
            use_container_width=True,
            hide_index=True
        )
        csv = plates.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "license_plates.csv", "text/csv")
    else:
        st.info("No plates detected yet. Start a source above.")

with c2:
    st.subheader("Summary")
    if not plates.empty:
        st.metric("Unique Cars", plates["car_id"].nunique())
        st.metric("Unique Plates", plates["license_number"].nunique())
        st.metric("Avg OCR Score", f"{plates['score'].mean():.3f}")
    else:
        st.write("—")

# ---------- LOG ----------
st.markdown("---")
st.subheader("Live Log")
log_ph = st.empty()
if not plates.empty:
    log = "\n".join(
        f"[{r.timestamp}] Car {r.car_id} → {r.license_number} ({r.score:.3f})"
        for r in plates.head(12).itertuples()
    )
    log_ph.code(log)
else:
    log_ph.code("Waiting for first detection...")

# Auto-refresh
time.sleep(20)
st.rerun()