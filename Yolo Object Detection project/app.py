import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# ---------------- Page Config ----------------
st.set_page_config(page_title="YOLO Vision App", layout="wide")
st.title("🚀 YOLO Object Vision System")

# ---------------- Session State ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "report" not in st.session_state:
    st.session_state.report = {}

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

input_source = st.sidebar.radio(
    "Select Input Source",
    ("Webcam", "Video Upload")
)

task = st.sidebar.selectbox(
    "Select Vision Task",
    (
        "Object Detection",
        "Object Counting",
        "Object Segmentation",
        "Object Tracking"
    )
)

start_btn = st.sidebar.button("▶ Start")
stop_btn = st.sidebar.button("⏹ Stop")

if start_btn:
    st.session_state.run = True
    st.session_state.report = {}  # reset report

if stop_btn:
    st.session_state.run = False

frame_window = st.image([])

# ---------------- Load YOLO Model ----------------
if task == "Object Segmentation":
    model = YOLO("yolov8n-seg.pt")
else:
    model = YOLO("yolov8n.pt")

# ---------------- Video Source ----------------
cap = None

if st.session_state.run:
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)

    elif input_source == "Video Upload":
        uploaded_video = st.sidebar.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_video.read())
            cap = cv2.VideoCapture(temp_file.name)

# ---------------- Main Processing Loop ----------------
if st.session_state.run and cap is not None:
    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # -------- Object Counting (Class-wise Report) --------
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name in st.session_state.report:
                    st.session_state.report[cls_name] += 1
                else:
                    st.session_state.report[cls_name] = 1

        # -------- Display Count on Frame --------
        if task == "Object Counting":
            total = sum(st.session_state.report.values())
            cv2.putText(
                annotated_frame,
                f"Count: {total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        frame_window.image(annotated_frame, channels="BGR")

    cap.release()

# ---------------- Final Detection Report ----------------
if not st.session_state.run and st.session_state.report:
    st.subheader("📊 Detection Report")

    for obj, count in st.session_state.report.items():
        st.write(f"**{obj}** : {count}")

    st.success(
        f"Total Objects Detected: {sum(st.session_state.report.values())}"
    )
