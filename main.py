import cv2
import time
import numpy as np
from datetime import datetime
import streamlit as st

# Optional deps
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# WebRTC
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ---------------------------
# Streamlit page config & CSS
# ---------------------------
st.set_page_config(
    page_title="Exam Monitor (Cloud)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { background-color: #f4f7fa; padding: 20px; border-radius: 12px; }
    .title { font-size: 2.6rem; color: #1a3c5a; text-align: center; margin-bottom: 6px; font-weight: 700; letter-spacing: -0.5px; }
    .subtitle { font-size: 1.05rem; color: #555; text-align: center; margin-bottom: 22px; font-weight: 400; }
    .stButton>button { width: 100%; padding: 12px; font-size: 16px; font-weight: bold; border-radius: 10px; transition: all 0.3s ease; border: none; margin-bottom: 10px; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
    .violation-box { background-color: #000; color: #0f0; padding: 16px; border-radius: 10px; max-height: 320px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.6; border: 1px solid #333; }
    .violation-box p { margin: 8px 0; color: #00ff88; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Core detector (no cv2 Tracker — works on cloud w/o opencv-contrib)
# ---------------------------
class ExamDetector:
    def __init__(self, use_yolo=YOLO_AVAILABLE, enable_sound=False):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required. Install mediapipe package.")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Movement
        self.prev_gray = None
        self.movement_window = []
        self.movement_threshold = 80000

        # Violations
        self.violations = []
        self.total_violations = 0

        # YOLO
        self.use_yolo = False
        self.yolo_model = None
        if use_yolo:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                self.use_yolo = True
                print("YOLO model loaded: yolov8n.pt")
            except Exception as e:
                print("YOLO load failed, falling back to heuristics:", e)
                self.use_yolo = False

        # cadence
        self.detection_interval = 3
        self.frame_counter = 0

        # face lost
        self.face_lost_counter = 0
        self.face_lost_threshold = 15

        # Audio (disabled by default on cloud)
        self.enable_sound = enable_sound

    # ---------------- face & gaze ----------------
    def detect_face_mesh(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        h, w = frame.shape[:2]
        if not results.multi_face_landmarks:
            return None, (None, None), True

        face_landmarks = results.multi_face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x2, y2 = min(max(xs) + 10, w - 1), min(max(ys) + 10, h - 1)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
        right_eye_idx = [263, 362, 387, 386, 385, 384, 398]

        def mean_point(idxs):
            pts_sel = [pts[i] for i in idxs if i < len(pts)]
            if not pts_sel:
                return None
            mx = int(sum([p[0] for p in pts_sel]) / len(pts_sel))
            my = int(sum([p[1] for p in pts_sel]) / len(pts_sel))
            return (mx, my)

        left_center = mean_point(left_eye_idx)
        right_center = mean_point(right_eye_idx)

        looking_forward = True
        if left_center and right_center:
            eye_x = (left_center[0] + right_center[0]) / 2
            face_center_x = (x1 + x2) / 2
            deviation = abs(eye_x - face_center_x)
            looking_forward = deviation < (bbox[2] * 0.25)

        return bbox, (left_center, right_center), looking_forward

    # ---------------- YOLO & heuristics ----------------
    def detect_objects_yolo(self, frame):
        phone_detected = False
        paper_detected = False
        CONF_THRESHOLD = 0.3
        MIN_AREA = 800
        MAX_AREA = 100000

        try:
            results = self.yolo_model(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=640)
            names = self.yolo_model.names

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = names[cls_id].lower()
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_AREA or area > MAX_AREA:
                        continue

                    phone_classes = [
                        "cell phone", "mobile phone", "phone", "smartphone",
                        "iphone", "android", "tablet", "remote",
                    ]
                    if any(pc in name for pc in phone_classes) and conf >= CONF_THRESHOLD:
                        phone_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Phone: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    paper_classes = [
                        "book", "paper", "notebook", "magazine",
                        "newspaper", "document", "letter", "card",
                        "envelope", "file",
                    ]
                    if any(pc in name for pc in paper_classes) and conf >= CONF_THRESHOLD:
                        paper_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Paper: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.detect_objects_heuristic(frame)

        return phone_detected, paper_detected

    def detect_objects_heuristic(self, frame):
        return self.detect_phone_advanced(frame), self.detect_paper_advanced(frame)

    def detect_phone_advanced(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 25000:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = w / float(h) if h else 0
                if 0.4 < ar < 2.0:
                    roi = gray[y : y + h, x : x + w]
                    if roi.size > 0 and np.mean(roi) < 100:
                        return True
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
        if lines is not None:
            v, h = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 15 or abs(angle) > 165:
                    h.append(line)
                elif 75 < abs(angle) < 105:
                    v.append(line)
            if len(h) >= 2 and len(v) >= 2:
                return True
        return False

    def detect_paper_advanced(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = w / float(h) if h else 0
                if 0.7 < ar < 1.5:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) >= 4:
                        return True
        lower_cream = np.array([15, 30, 200])
        upper_cream = np.array([35, 80, 255])
        mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
        contours_cream, _ = cv2.findContours(mask_cream, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_cream:
            if cv2.contourArea(cnt) > 12000:
                return True
        return False

    # ---------------- misc ----------------
    def detect_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(self.prev_gray, gray)
        _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        movement = np.sum(th)
        self.prev_gray = gray
        self.movement_window.append(movement)
        if len(self.movement_window) > 20:
            self.movement_window.pop(0)
        avg_m = np.mean(self.movement_window)
        return avg_m > self.movement_threshold

    def process(self, frame):
        self.frame_counter += 1
        annotated = frame.copy()

        face_bbox, left_right, looking_forward = self.detect_face_mesh(frame)

        # Draw face bbox if available
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Objects
        if self.use_yolo:
            phone, paper = self.detect_objects_yolo(annotated)
        else:
            phone, paper = self.detect_objects_heuristic(annotated)

        excessive_movement = self.detect_movement(frame)

        # Face presence -> absent violation
        if face_bbox is None:
            self.face_lost_counter += 1
            if self.face_lost_counter > self.face_lost_threshold:
                self._accumulate_violation("absent", 1)
        else:
            self.face_lost_counter = 0
            self._accumulate_violation("absent", 0)

        # Gaze
        if face_bbox is not None and not looking_forward:
            self._accumulate_violation("looking_away", 1)
        else:
            self._accumulate_violation("looking_away", 0)

        # Phone & paper (instant)
        self._accumulate_violation("phone", 1 if phone else 0)
        self._accumulate_violation("paper", 1 if paper else 0)

        # Movement
        self._accumulate_violation("movement", 1 if excessive_movement else 0)

        self._draw_status(
            annotated,
            face=(face_bbox is not None),
            movement=excessive_movement,
            phone=phone,
            paper=paper,
        )

        return annotated

    def _accumulate_violation(self, vtype, flag):
        if not hasattr(self, "_counters"):
            self._counters = {k: 0 for k in ["absent", "looking_away", "phone", "paper", "movement"]}
            self._thresholds = {"absent": 50, "looking_away": 100, "phone": 5, "paper": 3, "movement": 110}

        if flag:
            # phone & paper: instant
            if vtype in ["phone", "paper"]:
                self.total_violations += 1
                t = datetime.now().strftime("%H:%M:%S")
                self.violations.append({"type": vtype, "time": t})
                return
            self._counters[vtype] += 1
        else:
            self._counters[vtype] = max(0, self._counters[vtype] - 2)

        if self._counters[vtype] > self._thresholds[vtype]:
            self._counters[vtype] = 0
            self.total_violations += 1
            t = datetime.now().strftime("%H:%M:%S")
            self.violations.append({"type": vtype, "time": t})

    def _draw_status(self, frame, face, movement, phone, paper):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        y = 35
        cv2.putText(frame, f"Total Violations: {self.total_violations}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 30
        items = [
            ("Face", face),
            ("Movement Normal", not movement),
            ("Phone Clear", not phone),
            ("Paper Clear", not paper),
        ]
        for label, ok in items:
            color = (0, 255, 0) if ok else (0, 0, 255)
            txt = "OK" if ok else "VIOLATION"
            cv2.putText(frame, f"{label}: {txt}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24

# ---------------------------
# WebRTC video processor
# ---------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self, use_yolo=True, enable_sound=False):
        self.detector = ExamDetector(use_yolo=use_yolo and YOLO_AVAILABLE, enable_sound=enable_sound)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out = self.detector.process(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    # Helpers exposed to Streamlit UI
    def get_violations(self):
        return list(self.detector.violations), self.detector.total_violations

    def reset_violations(self):
        self.detector.violations = []
        self.detector.total_violations = 0

    def save_report(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"exam_report_{ts}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("=== Exam Monitoring Report ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Violations: {self.detector.total_violations}\n\n")
            for i, v in enumerate(self.detector.violations, 1):
                f.write(f"{i}. {v['type'].title()} at {v['time']}\n")
        return fname

# ---------------------------
# UI
# ---------------------------
def main():
    st.markdown('<div class="title">🎓 Exam Monitor (Streamlit Cloud)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered proctoring via WebRTC camera stream</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], gap="large")

    with col2:
        st.markdown("### Control Panel")
        use_yolo_checkbox = st.checkbox("✅ Use YOLO Detection", value=True, help="Enable YOLO for object detection")
        enable_sound = st.checkbox("🔊 Enable Sound Alerts (local only)", value=False, help="Enable beeps locally; often disabled on cloud")
        st.markdown("---")
        save_btn = st.button("💾 Save Report")
        reset_btn = st.button("🔄 Reset Violations")
        st.markdown("---")
        st.markdown("### Violations")
        viol_list = st.empty()

    with col1:
        st.markdown("#### Live Stream")
        ctx = webrtc_streamer(
            key="exam-monitor",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(use_yolo_checkbox, enable_sound),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                ]
            },
        )

    # Handle buttons & violations box
    if ctx and ctx.state.playing and ctx.video_transformer:
        vp: VideoProcessor = ctx.video_transformer  # type: ignore

        if save_btn:
            fname = vp.save_report()
            st.toast(f"📄 Report saved: {fname}")
        if reset_btn:
            vp.reset_violations()
            st.success("🔄 Violations reset")

        violations, total = vp.get_violations()
        if violations:
            html = (
                '<div class="violation-box">'
                + "".join(
                    [
                        f"<p><strong>#{i+1}</strong> {v['type'].title()} at {v['time']}</p>"
                        for i, v in enumerate(violations)
                    ]
                )
                + "</div>"
            )
        else:
            html = '<div class="violation-box">No violations yet.</div>'
        viol_list.markdown(html, unsafe_allow_html=True)
    else:
        # Not playing yet
        viol_list.markdown('<div class="violation-box">Start the stream to see violations...</div>', unsafe_allow_html=True)

    # Sidebar info
    st.sidebar.header("Detection Info", divider="gray")
    st.sidebar.markdown(
        """
**Instant Detection:**
- 📱 Phone: Immediate alert
- 📄 Paper: Immediate alert
- 👁️ Face tracking & gaze
- 🚶 Movement analysis
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Running on Streamlit Cloud: camera access is via your browser using WebRTC. No cv2.VideoCapture(0) on the server.")


if __name__ == "__main__":
    main()
