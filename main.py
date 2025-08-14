import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
from datetime import datetime

# MediaPipe for face mesh
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Optional: ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Streamlit page config
st.set_page_config(
    page_title="Exam Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f4f7fa;
        padding: 20px;
        border-radius: 12px;
    }
    .title {
        font-size: 2.8rem;
        color: #1a3c5a;
        text-align: center;
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    .stButton>button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
        border: none;
        margin-bottom: 10px;
    }
    .violation-box {
        background-color: #000;
        color: #0f0;
        padding: 16px;
        border-radius: 10px;
        max-height: 320px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #333;
    }
    .violation-box p {
        margin: 8px 0;
        color: #00ff88;
    }
</style>
""", unsafe_allow_html=True)


class ExamDetector:
    def __init__(self, use_yolo=YOLO_AVAILABLE):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required. Install mediapipe package.")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.prev_gray = None
        self.movement_window = []
        self.movement_threshold = 80000

        self.violations = []
        self.total_violations = 0

        self.use_yolo = False
        self.yolo_model = None
        if use_yolo:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.use_yolo = True
            except Exception as e:
                print(f'YOLO load failed: {e}')
                self.use_yolo = False

        self.frame_counter = 0

    def detect_face_mesh(self, frame):
        """Detect face using MediaPipe Face Mesh"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        h, w = frame.shape[:2]
        
        if not results.multi_face_landmarks:
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        # Calculate face bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x2, y2 = min(max(xs) + 10, w - 1), min(max(ys) + 10, h - 1)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # Eye landmarks for gaze detection
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

        # Determine if looking forward
        looking_forward = True
        if left_center and right_center:
            eye_x = (left_center[0] + right_center[0]) / 2
            face_center_x = (x1 + x2) / 2
            deviation = abs(eye_x - face_center_x)
            looking_forward = deviation < (bbox[2] * 0.25)

        return bbox, (left_center, right_center), looking_forward

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO"""
        phone_detected = False
        paper_detected = False
        CONF_THRESHOLD = 0.3

        try:
            results = self.yolo_model(frame, conf=CONF_THRESHOLD, verbose=False)
            names = self.yolo_model.names

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = names[cls_id].lower()
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    phone_classes = ['cell phone', 'mobile phone', 'phone', 'smartphone']
                    if any(phone_class in name for phone_class in phone_classes):
                        phone_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Phone: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    paper_classes = ['book', 'paper', 'notebook', 'magazine']
                    if any(paper_class in name for paper_class in paper_classes):
                        paper_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Paper: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        except Exception as e:
            return self.detect_objects_heuristic(frame)

        return phone_detected, paper_detected

    def detect_objects_heuristic(self, frame):
        """Basic object detection using computer vision techniques"""
        phone = self.detect_phone_advanced(frame)
        paper = self.detect_paper_advanced(frame)
        return phone, paper

    def detect_phone_advanced(self, frame):
        """Advanced phone detection using edge detection and contours"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 25000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h else 0
                if 0.4 < aspect_ratio < 2.0:
                    roi = gray[y:y+h, x:x+w]
                    if roi.size > 0:
                        mean_intensity = np.mean(roi)
                        if mean_intensity < 100:
                            return True
        return False

    def detect_paper_advanced(self, frame):
        """Advanced paper detection using color and shape analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White paper detection
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h else 0
                if 0.7 < aspect_ratio < 1.5:
                    return True
        return False

    def detect_movement(self, frame):
        """Detect excessive movement between frames"""
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
        
        avg_movement = np.mean(self.movement_window)
        return avg_movement > self.movement_threshold

    def process_frame(self, frame):
        """Process a single frame and return annotated result"""
        self.frame_counter += 1
        annotated = frame.copy()

        # Face detection
        face_result = self.detect_face_mesh(frame)
        face_bbox = face_result[0] if face_result[0] is not None else None
        looking_forward = face_result[2] if face_result[0] is not None else False

        # Object detection
        if self.use_yolo:
            phone, paper = self.detect_objects_yolo(annotated)
        else:
            phone, paper = self.detect_objects_heuristic(annotated)

        # Movement detection
        excessive_movement = self.detect_movement(frame)

        # Draw face rectangle if detected
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check violations and record them
        current_time = datetime.now().strftime('%H:%M:%S')
        
        if face_bbox is None:
            self._add_violation('absent', current_time)
        elif not looking_forward:
            self._add_violation('looking_away', current_time)
        if phone:
            self._add_violation('phone', current_time)
        if paper:
            self._add_violation('paper', current_time)
        if excessive_movement:
            self._add_violation('movement', current_time)

        # Draw status overlay
        self._draw_status(annotated, face_bbox is not None, looking_forward, phone, paper, excessive_movement)

        return annotated

    def _add_violation(self, vtype, time):
        """Add violation if not duplicate"""
        if not self.violations or self.violations[-1]['type'] != vtype:
            self.total_violations += 1
            self.violations.append({'type': vtype, 'time': time})

    def _draw_status(self, frame, face, looking_forward, phone, paper, movement):
        """Draw status overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        y = 35
        cv2.putText(frame, f'Total Violations: {self.total_violations}', 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 35
        
        status_items = [
            ('Face Detected', face),
            ('Looking Forward', looking_forward),
            ('Phone Clear', not phone),
            ('Paper Clear', not paper),
            ('Movement Normal', not movement)
        ]
        
        for label, is_ok in status_items:
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            status_text = 'OK' if is_ok else 'VIOLATION'
            cv2.putText(frame, f'{label}: {status_text}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25


def main():
    st.markdown('<div class="title">🎓 Exam Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered exam proctoring with real-time monitoring</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'capture_interval' not in st.session_state:
        st.session_state.capture_interval = 2

    # Check MediaPipe availability
    if not MP_AVAILABLE:
        st.error("❌ MediaPipe not available. Please install: pip install mediapipe")
        return

    # Sidebar
    st.sidebar.header("🔧 Detection Settings")
    use_yolo = st.sidebar.checkbox('🎯 Use YOLO Detection', 
                                  value=YOLO_AVAILABLE, 
                                  disabled=not YOLO_AVAILABLE)
    
    capture_interval = st.sidebar.slider('📷 Capture Interval (seconds)', 1, 5, 2)
    st.session_state.capture_interval = capture_interval

    # Main layout
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.markdown("### 📸 Camera Stream")
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button('🟢 Start Monitoring', disabled=st.session_state.monitoring):
                if st.session_state.detector is None:
                    st.session_state.detector = ExamDetector(use_yolo=use_yolo)
                st.session_state.monitoring = True
                st.success('✅ Monitoring started!')
        
        with col_btn2:
            if st.button('🔴 Stop Monitoring', disabled=not st.session_state.monitoring):
                st.session_state.monitoring = False
                st.success('⏹️ Monitoring stopped!')

        # Camera input and processing
        if st.session_state.monitoring:
            st.info("📷 Monitoring active - capturing frames automatically")
            
            # Create placeholder for real-time updates
            camera_placeholder = st.empty()
            
            # Auto-refresh mechanism
            if 'last_capture' not in st.session_state:
                st.session_state.last_capture = time.time()
            
            current_time = time.time()
            if current_time - st.session_state.last_capture >= st.session_state.capture_interval:
                st.session_state.last_capture = current_time
                
                # Get camera input
                camera_image = st.camera_input("📷 Camera", key=f"camera_{current_time}")
                
                if camera_image is not None:
                    # Convert to OpenCV format
                    image = Image.open(camera_image)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Process frame
                    processed_frame = st.session_state.detector.process_frame(frame)
                    
                    # Display processed image
                    camera_placeholder.image(processed_frame, channels='BGR', caption="📊 Live Analysis")
                
                # Auto-rerun to continue monitoring
                time.sleep(0.5)
                st.rerun()
            
        else:
            st.info("🔴 Monitoring stopped. Click 'Start Monitoring' to begin.")

    with col2:
        st.markdown("### 🎛️ Control Panel")
        
        if st.button('🔄 Reset Violations'):
            if st.session_state.detector:
                st.session_state.detector.violations = []
                st.session_state.detector.total_violations = 0
                st.success('✅ Violations reset!')
        
        st.markdown("---")
        st.markdown("### 🚨 Violations Log")
        
        if st.session_state.detector and st.session_state.detector.violations:
            # Display total
            st.metric("Total Violations", st.session_state.detector.total_violations)
            
            # Display recent violations
            violation_text = '<div class="violation-box">'
            for i, v in enumerate(reversed(st.session_state.detector.violations[-10:]), 1):
                violation_text += f'<p><strong>#{len(st.session_state.detector.violations)-i+1}</strong> {v["type"].replace("_", " ").title()} at {v["time"]}</p>'
            violation_text += '</div>'
            st.markdown(violation_text, unsafe_allow_html=True)
        else:
            st.markdown('<div class="violation-box">No violations detected yet. 🟢</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Try refreshing the page or check your internet connection.")
