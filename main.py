import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from datetime import datetime
import threading
import time
import logging

# Configure logging to suppress excessive WebRTC errors
logging.getLogger('aioice').setLevel(logging.ERROR)
logging.getLogger('aiortc').setLevel(logging.ERROR)

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
    .connection-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .init-success {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .loading-status {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Lightweight WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ],
    "iceCandidatePoolSize": 5
})

class ExamDetector:
    def __init__(self, use_yolo=False):
        # Immediate initialization - no heavy operations
        self.use_yolo_setting = use_yolo
        
        # Lazy initialization flags
        self.mediapipe_initialized = False
        self.yolo_initialized = False
        self.initialization_error = None
        
        # Basic attributes
        self.tracker = None
        self.tracking = False
        self.prev_gray = None
        self.movement_window = []
        self.movement_threshold = 80000
        self.violations = []
        self.total_violations = 0
        self.violations_lock = threading.Lock()
        self.frame_counter = 0
        self.detection_interval = 3
        self.face_lost_counter = 0
        self.face_lost_threshold = 15
        
        # Violation accumulation
        self._counters = {k: 0 for k in ['absent', 'looking_away', 'phone', 'paper', 'movement']}
        self._thresholds = {
            'absent': 50,
            'looking_away': 100,
            'phone': 5,
            'paper': 3,
            'movement': 110
        }
        
        # These will be initialized lazily
        self.mp_face_mesh = None
        self.face_mesh = None
        self.yolo_model = None
        self.use_yolo = False
        
        print("✅ ExamDetector created (lazy initialization)")

    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe"""
        if self.mediapipe_initialized or not MP_AVAILABLE:
            return
        
        try:
            print("🔄 Initializing MediaPipe...")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mediapipe_initialized = True
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            self.initialization_error = f"MediaPipe init failed: {e}"
            print(f"❌ MediaPipe initialization failed: {e}")

    def _init_yolo(self):
        """Lazy initialization of YOLO"""
        if self.yolo_initialized or not self.use_yolo_setting or not YOLO_AVAILABLE:
            return
        
        try:
            print("🔄 Initializing YOLO...")
            self.yolo_model = YOLO('yolov8n.pt')
            self.use_yolo = True
            self.yolo_initialized = True
            print("✅ YOLO initialized successfully")
        except Exception as e:
            self.initialization_error = f"YOLO init failed: {e}"
            print(f"❌ YOLO initialization failed: {e}")

    def get_initialization_status(self):
        """Get current initialization status"""
        mediapipe_status = "✅" if self.mediapipe_initialized else ("🔄" if MP_AVAILABLE else "❌")
        yolo_status = "✅" if self.yolo_initialized else ("🔄" if (self.use_yolo_setting and YOLO_AVAILABLE) else "➖")
        
        if self.initialization_error:
            return f"❌ {self.initialization_error}", False
        elif self.mediapipe_initialized and (not self.use_yolo_setting or self.yolo_initialized):
            return "✅ Fully initialized", True
        else:
            return f"🔄 Initializing... MediaPipe:{mediapipe_status} YOLO:{yolo_status}", False

    def detect_face_mesh(self, frame):
        """Detect face using MediaPipe Face Mesh with lazy initialization"""
        # Lazy initialize MediaPipe
        if not self.mediapipe_initialized:
            self._init_mediapipe()
        
        if not self.mediapipe_initialized or self.face_mesh is None:
            return None, None, False

        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            h, w = frame.shape[:2]
            
            if not results.multi_face_landmarks:
                return None, None, False

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
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None, False

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO with lazy initialization"""
        # Lazy initialize YOLO
        if not self.yolo_initialized:
            self._init_yolo()
        
        if not self.yolo_initialized or self.yolo_model is None:
            return False, False

        phone_detected = False
        paper_detected = False

        try:
            results = self.yolo_model(frame, conf=0.3, verbose=False, imgsz=640)
            names = self.yolo_model.names

            for r in results:
                if r.boxes is None:
                    continue
                    
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = names[cls_id].lower()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Phone detection
                    phone_classes = ['cell phone', 'mobile phone', 'phone', 'remote']
                    if any(phone_class in name for phone_class in phone_classes):
                        phone_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Phone', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Paper detection
                    paper_classes = ['book', 'paper', 'notebook']
                    if any(paper_class in name for paper_class in paper_classes):
                        paper_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Paper', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        except Exception as e:
            print(f"YOLO detection error: {e}")

        return phone_detected, paper_detected

    def detect_objects_heuristic(self, frame):
        """Basic object detection using computer vision"""
        try:
            phone = self.detect_phone_basic(frame)
            paper = self.detect_paper_basic(frame)
            return phone, paper
        except Exception as e:
            print(f"Heuristic detection error: {e}")
            return False, False

    def detect_phone_basic(self, frame):
        """Basic phone detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 15000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h else 0
                    if 0.4 < aspect_ratio < 2.5:
                        return True
            return False
        except Exception:
            return False

    def detect_paper_basic(self, frame):
        """Basic paper detection"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 8000:
                    return True
            return False
        except Exception:
            return False

    def detect_movement(self, frame):
        """Detect excessive movement"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return False
            
            diff = cv2.absdiff(self.prev_gray, gray)
            _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            movement = np.sum(th)
            self.prev_gray = gray.copy()
            
            self.movement_window.append(movement)
            if len(self.movement_window) > 20:
                self.movement_window.pop(0)
            
            return np.mean(self.movement_window) > self.movement_threshold
        except Exception:
            return False

    def _accumulate_violation(self, vtype, flag):
        """Accumulate violations with thresholds"""
        if flag:
            # Phone and paper are detected immediately
            if vtype in ['phone', 'paper']:
                with self.violations_lock:
                    self.total_violations += 1
                    t = datetime.now().strftime('%H:%M:%S')
                    self.violations.append({'type': vtype, 'time': t})
                    print(f'Violation: {vtype} at {t}')
                return

            # Other violations use accumulation
            self._counters[vtype] += 1
        else:
            self._counters[vtype] = max(0, self._counters[vtype] - 2)

        # Check threshold for accumulated violations
        if self._counters[vtype] > self._thresholds[vtype]:
            self._counters[vtype] = 0
            with self.violations_lock:
                self.total_violations += 1
                t = datetime.now().strftime('%H:%M:%S')
                self.violations.append({'type': vtype, 'time': t})
                print(f'Violation: {vtype} at {t}')

    def process_frame(self, frame):
        """Process frame and return annotated result with lazy initialization"""
        try:
            self.frame_counter += 1
            annotated = frame.copy()

            # Get initialization status
            init_status, fully_initialized = self.get_initialization_status()
            
            # Always show current status
            cv2.putText(annotated, f"Status: {init_status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # If not fully initialized, show basic info and return
            if not fully_initialized:
                cv2.putText(annotated, "Initializing models in background...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated, f"Frames processed: {self.frame_counter}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                return annotated

            # Full processing when initialized
            face_bbox = None
            left_right = (None, None)
            looking_forward = True

            # Face tracking with CSRT
            if self.tracking and (self.frame_counter % self.detection_interval != 0):
                try:
                    ok, bbox = self.tracker.update(frame)
                    if ok:
                        x, y, w, h = map(int, bbox)
                        face_bbox = (x, y, w, h)
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        self.tracking = False
                        self.tracker = None
                except Exception as e:
                    print(f"Tracker error: {e}")
                    self.tracking = False
                    self.tracker = None

            # Face detection with MediaPipe
            if not self.tracking or (self.frame_counter % self.detection_interval == 0):
                res = self.detect_face_mesh(frame)
                if res[0] is not None:
                    face_bbox, left_right, looking_forward = res
                    x, y, w, h = face_bbox
                    
                    # Initialize CSRT tracker
                    try:
                        self.tracker = cv2.TrackerCSRT_create()
                        self.tracker.init(frame, tuple(face_bbox))
                        self.tracking = True
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Tracker init error: {e}")
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Object detection
            if self.use_yolo:
                phone, paper = self.detect_objects_yolo(annotated)
            else:
                phone, paper = self.detect_objects_heuristic(annotated)

            # Movement detection
            excessive_movement = self.detect_movement(frame)

            # Violation processing
            if face_bbox is None:
                self.face_lost_counter += 1
                if self.face_lost_counter > self.face_lost_threshold:
                    self._accumulate_violation('absent', 1)
            else:
                self.face_lost_counter = 0
                self._accumulate_violation('absent', 0)

            if face_bbox is not None and not looking_forward:
                self._accumulate_violation('looking_away', 1)
            else:
                self._accumulate_violation('looking_away', 0)

            if phone:
                self._accumulate_violation('phone', 1)
            else:
                self._accumulate_violation('phone', 0)

            if paper:
                self._accumulate_violation('paper', 1)
            else:
                self._accumulate_violation('paper', 0)

            if excessive_movement:
                self._accumulate_violation('movement', 1)
            else:
                self._accumulate_violation('movement', 0)

            # Draw status overlay
            self._draw_status(annotated, face_bbox is not None, left_right[0] is not None,
                            looking_forward, phone, paper, excessive_movement)
            
            return annotated
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            cv2.putText(frame, f"Processing Error: {str(e)[:50]}...", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

    def _draw_status(self, frame, face, eyes, gaze, phone, paper, movement):
        """Draw status overlay on frame"""
        try:
            h, w = frame.shape[:2]
            
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 120), (420, 260), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            y = 145
            cv2.putText(frame, f'Total Violations: {self.total_violations}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 30
            
            items = [
                ('Face', face), 
                ('Movement Normal', not movement),
                ('Phone Clear', not phone),
                ('Paper Clear', not paper)
            ]
            
            for label, ok in items:
                color = (0, 255, 0) if ok else (0, 0, 255)
                txt = 'OK' if ok else 'VIOLATION'
                cv2.putText(frame, f'{label}: {txt}', 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 24
        except Exception as e:
            print(f"Status drawing error: {e}")

# THREAD-SAFE GLOBAL DETECTOR
detector_lock = threading.Lock()
global_detector = None

def get_global_detector():
    """Thread-safe getter for detector"""
    with detector_lock:
        return global_detector

def set_global_detector(detector):
    """Thread-safe setter for detector"""
    with detector_lock:
        global global_detector
        global_detector = detector

def video_frame_callback(frame):
    """WebRTC video frame callback with thread-safe detector access"""
    try:
        img = frame.to_ndarray(format="bgr24")
        
        # Get detector in thread-safe way
        detector = get_global_detector()
        
        if detector is not None:
            # Process the frame
            processed_img = detector.process_frame(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        else:
            # Return original frame with initialization message
            cv2.putText(img, "Detector not initialized - Click Initialize button", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, "WebRTC stream ready - waiting for detector", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
    except Exception as e:
        print(f"Frame callback error: {e}")
        try:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Callback Error: {str(e)[:30]}...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except:
            return frame

def main():
    st.markdown('<div class="title">🎓 Real-Time Exam Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">WebRTC-powered real-time exam proctoring with lazy loading</div>', unsafe_allow_html=True)

    # Check MediaPipe availability
    if not MP_AVAILABLE:
        st.error("❌ MediaPipe not available. Please install: pip install mediapipe")
        st.info("Run: `pip install mediapipe streamlit-webrtc`")
        return

    # Initialize session state for UI only
    if 'detector_initialized' not in st.session_state:
        st.session_state.detector_initialized = False
    if 'connection_state' not in st.session_state:
        st.session_state.connection_state = "disconnected"

    # Sidebar configuration
    st.sidebar.header("🔧 Settings")
    
    use_yolo = st.sidebar.checkbox('🎯 Use YOLO Detection', 
                                  value=False,
                                  disabled=not YOLO_AVAILABLE,
                                  help="Enable YOLO for better object detection")
    
    if not YOLO_AVAILABLE:
        st.sidebar.info("💡 Install ultralytics for YOLO: pip install ultralytics")
    
    # Initialize/Reset detector
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("🚀 Initialize"):
            try:
                # Quick initialization - no heavy operations
                new_detector = ExamDetector(use_yolo=use_yolo)
                set_global_detector(new_detector)
                st.session_state.detector_initialized = True
                st.success("✅ Detector created! Models will load in background during streaming.")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                set_global_detector(None)
                st.session_state.detector_initialized = False
                st.error(f"❌ Initialization failed: {e}")
    
    with col_btn2:
        if st.button("🔄 Reset"):
            detector = get_global_detector()
            if detector:
                with detector.violations_lock:
                    detector.violations = []
                    detector.total_violations = 0
                st.success("✅ Violations reset!")
            else:
                st.warning("⚠️ No detector to reset. Initialize first.")

    # Show detector status
    detector = get_global_detector()
    if detector:
        status_msg, fully_ready = detector.get_initialization_status()
        if fully_ready:
            st.sidebar.markdown(f'<div class="init-success">{status_msg}</div>', 
                               unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="loading-status">{status_msg}</div>', 
                               unsafe_allow_html=True)

    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 📹 Real-Time Video Stream")
        st.info("💡 **New**: Models load in the background! You can start the stream immediately after clicking Initialize.")
        
        connection_placeholder = st.empty()
        
        # WebRTC Streamer - lightweight config for faster connection
        webrtc_ctx = webrtc_streamer(
            key="exam-monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 15},  # Reduced framerate for stability
                "audio": False
            },
            async_processing=True,
        )
        
        # Update connection status
        if webrtc_ctx.state.playing:
            if st.session_state.detector_initialized:
                connection_placeholder.markdown(
                    '<div class="connection-status connected">🟢 Stream Active - AI models loading in background</div>', 
                    unsafe_allow_html=True
                )
            else:
                connection_placeholder.markdown(
                    '<div class="connection-status disconnected">🟡 Stream Active but Detector Not Initialized</div>', 
                    unsafe_allow_html=True
                )
            st.session_state.connection_state = "connected"
        else:
            connection_placeholder.markdown(
                '<div class="connection-status disconnected">🔴 Stream Inactive - Click START to begin</div>', 
                unsafe_allow_html=True
            )
            st.session_state.connection_state = "disconnected"

    with col2:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("### 🚨 Live Violations")
        
        violations_placeholder = st.empty()
        
        # Update violations display using global detector
        detector = get_global_detector()
        if detector and st.session_state.detector_initialized:
            try:
                with detector.violations_lock:
                    # Metrics
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Total", detector.total_violations)
                    with col_m2:
                        st.metric("Frames", detector.frame_counter)
                    
                    # Recent violations
                    if detector.violations:
                        violation_text = '<div class="violation-box">'
                        recent_violations = detector.violations[-8:]  # Last 8 violations
                        
                        for i, v in enumerate(reversed(recent_violations), 1):
                            violation_text += f'<p><strong>#{len(recent_violations)-i+1}</strong> {v["type"]} at {v["time"]}</p>'
                        
                        violation_text += '</div>'
                        violations_placeholder.markdown(violation_text, unsafe_allow_html=True)
                    else:
                        violations_placeholder.markdown('<div class="violation-box">No violations detected. 🟢</div>', unsafe_allow_html=True)
            except Exception as e:
                violations_placeholder.markdown(f'<div class="violation-box">Error reading violations: {e}</div>', unsafe_allow_html=True)
        else:
            violations_placeholder.markdown('<div class="violation-box">Detector not initialized. Click "🚀 Initialize" to start.</div>', unsafe_allow_html=True)
        
        # System status
        st.markdown("### ⚙️ System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.metric("MediaPipe", "✅" if MP_AVAILABLE else "❌")
        with status_col2:
            st.metric("YOLO", "✅" if YOLO_AVAILABLE else "❌")
        with status_col3:
            st.metric("Detector", "✅" if st.session_state.detector_initialized else "❌")

        # Instructions
        with st.expander("📋 New Lazy Loading Instructions", expanded=True):
            st.markdown("""
            **🚀 Quick Start (No More Timeouts!):**
            1. Click **"🚀 Initialize"** → Instant response ⚡
            2. Click **"START"** immediately → Stream connects fast 🎥
            3. Models load in background while you see the stream 🧠
            4. Full detection starts automatically when models are ready ✅
            
            **🎯 What You'll See:**
            - **First**: Stream with "Initializing models..." overlay
            - **Then**: Basic OpenCV processing (face rectangles, frame counter)
            - **Finally**: Full AI detection (MediaPipe + YOLO if enabled)
            
            **⚡ Why This is Better:**
            - ✅ No WebRTC connection timeouts
            - ✅ Immediate visual feedback
            - ✅ Progressive enhancement
            - ✅ Models load only when needed
            
            **🔧 Troubleshooting:**
            - If stream won't start: Refresh page and try again
            - If models don't load: Check console for errors
            - Connection issues: Allow camera permissions
            """)

        # Auto-refresh for real-time updates (reduced frequency)
        if (webrtc_ctx.state.playing and 
            st.session_state.connection_state == "connected" and 
            st.session_state.detector_initialized):
            time.sleep(3)  # Increased to 3 seconds for stability
            st.rerun()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Make sure you have installed: pip install streamlit-webrtc mediapipe opencv-python")
