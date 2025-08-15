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
    print(f"✅ MediaPipe imported successfully (version: {mp.__version__})")
except Exception as e:
    MP_AVAILABLE = False
    print(f"❌ MediaPipe import failed: {e}")
    print("💡 Install MediaPipe: pip install mediapipe")

# Optional: ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ Ultralytics YOLO imported successfully")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"❌ Ultralytics import failed: {e}")
    print("💡 Install Ultralytics: pip install ultralytics")

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
</style>
""", unsafe_allow_html=True)

# Enhanced WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
    ],
    "iceCandidatePoolSize": 10
})

class ExamDetector:
    def __init__(self, use_yolo=False):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required. Install mediapipe package.")

        # Initialize MediaPipe
        self.mp_face_mesh = None
        self.face_mesh = None
        self.use_mediapipe = False
        
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            print("✅ MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"❌ MediaPipe initialization failed: {e}")
            print("⚠️ Using fallback face detection (OpenCV)")
            self.use_mediapipe = False

        # Tracking
        self.tracker = None
        self.tracking = False
        
        # Movement detection
        self.prev_gray = None
        self.movement_window = []
        self.movement_threshold = 80000

        # Violations
        self.violations = []
        self.total_violations = 0
        self.violations_lock = threading.Lock()

        # YOLO setup
        self.use_yolo = False
        self.yolo_model = None
        if use_yolo and YOLO_AVAILABLE:
            try:
                print("🔄 Loading YOLO model...")
                self.yolo_model = YOLO('yolov8n.pt')
                self.use_yolo = True
                print("✅ YOLO model loaded successfully")
            except Exception as e:
                print(f'❌ YOLO load failed: {e}')
                print("⚠️ Continuing without YOLO - using basic detection")
                self.use_yolo = False
        elif use_yolo and not YOLO_AVAILABLE:
            print("⚠️ YOLO requested but not available - using basic detection")
        else:
            print("ℹ️ Using basic object detection (no YOLO)")

        # Frame processing
        self.frame_counter = 0
        self.detection_interval = 3
        self.last_process_time = time.time()
        
        # Face tracking
        self.face_lost_counter = 0
        self.face_lost_threshold = 15

        # Violation accumulation (like in working code)
        self._counters = {k: 0 for k in ['absent', 'looking_away', 'phone', 'paper', 'movement']}
        self._thresholds = {
            'absent': 50,
            'looking_away': 100,
            'phone': 5,
            'paper': 3,
            'movement': 110
        }

        print("✅ ExamDetector initialized successfully")

    def detect_face_mesh(self, frame):
        """Detect face using MediaPipe Face Mesh or OpenCV fallback"""
        if self.use_mediapipe and self.face_mesh is not None:
            return self._detect_face_mediapipe(frame)
        else:
            return self._detect_face_opencv(frame)
    
    def _detect_face_mediapipe(self, frame):
        """Detect face using MediaPipe Face Mesh"""
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
            print(f"MediaPipe face detection error: {e}")
            return None, None, False
    
    def _detect_face_opencv(self, frame):
        """Fallback face detection using OpenCV"""
        try:
            # Load OpenCV face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                bbox = (x, y, w, h)
                
                # Simple gaze detection (assume looking forward if face is detected)
                looking_forward = True
                
                # Estimate eye positions
                eye_y = y + int(h * 0.4)
                left_eye_x = x + int(w * 0.3)
                right_eye_x = x + int(w * 0.7)
                left_center = (left_eye_x, eye_y)
                right_center = (right_eye_x, eye_y)
                
                return bbox, (left_center, right_center), looking_forward
            else:
                return None, None, False
                
        except Exception as e:
            print(f"OpenCV face detection error: {e}")
            return None, None, False

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO"""
        phone_detected = False
        paper_detected = False
        
        if not self.use_yolo:
            return phone_detected, paper_detected

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
        """Accumulate violations with thresholds (like in working code)"""
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
        """Process frame and return annotated result (like working code architecture)"""
        try:
            self.frame_counter += 1
            annotated = frame.copy()

            face_bbox = None
            left_right = (None, None)
            looking_forward = True

            # Face tracking with CSRT (like working code)
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

            # Violation processing (like working code)
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
        """Draw status overlay on frame (like working code)"""
        try:
            h, w = frame.shape[:2]
            
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (420, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            y = 35
            cv2.putText(frame, f'Total Violations: {self.total_violations}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 30
            
            # Show detection method
            method = "MediaPipe" if self.use_mediapipe else "OpenCV"
            cv2.putText(frame, f'Detection: {method}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 24
            
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

# THREAD-SAFE GLOBAL DETECTOR - This is the key!
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
        print(f"🔧 Global detector set: {detector is not None}")

def debug_detector_status():
    """Debug function to check detector status"""
    detector = get_global_detector()
    if detector:
        print(f"✅ Detector exists: {type(detector)}")
        print(f"   - MediaPipe: {detector.use_mediapipe}")
        print(f"   - Face mesh: {detector.face_mesh is not None}")
        print(f"   - YOLO: {detector.use_yolo}")
        print(f"   - Frame counter: {detector.frame_counter}")
        print(f"   - Detection method: {'MediaPipe' if detector.use_mediapipe else 'OpenCV'}")
    else:
        print("❌ No detector found")

def video_frame_callback(frame):
    """WebRTC video frame callback with thread-safe detector access"""
    try:
        img = frame.to_ndarray(format="bgr24")
        
        # Get detector in thread-safe way
        detector = get_global_detector()
        
        if detector is not None:
            try:
                # Process the frame
                processed_img = detector.process_frame(img)
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as e:
                print(f"Frame processing error in callback: {e}")
                # Return frame with error message
                cv2.putText(img, f"Processing Error: {str(e)[:40]}...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(img, "Detector initialized but processing failed", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            # Return original frame with initialization message
            cv2.putText(img, "Detector not initialized - Click Initialize button", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, "System ready for detection", 
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
    st.markdown('<div class="subtitle">WebRTC-powered real-time exam proctoring</div>', unsafe_allow_html=True)

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
    if 'init_message' not in st.session_state:
        st.session_state.init_message = ""

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
            with st.spinner("Initializing detector..."):
                try:
                    # Clear any previous detector
                    set_global_detector(None)
                    st.session_state.detector_initialized = False
                    
                    # Create detector and set it globally (thread-safe)
                    print("🔄 Creating ExamDetector...")
                    new_detector = ExamDetector(use_yolo=use_yolo)
                    set_global_detector(new_detector)
                    st.session_state.detector_initialized = True
                    st.session_state.init_message = "✅ Detector initialized successfully!"
                    st.success("✅ Detector initialized!")
                    print("✅ Global detector set successfully")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    print(f"❌ Initialization error: {e}")
                    set_global_detector(None)
                    st.session_state.detector_initialized = False
                    st.session_state.init_message = f"❌ Initialization failed: {str(e)}"
                    st.error(f"❌ Initialization failed: {str(e)}")
                    st.info("💡 Try installing required packages: pip install mediapipe opencv-python")
    
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
    
    # Debug button
    if st.sidebar.button("🐛 Debug Status"):
        debug_detector_status()
        detector = get_global_detector()
        if detector:
            st.success("✅ Detector is initialized and working")
            st.info(f"Frame counter: {detector.frame_counter}")
            st.info(f"Detection method: {'MediaPipe' if detector.use_mediapipe else 'OpenCV'}")
            st.info(f"YOLO enabled: {detector.use_yolo}")
        else:
            st.error("❌ Detector is not initialized")
            st.info("Click '🚀 Initialize' to start the detector")

    # Show initialization status
    if st.session_state.init_message:
        if "✅" in st.session_state.init_message:
            st.markdown(f'<div class="init-success">{st.session_state.init_message}</div>', 
                       unsafe_allow_html=True)
        else:
            st.error(st.session_state.init_message)

    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 📹 Real-Time Video Stream")
        
        connection_placeholder = st.empty()
        
        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="exam-monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 30},
                "audio": False
            },
            async_processing=True,
        )
        
        # Update connection status
        if webrtc_ctx.state.playing:
            if st.session_state.detector_initialized:
                connection_placeholder.markdown(
                    '<div class="connection-status connected">🟢 Stream Active - Real-time monitoring in progress</div>', 
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
                '<div class="connection-status disconnected">🔴 Stream Inactive - Click START to begin monitoring</div>', 
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

        # Auto-refresh for real-time updates
        if (webrtc_ctx.state.playing and 
            st.session_state.connection_state == "connected" and 
            st.session_state.detector_initialized):
            time.sleep(2)
            st.rerun()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Make sure you have installed: pip install streamlit-webrtc mediapipe opencv-python")
