import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from datetime import datetime
import threading
import queue
import time
import logging
import traceback

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

# Custom CSS (same as original)
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
    .webrtc-container {
        border: 2px solid #c5d9f1;
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
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
</style>
""", unsafe_allow_html=True)

# Enhanced WebRTC configuration (same as original)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
        {"urls": ["stun:stun.voiparound.com"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject", 
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        }
    ],
    "iceCandidatePoolSize": 10
})

class ExamDetector:
    def __init__(self, use_yolo=False):
        self.initialized = False
        self.initialization_error = None
        
        try:
            if not MP_AVAILABLE:
                raise RuntimeError("MediaPipe is required. Install mediapipe package.")

            # Update connection status
        if webrtc_ctx.state.playing:
            connection_placeholder.markdown(
                '<div class="connection-status connected">🟢 Stream Active - Real-time monitoring in progress</div>', 
                unsafe_allow_html=True
            )
            st.session_state.connection_state = "connected"
        else:
            connection_placeholder.markdown(
                '<div class="connection-status disconnected">🔴 Stream Inactive - Click START to begin monitoring</div>', 
                unsafe_allow_html=True
            )
            st.session_state.connection_state = "disconnected"
        
        # Troubleshooting tips
        if not webrtc_ctx.state.playing:
            st.info("""
            **🔧 Connection Issues?**
            1. Allow camera permissions in your browser
            2. Try refreshing the page
            3. Check if other apps are using your camera
            4. Try a different browser (Chrome/Firefox recommended)
            5. Check your internet connection
            """)

    with col2:
        st.markdown("### 🎛️ Control Panel")
        
        # Real-time violations display
        st.markdown("### 🚨 Live Violations")
        
        violations_placeholder = st.empty()
        
        # Update violations display
        if st.session_state.exam_detector and st.session_state.exam_detector.is_ready():
            detector = st.session_state.exam_detector
            
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
            if st.session_state.exam_detector is None:
                violations_placeholder.markdown('<div class="violation-box">Detector not initialized. Click "Initialize" to start.</div>', unsafe_allow_html=True)
            else:
                error_msg = st.session_state.exam_detector.initialization_error or "Unknown initialization error"
                violations_placeholder.markdown(f'<div class="violation-box">Detector initialization failed: {error_msg}</div>', unsafe_allow_html=True)
        
        # System status
        st.markdown("### ⚙️ System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("MediaPipe", "✅" if MP_AVAILABLE else "❌")
        with status_col2:
            st.metric("YOLO", "✅" if YOLO_AVAILABLE else "❌")
        
        # Detector status
        if st.session_state.exam_detector:
            detector_status = "✅ Ready" if st.session_state.exam_detector.is_ready() else "❌ Error"
            st.metric("Detector", detector_status)
            
            if st.session_state.exam_detector.error_count > 0:
                st.warning(f"⚠️ {st.session_state.exam_detector.error_count} processing errors detected")
        
        # Instructions
        with st.expander("📋 Instructions", expanded=False):
            st.markdown("""
            **🚀 Getting Started:**
            1. Click **"Initialize"** to load the detector
            2. Click **"START"** on the video stream
            3. Allow camera permissions in your browser
            4. Monitor real-time violations on the right
            
            **🎯 What We Monitor:**
            - 👤 Face presence and tracking
            - 👁️ Gaze direction (looking away)
            - 📱 Phone/device detection
            - 📄 Paper/book detection
            - 🚶 Excessive movement
            
            **⚡ Real-Time Features:**
            - Zero-latency WebRTC streaming
            - Live violation alerts
            - Continuous monitoring
            - Browser-based (no downloads needed)
            
            **🔧 Troubleshooting:**
            - If connection fails, try refreshing the page
            - Ensure camera permissions are granted
            - Use Chrome or Firefox for best compatibility
            - Close other applications using the camera
            - If you see "Callback Error", click Initialize first
            
            **🐛 Error Handling:**
            - The system now has robust error recovery
            - Processing errors are logged and handled gracefully
            - Frame processing continues even if some components fail
            - Error counts are tracked and displayed
            """)

        # Debug information (for development)
        if st.checkbox("🐛 Show Debug Info", value=False):
            if st.session_state.exam_detector:
                detector = st.session_state.exam_detector
                st.json({
                    "Initialized": detector.is_ready(),
                    "Error Count": detector.error_count,
                    "Frame Counter": detector.frame_counter,
                    "Use YOLO": detector.use_yolo,
                    "Violations": len(detector.violations),
                    "Initialization Error": detector.initialization_error
                })

        # Auto-refresh for real-time updates (reduced frequency)
        if webrtc_ctx.state.playing and st.session_state.connection_state == "connected":
            time.sleep(2)  # Update every 2 seconds
            st.rerun()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Make sure you have installed required packages")
        st.code("""
        # Install required packages:
        pip install streamlit-webrtc mediapipe opencv-python numpy
        
        # Optional for better detection:
        pip install ultralytics
        """)
        
        # Show full traceback for debugging
        with st.expander("🔍 Full Error Details"):
            st.text(traceback.format_exc()) Initialize MediaPipe with more robust settings
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3
            )

            self.prev_gray = None
            self.movement_window = []
            self.movement_threshold = 50000

            self.violations = []
            self.total_violations = 0
            self.violations_lock = threading.Lock()

            # YOLO initialization with better error handling
            self.use_yolo = False
            self.yolo_model = None
            if use_yolo and YOLO_AVAILABLE:
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                    self.use_yolo = True
                    print("✅ YOLO model loaded successfully")
                except Exception as e:
                    print(f'⚠️ YOLO load failed: {e}')
                    self.use_yolo = False

            self.frame_counter = 0
            self.detection_interval = 5
            self.last_process_time = time.time()
            self.error_count = 0
            self.max_errors = 10
            
            self.initialized = True
            print("✅ ExamDetector initialized successfully")
            
        except Exception as e:
            self.initialization_error = str(e)
            print(f"❌ ExamDetector initialization failed: {e}")
            print(traceback.format_exc())

    def is_ready(self):
        """Check if detector is ready for processing"""
        return self.initialized and self.initialization_error is None

    def detect_face_mesh(self, frame):
        """Detect face using MediaPipe Face Mesh with enhanced error handling"""
        if not self.is_ready():
            return None, None, False
            
        try:
            # Ensure frame is in correct format
            if frame is None or frame.size == 0:
                return None, None, False
                
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return None, None, False
            
            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(img_rgb)
            
            if not results.multi_face_landmarks:
                return None, None, False

            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates with bounds checking
            pts = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                # Ensure coordinates are within bounds
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                pts.append((x, y))

            if not pts:
                return None, None, False

            # Calculate face bounding box with safety checks
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            
            if not xs or not ys:
                return None, None, False
                
            x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
            x2, y2 = min(max(xs) + 10, w - 1), min(max(ys) + 10, h - 1)
            
            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                return None, None, False
                
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
                try:
                    eye_x = (left_center[0] + right_center[0]) / 2
                    face_center_x = (x1 + x2) / 2
                    deviation = abs(eye_x - face_center_x)
                    looking_forward = deviation < (bbox[2] * 0.3)
                except:
                    looking_forward = True

            return bbox, (left_center, right_center), looking_forward
            
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 3:  # Only log first few errors
                print(f"Face detection error: {e}")
            return None, None, False

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO with enhanced error handling"""
        phone_detected = False
        paper_detected = False
        
        if not self.use_yolo or not self.is_ready():
            return phone_detected, paper_detected

        try:
            if frame is None or frame.size == 0:
                return phone_detected, paper_detected
                
            # Run YOLO inference with lower image size for performance
            results = self.yolo_model(frame, conf=0.4, verbose=False, imgsz=320)
            
            if not results:
                return phone_detected, paper_detected
                
            names = self.yolo_model.names

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                    
                for box in r.boxes:
                    try:
                        cls_id = int(box.cls[0])
                        name = names[cls_id].lower()
                        conf = float(box.conf[0])
                        
                        if conf < 0.4:  # Skip low confidence detections
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Ensure valid coordinates
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w-1, x2), min(h-1, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Phone detection
                        phone_classes = ['cell phone', 'mobile phone', 'phone', 'remote']
                        if any(phone_class in name for phone_class in phone_classes):
                            phone_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, 'Phone', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Paper detection
                        paper_classes = ['book', 'paper', 'notebook']
                        if any(paper_class in name for paper_class in paper_classes):
                            paper_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, 'Paper', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    except Exception as box_error:
                        continue  # Skip problematic boxes

        except Exception as e:
            self.error_count += 1
            if self.error_count <= 3:
                print(f"YOLO detection error: {e}")

        return phone_detected, paper_detected

    def detect_objects_heuristic(self, frame):
        """Basic object detection using computer vision with better error handling"""
        try:
            if frame is None or frame.size == 0:
                return False, False
                
            phone = self.detect_phone_basic(frame)
            paper = self.detect_paper_basic(frame)
            return phone, paper
        except Exception as e:
            if self.error_count <= 3:
                print(f"Heuristic detection error: {e}")
            return False, False

    def detect_phone_basic(self, frame):
        """Basic phone detection with error handling"""
        try:
            if frame is None or frame.size == 0:
                return False
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 15000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    if 0.4 < aspect_ratio < 2.5:
                        return True
            return False
        except Exception:
            return False

    def detect_paper_basic(self, frame):
        """Basic paper detection with error handling"""
        try:
            if frame is None or frame.size == 0:
                return False
                
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
        """Detect excessive movement with enhanced error handling"""
        try:
            if frame is None or frame.size == 0:
                return False
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is None:
                self.prev_gray = gray.copy()
                return False
            
            # Ensure same dimensions
            if gray.shape != self.prev_gray.shape:
                self.prev_gray = gray.copy()
                return False
            
            diff = cv2.absdiff(self.prev_gray, gray)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            movement = np.sum(th)
            self.prev_gray = gray.copy()
            
            self.movement_window.append(movement)
            if len(self.movement_window) > 10:
                self.movement_window.pop(0)
            
            return len(self.movement_window) > 5 and np.mean(self.movement_window) > self.movement_threshold
        except Exception:
            return False

    def process_frame(self, frame):
        """Process frame and return annotated result with comprehensive error handling"""
        if not self.is_ready():
            # Draw error message on frame
            if self.initialization_error:
                cv2.putText(frame, f"Init Error: {self.initialization_error[:40]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        try:
            # Validate input frame
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
            if frame.size == 0:
                return frame
                
            # Ensure frame has proper dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return frame
                
            self.frame_counter += 1
            annotated = frame.copy()
            
            # Throttle processing for performance
            current_time = time.time()
            should_process = (current_time - self.last_process_time > 0.3)  # Process every 300ms

            # Face detection (always do this as it's most important)
            try:
                face_result = self.detect_face_mesh(frame)
                face_bbox = face_result[0] if face_result else None
                looking_forward = face_result[2] if face_result else False
            except Exception as e:
                face_bbox = None
                looking_forward = False
                if self.error_count <= 3:
                    print(f"Face detection failed: {e}")

            # Draw face rectangle if detected
            if face_bbox is not None:
                try:
                    x, y, w, h = face_bbox
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except:
                    pass

            phone = paper = False
            excessive_movement = False
            
            if should_process:
                self.last_process_time = current_time
                
                # Object detection
                try:
                    if self.use_yolo:
                        phone, paper = self.detect_objects_yolo(annotated.copy())  # Work on copy
                    else:
                        phone, paper = self.detect_objects_heuristic(annotated.copy())
                except Exception as e:
                    if self.error_count <= 3:
                        print(f"Object detection failed: {e}")
                
                # Movement detection
                try:
                    excessive_movement = self.detect_movement(frame)
                except Exception as e:
                    if self.error_count <= 3:
                        print(f"Movement detection failed: {e}")
                
                # Check and record violations
                try:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    with self.violations_lock:
                        if face_bbox is None:
                            self._add_violation('Face Absent', timestamp)
                        elif not looking_forward:
                            self._add_violation('Looking Away', timestamp)
                        if phone:
                            self._add_violation('Phone Detected', timestamp)
                        if paper:
                            self._add_violation('Paper Detected', timestamp)
                        if excessive_movement:
                            self._add_violation('Excessive Movement', timestamp)
                except Exception as e:
                    if self.error_count <= 3:
                        print(f"Violation recording failed: {e}")

            # Draw status overlay
            try:
                self._draw_status(annotated, face_bbox is not None, looking_forward, phone, paper, excessive_movement)
            except Exception as e:
                if self.error_count <= 3:
                    print(f"Status drawing failed: {e}")
            
            return annotated
            
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 5:
                print(f"Frame processing error: {e}")
                print(traceback.format_exc())
            
            # Return frame with error message
            try:
                if frame is not None and frame.size > 0:
                    cv2.putText(frame, f"Processing Error", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Errors: {self.error_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    return frame
                else:
                    return np.zeros((480, 640, 3), dtype=np.uint8)
            except:
                return np.zeros((480, 640, 3), dtype=np.uint8)

    def _add_violation(self, vtype, time):
        """Add violation if not duplicate"""
        try:
            if not self.violations or self.violations[-1]['type'] != vtype:
                self.total_violations += 1
                self.violations.append({'type': vtype, 'time': time})
                # Limit violation history to prevent memory issues
                if len(self.violations) > 100:
                    self.violations = self.violations[-50:]  # Keep last 50
        except Exception as e:
            print(f"Violation logging error: {e}")

    def _draw_status(self, frame, face, looking_forward, phone, paper, movement):
        """Draw status overlay on frame with error handling"""
        try:
            if frame is None or frame.size == 0:
                return
                
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return
            
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (min(400, w-10), min(140, h-10)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Status text
            y = 30
            cv2.putText(frame, f'Violations: {self.total_violations}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 25
            
            status_items = [
                ('Face', face),
                ('Gaze', looking_forward),
                ('Phone', not phone),
                ('Paper', not paper)
            ]
            
            for label, is_ok in status_items:
                if y >= h - 20:  # Prevent drawing outside frame
                    break
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                status = 'OK' if is_ok else 'VIOLATION'
                cv2.putText(frame, f'{label}: {status}', 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 22
        except Exception as e:
            if self.error_count <= 3:
                print(f"Status drawing error: {e}")

# Global detector instance (accessible from callback thread)
GLOBAL_DETECTOR = None
GLOBAL_DETECTOR_LOCK = threading.Lock()

# Session state initialization
if 'exam_detector' not in st.session_state:
    st.session_state.exam_detector = None
if 'connection_state' not in st.session_state:
    st.session_state.connection_state = "disconnected"

def set_global_detector(detector):
    """Thread-safe way to set the global detector"""
    global GLOBAL_DETECTOR
    with GLOBAL_DETECTOR_LOCK:
        GLOBAL_DETECTOR = detector

def get_global_detector():
    """Thread-safe way to get the global detector"""
    global GLOBAL_DETECTOR
    with GLOBAL_DETECTOR_LOCK:
        return GLOBAL_DETECTOR

def video_frame_callback(frame):
    """Enhanced WebRTC video frame callback using global detector"""
    try:
        # Convert frame to numpy array with error handling
        if frame is None:
            return av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
        
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as e:
            print(f"Frame conversion error: {e}")
            return av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
        
        # Validate frame dimensions and data
        if img is None or img.size == 0:
            return av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
        
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Invalid frame shape: {img.shape}")
            return av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
        
        # Get detector from global variable (thread-safe)
        detector = get_global_detector()
        
        # Process frame if detector is available and ready
        if detector is not None and detector.is_ready():
            try:
                processed_img = detector.process_frame(img.copy())
                
                # Validate processed image
                if processed_img is None or processed_img.size == 0:
                    processed_img = img
                    
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as process_error:
                print(f"Frame processing error: {process_error}")
                # Return original frame with error indicator
                cv2.putText(img, "Processing Error", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            # Detector not ready - show status on frame
            cv2.putText(img, "Detector Not Ready", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, "Click Initialize", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
    except Exception as e:
        print(f"Frame callback error: {e}")
        # Return a black frame as last resort
        try:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Callback Error", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(error_frame, format="bgr24")
        except:
            return frame  # Return original frame if all else fails

def main():
    st.markdown('<div class="title">🎓 Real-Time Exam Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">WebRTC-powered real-time exam proctoring</div>', unsafe_allow_html=True)

    # Check MediaPipe availability
    if not MP_AVAILABLE:
        st.error("❌ MediaPipe not available. Please install: pip install mediapipe")
        st.info("Run: `pip install mediapipe streamlit-webrtc`")
        return

    # Sidebar configuration
    st.sidebar.header("🔧 Settings")
    
    # Connection troubleshooting
    st.sidebar.subheader("🌐 Connection")
    if st.sidebar.button("🔧 Test Connection"):
        st.sidebar.info("Testing WebRTC connection... Check browser console for details.")
    
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
                    detector = ExamDetector(use_yolo=use_yolo)
                    if detector.is_ready():
                        st.session_state.exam_detector = detector
                        set_global_detector(detector)  # Set global detector for callback
                        st.success("✅ Detector initialized!")
                    else:
                        st.error(f"❌ Initialization failed: {detector.initialization_error}")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
    
    with col_btn2:
        if st.button("🔄 Reset"):
            if st.session_state.exam_detector:
                st.session_state.exam_detector.violations = []
                st.session_state.exam_detector.total_violations = 0
                st.session_state.exam_detector.error_count = 0
                # Also reset global detector
                set_global_detector(st.session_state.exam_detector)
            st.success("✅ Violations reset!")

    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 📹 Real-Time Video Stream")
        
        # Connection status indicator
        connection_placeholder = st.empty()
        
        # WebRTC Streamer (unchanged WebRTC configuration)
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
        
