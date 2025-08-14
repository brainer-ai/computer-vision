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

# Enhanced WebRTC configuration with multiple STUN/TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        # Google STUN servers
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        
        # Additional public STUN servers
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
        {"urls": ["stun:stun.voiparound.com"]},
        
        # TURN servers with fallback options
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
        self.movement_threshold = 50000

        self.violations = []
        self.total_violations = 0
        self.violations_lock = threading.Lock()

        self.use_yolo = False
        self.yolo_model = None
        if use_yolo and YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.use_yolo = True
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f'YOLO load failed: {e}')
                self.use_yolo = False

        self.frame_counter = 0
        self.detection_interval = 5  # Process every 5th frame for performance
        self.last_process_time = time.time()

    def detect_face_mesh(self, frame):
        """Detect face using MediaPipe Face Mesh with error handling"""
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
                looking_forward = deviation < (bbox[2] * 0.3)

            return bbox, (left_center, right_center), looking_forward
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None, False

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO with error handling"""
        phone_detected = False
        paper_detected = False
        
        if not self.use_yolo:
            return phone_detected, paper_detected

        try:
            results = self.yolo_model(frame, conf=0.4, verbose=False, imgsz=416)
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
        """Detect excessive movement with error handling"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return False
            
            diff = cv2.absdiff(self.prev_gray, gray)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            movement = np.sum(th)
            self.prev_gray = gray.copy()
            
            self.movement_window.append(movement)
            if len(self.movement_window) > 10:
                self.movement_window.pop(0)
            
            return np.mean(self.movement_window) > self.movement_threshold
        except Exception:
            return False

    def process_frame(self, frame):
        """Process frame and return annotated result with error handling"""
        try:
            self.frame_counter += 1
            annotated = frame.copy()
            
            # Skip processing on some frames for performance
            current_time = time.time()
            should_process = (current_time - self.last_process_time > 0.2)  # Process every 200ms

            # Face detection (always do this)
            face_result = self.detect_face_mesh(frame)
            face_bbox = face_result[0] if face_result else None
            looking_forward = face_result[2] if face_result else False

            # Draw face rectangle
            if face_bbox is not None:
                x, y, w, h = face_bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            phone = paper = False
            excessive_movement = False
            
            if should_process:
                self.last_process_time = current_time
                
                # Object detection
                if self.use_yolo:
                    phone, paper = self.detect_objects_yolo(annotated)
                else:
                    phone, paper = self.detect_objects_heuristic(annotated)
                
                # Movement detection
                excessive_movement = self.detect_movement(frame)
                
                # Check and record violations
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

            # Draw status overlay
            self._draw_status(annotated, face_bbox is not None, looking_forward, phone, paper, excessive_movement)
            
            return annotated
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            # Return original frame with error message
            cv2.putText(frame, f"Processing Error: {str(e)[:50]}...", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

    def _add_violation(self, vtype, time):
        """Add violation if not duplicate"""
        if not self.violations or self.violations[-1]['type'] != vtype:
            self.total_violations += 1
            self.violations.append({'type': vtype, 'time': time})

    def _draw_status(self, frame, face, looking_forward, phone, paper, movement):
        """Draw status overlay on frame"""
        try:
            h, w = frame.shape[:2]
            
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
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
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                status = 'OK' if is_ok else 'VIOLATION'
                cv2.putText(frame, f'{label}: {status}', 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 22
        except Exception as e:
            print(f"Status drawing error: {e}")

# Global detector instance
if 'exam_detector' not in st.session_state:
    st.session_state.exam_detector = None
if 'connection_state' not in st.session_state:
    st.session_state.connection_state = "disconnected"

def video_frame_callback(frame):
    """WebRTC video frame callback with error handling"""
    try:
        img = frame.to_ndarray(format="bgr24")
        
        if st.session_state.exam_detector is not None:
            # Process the frame
            processed_img = st.session_state.exam_detector.process_frame(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        else:
            # Return original frame if detector not initialized
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print(f"Frame callback error: {e}")
        # Return original frame on error
        try:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Callback Error", (10, 30), 
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
                    st.session_state.exam_detector = ExamDetector(use_yolo=use_yolo)
                    st.success("✅ Detector initialized!")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
    
    with col_btn2:
        if st.button("🔄 Reset"):
            if st.session_state.exam_detector:
                st.session_state.exam_detector.violations = []
                st.session_state.exam_detector.total_violations = 0
            st.success("✅ Violations reset!")

    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 📹 Real-Time Video Stream")
        
        # Connection status indicator
        connection_placeholder = st.empty()
        
        # WebRTC Streamer with enhanced error handling
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
        if st.session_state.exam_detector:
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
            violations_placeholder.markdown('<div class="violation-box">Detector not initialized. Click "Initialize" to start.</div>', unsafe_allow_html=True)
        
        # System status
        st.markdown("### ⚙️ System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("MediaPipe", "✅" if MP_AVAILABLE else "❌")
        with status_col2:
            st.metric("YOLO", "✅" if YOLO_AVAILABLE else "❌")
        
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
            """)

        # Auto-refresh for real-time updates (reduced frequency to prevent errors)
        if webrtc_ctx.state.playing and st.session_state.connection_state == "connected":
            time.sleep(2)  # Increased interval to reduce load
            st.rerun()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Make sure you have installed: pip install streamlit-webrtc mediapipe opencv-python")
        st.code("""
        # Install required packages:
        pip install streamlit-webrtc mediapipe opencv-python numpy
        
        # Optional for better detection:
        pip install ultralytics
        """)
