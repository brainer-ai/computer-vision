import cv2
import time
import numpy as np
import os
from datetime import datetime
import streamlit as st
from PIL import Image
import io
import base64

# Optional: ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# MediaPipe for face mesh
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Streamlit page config
st.set_page_config(
    page_title="Exam Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
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
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .start-btn {
        background-color: #28a745;
        color: white;
    }
    .stop-btn {
        background-color: #dc3545;
        color: white;
    }
    .save-btn {
        background-color: #007bff;
        color: white;
    }
    .reset-btn {
        background-color: #ffc107;
        color: #212529;
    }
    .stImage {
        border: 2px solid #c5d9f1;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
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
    .sidebar .stCheckbox > label, .sidebar .stRadio > label {
        font-size: 16px;
        color: #1a3c5a;
        font-weight: 500;
    }
    .sidebar .stMarkdown h3 {
        color: #1a3c5a;
        font-size: 1.3rem;
        margin-bottom: 12px;
    }
    [data-testid="column"] > div > .stMarkdown > div:first-child {
        font-weight: 600;
        color: #1a3c5a;
        font-size: 1.25rem;
        margin-bottom: 16px;
    }
    hr {
        border-color: #ddd;
        margin: 15px 0;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .violation-alert {
        background-color: #ff4757;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .camera-instructions {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


class ExamDetector:
    def __init__(self, use_yolo=YOLO_AVAILABLE):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required. Install mediapipe package.")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)

        self.prev_gray = None
        self.movement_window = []
        self.movement_threshold = 80000

        self.violations = []
        self.total_violations = 0
        self.session_start_time = datetime.now()

        self.use_yolo = False
        self.yolo_model = None
        if use_yolo:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.use_yolo = True
                st.success('✅ YOLO model loaded: yolov8n.pt')
            except Exception as e:
                st.warning(f'⚠️ YOLO load failed, using basic detection: {e}')
                self.use_yolo = False

        self.frame_counter = 0
        
        # Detection sensitivity settings
        self.detection_history = {
            'absent': [],
            'looking_away': [],
            'phone': [],
            'paper': [],
            'movement': []
        }
        
        self.violation_thresholds = {
            'phone': 1,  # Immediate detection
            'paper': 1,  # Immediate detection
            'absent': 3,  # Need 3 consecutive detections
            'looking_away': 5,  # Need 5 consecutive detections
            'movement': 3   # Need 3 consecutive detections
        }

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
                        'cell phone', 'mobile phone', 'phone', 'smartphone',
                        'iphone', 'android', 'tablet', 'remote'
                    ]
                    if any(phone_class in name for phone_class in phone_classes) and conf >= CONF_THRESHOLD:
                        phone_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Phone: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    paper_classes = [
                        'book', 'paper', 'notebook', 'magazine', 
                        'newspaper', 'document', 'letter', 'card',
                        'envelope', 'file'
                    ]
                    if any(paper_class in name for paper_class in paper_classes) and conf >= CONF_THRESHOLD:
                        paper_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Paper: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        except Exception as e:
            st.error(f"YOLO detection error: {e}")
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
        
        # Edge-based detection for rectangular devices
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180.0 / np.pi
                
                if abs(angle) < 15 or abs(angle) > 165:
                    horizontal_lines.append(line)
                elif 75 < abs(angle) < 105:
                    vertical_lines.append(line)
            
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h else 0
                if 0.7 < aspect_ratio < 1.5:
                    perimeter = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    if len(approx) >= 4:
                        return True
        
        # Cream/off-white paper detection
        lower_cream = np.array([15, 30, 200])
        upper_cream = np.array([35, 80, 255])
        
        mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
        contours_cream, _ = cv2.findContours(mask_cream, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_cream:
            if cv2.contourArea(cnt) > 12000:
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

        # Process violations with history-based detection
        self._process_violations({
            'absent': face_bbox is None,
            'looking_away': face_bbox is not None and not looking_forward,
            'phone': phone,
            'paper': paper,
            'movement': excessive_movement
        })

        # Draw status overlay
        self._draw_status(annotated, {
            'face': face_bbox is not None,
            'looking_forward': looking_forward,
            'phone': phone,
            'paper': paper,
            'movement': excessive_movement
        })

        return annotated

    def _process_violations(self, detections):
        """Process violations with history-based filtering"""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        for violation_type, detected in detections.items():
            history = self.detection_history[violation_type]
            threshold = self.violation_thresholds[violation_type]
            
            # Add current detection to history
            history.append(detected)
            
            # Keep only recent history (last 10 frames)
            if len(history) > 10:
                history.pop(0)
            
            # Check if violation threshold is met
            if len(history) >= threshold and all(history[-threshold:]):
                # Record violation if not already recorded recently
                if not self.violations or self.violations[-1]['type'] != violation_type:
                    self.total_violations += 1
                    self.violations.append({
                        'type': violation_type,
                        'time': current_time,
                        'frame': self.frame_counter
                    })

    def _draw_status(self, frame, status):
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
            ('Face Detected', status['face']),
            ('Looking Forward', status['looking_forward']),
            ('Phone Clear', not status['phone']),
            ('Paper Clear', not status['paper']),
            ('Movement Normal', not status['movement'])
        ]
        
        for label, is_ok in status_items:
            color = (0, 255, 0) if is_ok else (0, 0, 255)
            status_text = 'OK' if is_ok else 'VIOLATION'
            cv2.putText(frame, f'{label}: {status_text}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

    def get_violation_summary(self):
        """Get summary of violations"""
        violation_counts = {}
        for v in self.violations:
            vtype = v['type']
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        return {
            'total': self.total_violations,
            'by_type': violation_counts,
            'session_duration': str(datetime.now() - self.session_start_time).split('.')[0],
            'latest_violations': self.violations[-10:]  # Last 10 violations
        }


def main():
    st.markdown('<div class="title">🎓 Exam Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered exam proctoring with face tracking and object detection</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False

    # Sidebar configuration
    st.sidebar.header("🔧 Detection Settings", divider="gray")
    
    # Check if required packages are available
    if not MP_AVAILABLE:
        st.sidebar.error("❌ MediaPipe not available. Please install: pip install mediapipe")
        return
    
    use_yolo = st.sidebar.checkbox('🎯 Use YOLO Detection', 
                                  value=YOLO_AVAILABLE, 
                                  disabled=not YOLO_AVAILABLE,
                                  help="Enable advanced object detection")
    
    if not YOLO_AVAILABLE:
        st.sidebar.info("💡 Install ultralytics for YOLO: pip install ultralytics")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🔍 Detection Features:**
    - 👤 Real-time face tracking
    - 👁️ Gaze direction monitoring  
    - 📱 Phone detection (instant alert)
    - 📄 Paper detection (instant alert)
    - 🚶 Movement analysis
    - 📊 Violation logging & reporting
    """)

    # Main interface
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 📸 Camera Input")
        
        # Camera input method selection
        input_method = st.radio("Choose input method:", 
                               ["📷 Live Camera", "🖼️ Upload Image", "🎥 Upload Video"], 
                               horizontal=True)
        
        if input_method == "📷 Live Camera":
            st.markdown('<div class="camera-instructions">📋 <strong>Instructions:</strong><br>1. Click "Take Photo" to capture from your camera<br>2. The system will analyze the image for violations<br>3. Take multiple photos to build a monitoring session</div>', unsafe_allow_html=True)
            
            # Camera input
            camera_image = st.camera_input("📷 Take a photo for monitoring", key="camera")
            
            if camera_image is not None:
                # Convert to OpenCV format
                image = Image.open(camera_image)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Initialize detector if not exists
                if st.session_state.detector is None:
                    with st.spinner("🔄 Initializing detector..."):
                        st.session_state.detector = ExamDetector(use_yolo=use_yolo)
                
                # Process frame
                with st.spinner("🔍 Analyzing image..."):
                    processed_frame = st.session_state.detector.process_frame(frame)
                
                # Display processed image
                st.image(processed_frame, channels='BGR', caption="📊 Analysis Result", use_container_width=True)

        elif input_method == "🖼️ Upload Image":
            uploaded_file = st.file_uploader("Choose an image file", 
                                           type=['png', 'jpg', 'jpeg'],
                                           help="Upload an image to analyze for violations")
            
            if uploaded_file is not None:
                # Convert to OpenCV format
                image = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Initialize detector if not exists
                if st.session_state.detector is None:
                    with st.spinner("🔄 Initializing detector..."):
                        st.session_state.detector = ExamDetector(use_yolo=use_yolo)
                
                # Process frame
                with st.spinner("🔍 Analyzing image..."):
                    processed_frame = st.session_state.detector.process_frame(frame)
                
                # Display results
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="📁 Original Image", use_container_width=True)
                with col_img2:
                    st.image(processed_frame, channels='BGR', caption="📊 Analysis Result", use_container_width=True)

        else:  # Upload Video
            st.info("🎥 Video processing: Upload a video file to analyze frame by frame")
            uploaded_video = st.file_uploader("Choose a video file", 
                                            type=['mp4', 'avi', 'mov'],
                                            help="Upload a video to analyze for violations")
            
            if uploaded_video is not None:
                # Save uploaded video temporarily
                temp_video_path = "temp_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                # Initialize detector if not exists
                if st.session_state.detector is None:
                    with st.spinner("🔄 Initializing detector..."):
                        st.session_state.detector = ExamDetector(use_yolo=use_yolo)
                
                # Process video
                process_video_btn = st.button("🎬 Process Video", type="primary")
                
                if process_video_btn:
                    process_uploaded_video(temp_video_path, st.session_state.detector)
                
                # Cleanup
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

    with col2:
        st.markdown("### 🎛️ Control Panel")
        
        # Control buttons
        if st.button('🔄 Reset Session', help="Clear all violations and start fresh", type="secondary"):
            st.session_state.detector = None
            st.success('✅ Session reset successfully!')
        
        if st.button('💾 Download Report', help="Download violation report", type="primary"):
            if st.session_state.detector and st.session_state.detector.violations:
                report_data = generate_report(st.session_state.detector)
                st.download_button(
                    label="📄 Download Report",
                    data=report_data,
                    file_name=f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ No violations to report")

        st.markdown("---")
        
        # Violation display
        st.markdown("### 🚨 Violations Log")
        
        if st.session_state.detector and st.session_state.detector.violations:
            violation_summary = st.session_state.detector.get_violation_summary()
            
            # Summary metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total Violations", violation_summary['total'])
            with col_metric2:
                st.metric("Session Time", violation_summary['session_duration'])
            
            # Violations by type
            if violation_summary['by_type']:
                st.markdown("**📊 Violations by Type:**")
                for vtype, count in violation_summary['by_type'].items():
                    st.write(f"• {vtype.replace('_', ' ').title()}: {count}")
            
            # Recent violations
            st.markdown("**🕒 Recent Violations:**")
            violation_text = '<div class="violation-box">'
            
            for i, v in enumerate(reversed(violation_summary['latest_violations']), 1):
                violation_text += f'<p><strong>#{len(violation_summary["latest_violations"])-i+1}</strong> {v["type"].replace("_", " ").title()} at {v["time"]}</p>'
            
            violation_text += '</div>'
            st.markdown(violation_text, unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="violation-box">No violations detected yet. 🟢</div>', unsafe_allow_html=True)

    # Instructions and info
    with st.expander("📋 How to Use", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started:
        
        1. **📷 Choose Input Method**: Select live camera, image upload, or video upload
        2. **⚙️ Configure Settings**: Enable YOLO detection if available for better accuracy
        3. **📸 Capture/Upload**: Take photos or upload files for analysis
        4. **👁️ Monitor Results**: View real-time analysis and violation detection
        5. **📊 Review Report**: Check violation summary and download detailed reports
        
        ### 🎯 What We Detect:
        
        - **👤 Face Presence**: Ensures person is visible in frame
        - **👁️ Gaze Direction**: Monitors if person is looking at screen
        - **📱 Phone Usage**: Detects mobile phones or similar devices
        - **📄 Paper Materials**: Identifies unauthorized papers or books
        - **🚶 Excessive Movement**: Tracks unusual movement patterns
        
        ### 🔧 Technical Requirements:
        
        - **📦 Required**: MediaPipe (face detection)
        - **🎯 Optional**: Ultralytics YOLO (enhanced object detection)
        - **🌐 Browser**: Modern browser with camera access
        - **📱 Mobile**: Works on mobile devices with cameras
        """)

    # Performance metrics
    if st.session_state.detector:
        with st.expander("📈 Session Statistics", expanded=False):
            stats = st.session_state.detector.get_violation_summary()
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Frames Processed", st.session_state.detector.frame_counter)
            
            with col_stat2:
                violation_rate = (stats['total'] / max(st.session_state.detector.frame_counter, 1)) * 100
                st.metric("Violation Rate", f"{violation_rate:.1f}%")
            
            with col_stat3:
                avg_violations = stats['total'] / max(1, len(stats['session_duration'].split(':')[0]))
                st.metric("Violations/Hour", f"{avg_violations:.1f}")


def process_uploaded_video(video_path, detector):
    """Process uploaded video file frame by frame"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    
    st.info(f"🎥 Processing video: {total_frames} frames, {duration:.1f}s duration")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    frame_count = 0
    process_every_n = max(1, fps // 2)  # Process 2 frames per second
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame to improve performance
            if frame_count % process_every_n == 0:
                processed_frame = detector.process_frame(frame)
                
                # Update display
                frame_placeholder.image(processed_frame, channels='BGR', 
                                      caption=f"Frame {frame_count}/{total_frames}", 
                                      use_container_width=True)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} - "
                               f"Violations detected: {detector.total_violations}")
            
            # Add small delay to prevent overwhelming the interface
            if frame_count % 30 == 0:
                time.sleep(0.1)
    
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
    
    finally:
        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"✅ Video processing complete! Total violations: {detector.total_violations}")


def generate_report(detector):
    """Generate a detailed text report"""
    stats = detector.get_violation_summary()
    
    report_lines = [
        "=" * 50,
        "📊 EXAM MONITORING REPORT",
        "=" * 50,
        f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"⏱️ Session Duration: {stats['session_duration']}",
        f"📈 Frames Processed: {detector.frame_counter}",
        f"🚨 Total Violations: {stats['total']}",
        "",
        "📋 VIOLATION BREAKDOWN:",
        "-" * 30,
    ]
    
    if stats['by_type']:
        for vtype, count in stats['by_type'].items():
            percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
            report_lines.append(f"• {vtype.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        report_lines.append("• No violations detected")
    
    report_lines.extend([
        "",
        "🕒 DETAILED VIOLATION LOG:",
        "-" * 30,
    ])
    
    if detector.violations:
        for i, violation in enumerate(detector.violations, 1):
            report_lines.append(
                f"{i:3d}. [{violation['time']}] "
                f"{violation['type'].replace('_', ' ').title()} "
                f"(Frame: {violation.get('frame', 'N/A')})"
            )
    else:
        report_lines.append("• No violations recorded")
    
    report_lines.extend([
        "",
        "🔍 DETECTION SUMMARY:",
        "-" * 30,
        f"• Detection Method: {'YOLO + Heuristic' if detector.use_yolo else 'Heuristic Only'}",
        f"• Face Detection: MediaPipe Face Mesh",
        f"• Movement Threshold: {detector.movement_threshold}",
        "",
        "📝 RECOMMENDATIONS:",
        "-" * 30,
    ])
    
    if stats['total'] == 0:
        report_lines.append("✅ Excellent performance! No violations detected.")
    elif stats['total'] < 5:
        report_lines.append("✨ Good performance with minimal violations.")
    elif stats['total'] < 15:
        report_lines.append("⚠️ Moderate violations detected. Review behavior.")
    else:
        report_lines.append("🚨 High violation count. Requires attention.")
    
    report_lines.extend([
        "",
        "=" * 50,
        "📄 End of Report",
        "=" * 50
    ])
    
    return "\n".join(report_lines)


# Additional utility functions
def create_demo_frame():
    """Create a demo frame for testing"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Add some demo text
    cv2.putText(frame, "DEMO MODE", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Upload an image or use camera", (120, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return frame


def validate_image_format(uploaded_file):
    """Validate uploaded image format"""
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        st.error(f"❌ Invalid image format: {e}")
        return None


# Error handling wrapper
def safe_process_frame(detector, frame):
    """Safely process frame with error handling"""
    try:
        return detector.process_frame(frame)
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        # Return original frame with error overlay
        error_frame = frame.copy()
        cv2.putText(error_frame, "PROCESSING ERROR", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_frame


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.info("💡 Try refreshing the page or check your internet connection.")
        
        # Display error details in expander for debugging
        with st.expander("🔧 Technical Details", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language='python')
