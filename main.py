import cv2
import time
import numpy as np
from datetime import datetime
import pygame
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

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
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .stats-label {
        font-size: 1rem;
        text-align: center;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


class ExamDetector:
    def __init__(self, use_yolo=YOLO_AVAILABLE, enable_sound=True):
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
                print('YOLO model loaded: yolov8n.pt')
            except Exception as e:
                print('YOLO load failed, falling back to heuristics:', e)
                self.use_yolo = False

        self.detection_interval = 3
        self.frame_counter = 0
        self.face_lost_counter = 0
        self.face_lost_threshold = 15

        self.enable_sound = enable_sound
        if self.enable_sound:
            try:
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
                pygame.mixer.init()
                self._generate_alert_sound()
            except Exception as e:
                print(f"Sound initialization failed: {e}")
                self.enable_sound = False

        # إضافة متغيرات للتتبع
        self._counters = {k: 0 for k in ['absent', 'looking_away', 'phone', 'paper', 'movement']}
        self._thresholds = {
            'absent': 50,
            'looking_away': 100,
            'phone': 5,
            'paper': 3,
            'movement': 110
        }

    def detect_face_mesh(self, frame):
        """اكتشاف الوجه باستخدام MediaPipe Face Mesh"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        h, w = frame.shape[:2]
        
        if not results.multi_face_landmarks:
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        # حساب حدود الوجه
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x2, y2 = min(max(xs) + 10, w - 1), min(max(ys) + 10, h - 1)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # اكتشاف العينين
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

        # تحديد اتجاه النظر
        looking_forward = True
        if left_center and right_center:
            eye_x = (left_center[0] + right_center[0]) / 2
            face_center_x = (x1 + x2) / 2
            deviation = abs(eye_x - face_center_x)
            looking_forward = deviation < (bbox[2] * 0.25)

        return bbox, (left_center, right_center), looking_forward

    def detect_objects_yolo(self, frame):
        """اكتشاف الكائنات باستخدام YOLO"""
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

                    phone_classes = ['cell phone', 'mobile phone', 'phone', 'smartphone',
                                   'iphone', 'android', 'tablet', 'remote']
                    if any(phone_class in name for phone_class in phone_classes) and conf >= CONF_THRESHOLD:
                        phone_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Phone: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    paper_classes = ['book', 'paper', 'notebook', 'magazine', 
                                   'newspaper', 'document', 'letter', 'card',
                                   'envelope', 'file']
                    if any(paper_class in name for paper_class in paper_classes) and conf >= CONF_THRESHOLD:
                        paper_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Paper: {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.detect_objects_heuristic(frame)

        return phone_detected, paper_detected

    def detect_objects_heuristic(self, frame):
        """اكتشاف الكائنات باستخدام الطرق التقليدية"""
        phone = self.detect_phone_advanced(frame)
        paper = self.detect_paper_advanced(frame)
        return phone, paper

    def detect_phone_advanced(self, frame):
        """اكتشاف الهاتف المتقدم"""
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
        """اكتشاف الورق المتقدم"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
        return False

    def detect_movement(self, frame):
        """اكتشاف الحركة المفرطة"""
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

    def _generate_alert_sound(self):
        """توليد صوت التنبيه"""
        try:
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            frames = int(duration * sample_rate)
            
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            arr = (arr * 32767).astype(np.int16)
            arr = np.ascontiguousarray(arr)
            
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            stereo_arr = np.ascontiguousarray(stereo_arr)
            
            self.alert_sound = pygame.sndarray.make_sound(stereo_arr)
        except Exception as e:
            print(f"Alert sound generation failed: {e}")
            self.enable_sound = False

    def _play_alert_sound(self, violation_type):
        """تشغيل صوت التنبيه"""
        if self.enable_sound and hasattr(self, 'alert_sound'):
            try:
                if violation_type == 'absent':
                    self.alert_sound.play()
                elif violation_type == 'looking_away':
                    self.alert_sound.play()
                    time.sleep(0.1)
                    self.alert_sound.play()
                elif violation_type == 'movement':
                    for i in range(3):
                        self.alert_sound.play()
                        if i < 2:
                            time.sleep(0.05)
                else:
                    self.alert_sound.play()
            except Exception as e:
                print(f"Sound play error: {e}")
                self.enable_sound = False

    def _accumulate_violation(self, vtype, flag):
        """تجميع المخالفات"""
        if flag:
            # الهاتف والورق يتم تسجيلهما فورًا
            if vtype in ['phone', 'paper']:
                self.total_violations += 1
                t = datetime.now().strftime('%H:%M:%S')
                self.violations.append({'type': vtype, 'time': t})
                print(f'Violation: {vtype} at {t}')
                self._play_alert_sound(vtype)
                return

            # باقي المخالفات تستخدم التراكم
            self._counters[vtype] += 1
        else:
            self._counters[vtype] = max(0, self._counters[vtype] - 2)

        # التحقق من تجاوز العتبة
        if self._counters[vtype] > self._thresholds[vtype]:
            self._counters[vtype] = 0
            self.total_violations += 1
            t = datetime.now().strftime('%H:%M:%S')
            self.violations.append({'type': vtype, 'time': t})
            print(f'Violation: {vtype} at {t}')
            self._play_alert_sound(vtype)

    def _draw_status(self, frame, face, eyes, gaze, phone, paper, movement):
        """رسم حالة المراقبة على الإطار"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y = 35
        cv2.putText(frame, f'Total Violations: {self.total_violations}', 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y += 30
        
        items = [
            ('Face', face), 
            ('Movement Normal', not movement),
            ('Phone Clear', not phone),
            ('Paper Clear', not paper)
        ]
        
        for label, ok in items:
            color = (0,255,0) if ok else (0,0,255)
            txt = 'OK' if ok else 'VIOLATION'
            cv2.putText(frame, f'{label}: {txt}', 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24

    def process(self, frame):
        """معالجة الإطار الرئيسية"""
        self.frame_counter += 1
        annotated = frame.copy()

        # اكتشاف الوجه
        face_bbox = None
        left_right = (None, None)
        looking_forward = True

        res = self.detect_face_mesh(frame)
        if res[0] is not None:
            face_bbox, left_right, looking_forward = res
            x, y, w, h = face_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # اكتشاف الكائنات
        if self.use_yolo:
            phone, paper = self.detect_objects_yolo(annotated)
        else:
            phone, paper = self.detect_objects_heuristic(annotated)

        # اكتشاف الحركة المفرطة
        excessive_movement = self.detect_movement(frame)

        # تسجيل المخالفات
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

        # رسم المعلومات على الإطار
        self._draw_status(annotated, face_bbox is not None, left_right[0] is not None,
                        looking_forward, phone, paper, excessive_movement)

        return annotated


class VideoTransformer(VideoTransformerBase):
    """فئة معالجة الفيديو باستخدام VideoTransformerBase"""
    
    def __init__(self):
        self.detector = None
    
    def set_detector(self, detector):
        """تعيين كاشف المخالفات"""
        self.detector = detector
    
    def transform(self, frame):
        """معالجة إطارات الفيديو"""
        img = frame.to_ndarray(format="bgr24")
        
        # إذا تم تعيين الكاشف، قم بمعالجة الإطار
        if self.detector is not None:
            try:
                img = self.detector.process(img)
            except Exception as e:
                print(f"Error in frame processing: {e}")
                # في حالة حدوث خطأ، ارجع الإطار الأصلي
                pass
        
        return img


# إعداد WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# متغيرات عامة للتطبيق
if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'video_transformer' not in st.session_state:
    st.session_state.video_transformer = VideoTransformer()


def main():
    st.markdown('<div class="title">🎓 Exam Monitor with WebRTC</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered exam proctoring with real-time camera streaming</div>', unsafe_allow_html=True)

    # الشريط الجانبي للإعدادات
    st.sidebar.header("⚙️ Settings", divider="gray")
    use_yolo_checkbox = st.sidebar.checkbox('✅ Use YOLO Detection', value=YOLO_AVAILABLE, 
                                           help="Enable YOLO for object detection")
    enable_sound = st.sidebar.checkbox('🔊 Enable Sound Alerts', value=True, 
                                      help="Play alert sounds on violations")

    # إعداد الكاشف
    if st.sidebar.button("🚀 Initialize Detector"):
        try:
            st.session_state.detector = ExamDetector(
                use_yolo=use_yolo_checkbox and YOLO_AVAILABLE,
                enable_sound=enable_sound
            )
            st.session_state.video_transformer.set_detector(st.session_state.detector)
            st.sidebar.success("✅ Detector initialized!")
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing detector: {e}")

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Detection Info", divider="gray")
    st.sidebar.markdown("""
    **Instant Detection:**
    - 📱 Phone: Immediate alert
    - 📄 Paper: Immediate alert
    - 👁️ Face tracking with MediaPipe
    - 🔍 Gaze direction monitoring
    - 🚶 Movement analysis
    
    **Status:**
    """ + (f"✅ YOLO Available" if YOLO_AVAILABLE else "⚠️ YOLO Not Available") + """
    """ + (f"✅ MediaPipe Available" if MP_AVAILABLE else "❌ MediaPipe Not Available"))

    # التخطيط الرئيسي
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.header("📹 Live Camera Stream")
        
        # بث الكاميرا باستخدام WebRTC
        webrtc_ctx = webrtc_streamer(
            key="exam-monitor",
            video_transformer_factory=lambda: st.session_state.video_transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        if webrtc_ctx.state.playing:
            st.success("🟢 Camera is active and monitoring")
            
            # تحديث الكاشف إذا لم يتم تعيينه بعد
            if (st.session_state.detector is not None and 
                st.session_state.video_transformer.detector is None):
                st.session_state.video_transformer.set_detector(st.session_state.detector)
                
        else:
            st.info("📷 Click 'START' to begin monitoring")

    with col2:
        st.header("🎛️ Control Panel")
        
        # الإحصائيات
        if st.session_state.detector is not None:
            st.markdown(f"""
            <div class="stats-container">
                <div class="stats-number">{st.session_state.detector.total_violations}</div>
                <div class="stats-label">Total Violations</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stats-container">
                <div class="stats-number">0</div>
                <div class="stats-label">Initialize Detector First</div>
            </div>
            """, unsafe_allow_html=True)

        # أزرار التحكم
        if st.button('💾 Save Report', key='save', help="Save violation report", use_container_width=True):
            if st.session_state.detector is not None:
                save_report(st.session_state.detector)
                st.success('✅ Report saved successfully')
            else:
                st.error('❌ No detector initialized')

        if st.button('🔄 Reset Violations', key='reset', help="Clear all violations", use_container_width=True):
            if st.session_state.detector is not None:
                st.session_state.detector.violations = []
                st.session_state.detector.total_violations = 0
                st.success('🔄 Violations reset')
            else:
                st.error('❌ No detector initialized')

        st.markdown('---')
        st.header('📋 Recent Violations')
        
        # عرض المخالفات
        if st.session_state.detector is not None and st.session_state.detector.violations:
            violations_text = '<div class="violation-box">' + ''.join([
                f'<p><strong>#{len(st.session_state.detector.violations)-i}</strong> '
                f'{v["type"].title().replace("_", " ")} at {v["time"]}</p>'
                for i, v in enumerate(reversed(st.session_state.detector.violations[-10:]))  # أحدث 10 مخالفات
            ]) + '</div>'
            st.markdown(violations_text, unsafe_allow_html=True)
        else:
            st.markdown('<div class="violation-box">No violations detected yet.</div>', 
                       unsafe_allow_html=True)

        # تحديث في الوقت الفعلي
        if webrtc_ctx.state.playing and st.session_state.detector is not None:
            # تحديث تلقائي كل ثانية
            time.sleep(0.1)
            st.rerun()


def save_report(detector):
    """حفظ تقرير المراقبة"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'exam_report_{timestamp}.txt'
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('=== Exam Monitoring Report ===\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total Violations: {detector.total_violations}\n\n')
        f.write('Violation Details:\n')
        f.write('-' * 40 + '\n')
        
        if detector.violations:
            for i, v in enumerate(detector.violations, 1):
                violation_name = v['type'].title().replace('_', ' ')
                f.write(f"{i:2d}. {violation_name:<15} at {v['time']}\n")
        else:
            f.write("No violations recorded.\n")
        
        f.write('\n' + '=' * 50 + '\n')
        f.write('Report generated by Exam Monitor System\n')
    
    st.success(f"📄 Report saved as: {fname}")


if __name__ == '__main__':
    main()
