"""
MIVR-CEIQ Foot Detector v3.3
Uses AdaptiveKalmanPoint from adaptive_kalman.py

改進：
  - 靜止：點完全不動（Q 壓到極低）
  - 移動：Q 自動升高，幾乎零延遲
  - 遮擋：速度衰減預測，不飄移
  - 異常值：閘門過濾誤偵測跳點

Controls:
  B/b  brightness +/-5
  C/c  contrast   +/-0.1
  K    toggle adaptive Kalman
  D    toggle debug overlay (Q value, state)
  T    toggle trail
  R    reset
  +/-  confidence
  S    screenshot
  Q    quit
"""

import cv2, numpy as np, argparse, time, os, csv
from pathlib import Path
from collections import deque, defaultdict

# Import adaptive Kalman (must be in same directory)
try:
    from adaptive_kalman import MultiFootTracker, AKFConfig
    AKF_AVAILABLE = True
except ImportError:
    AKF_AVAILABLE = False
    print("[WARN] adaptive_kalman.py not found, falling back to basic Kalman")

DATA_DIR      = "data/foot_detection"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections.csv"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640
ANKLE_L, ANKLE_R = 15, 16
KNEE_L,  KNEE_R  = 13, 14
TRAIL_LEN = 25


# ─────────────────────────────────────────────────────
# Fallback basic Kalman (if adaptive_kalman.py missing)
# ─────────────────────────────────────────────────────

class BasicKalmanPoint:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kf.errorCovPost      = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.missing_frames = 0
        self.is_still = False
        self.current_q = 0.01

    def update(self, x, y):
        m = np.array([[x],[y]], np.float32)
        if not self.initialized:
            self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        self.kf.predict()
        s = self.kf.correct(m)
        self.missing_frames = 0
        return float(s[0][0]), float(s[1][0])

    def predict_only(self):
        self.missing_frames += 1
        p = self.kf.predict()
        return float(p[0][0]), float(p[1][0])

    def reset(self): self.__init__()

    @property
    def alive(self): return self.missing_frames < 12

    @property
    def state_info(self): return "basic"


class BasicMultiTracker:
    def __init__(self):
        self._t = defaultdict(BasicKalmanPoint)

    def update(self, pid, side, rx, ry):
        return self._t[(pid,side)].update(rx, ry)

    def predict_missing(self, seen):
        dead = [k for k,t in self._t.items()
                if k not in seen and not t.alive]
        for k in dead: del self._t[k]
        for k,t in self._t.items():
            if k not in seen: t.predict_only()

    def get_info(self, pid, side): return self._t.get((pid,side), BasicKalmanPoint()).state_info
    def reset_all(self):           self._t.clear()


# ─────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────

def open_camera(index):
    for backend, name in [(cv2.CAP_DSHOW,"DSHOW"),
                          (cv2.CAP_MSMF, "MSMF"),
                          (cv2.CAP_ANY,  "ANY")]:
        print(f"  cam {index} [{name}] ...", end=" ", flush=True)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened(): print("fail"); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(20):
            ret, f = cap.read()
            if ret and f is not None and f.size > 0 and f.mean() > 2.0:
                print(f"OK  {int(cap.get(3))}x{int(cap.get(4))}")
                return cap
            time.sleep(0.1)
        print("no frame"); cap.release()
    return None

def find_camera():
    for i in range(5):
        c = open_camera(i)
        if c: print(f"[Camera] Using index {i}"); return c
    return None


# ─────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────

def draw_trail(disp, trail, color):
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        if pts[i-1] and pts[i]:
            fade = i / n
            c = tuple(int(v * fade) for v in color)
            cv2.line(disp, pts[i-1], pts[i], c, 2)


def draw_hud(disp, feet, fps, conf, alpha, beta,
             kalman_on, debug_on, trail_on, tag, dw, dh):
    hc = (0, 220, 110) if feet > 0 else (60, 60, 60)
    cv2.rectangle(disp, (0,0), (dw, 52), (0,0,0), -1)
    cv2.putText(disp,
                f"Feet:{feet}  FPS:{fps:.1f}  Conf:{conf:.2f}  "
                f"Bright:+{beta}  Contrast:x{alpha:.1f}  "
                f"AKF:{'ON' if kalman_on else 'OFF'}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, hc, 2)
    cv2.putText(disp,
                "B/b=bright  C/c=contrast  K=AKF  D=debug  T=trail  "
                "R=reset  +/-=conf  S=shot  Q=quit"
                f"  [{tag}]",
                (10, dh-9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70,70,120), 1)


# ─────────────────────────────────────────────────────
# Main detect
# ─────────────────────────────────────────────────────

def detect(camera_index, save_csv_flag):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics"); return

    if os.path.exists(TRAINED_MODEL):
        model = YOLO(TRAINED_MODEL); use_pose = False; tag = "custom"
    else:
        model = YOLO("yolov8n-pose.pt"); use_pose = True
        tag = "YOLOv8n-Pose"
        print(f"[Model] {tag}")

    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None:
        print("[ERROR] No camera. Try --camera -1 or check USB mode (PC Remote).")
        return

    # 建立追蹤器
    if AKF_AVAILABLE:
        cfg = AKFConfig(
            q_base           = 0.5,
            q_max            = 800.0,
            r_noise          = 8.0,
            innovation_window= 5,
            innovation_gain  = 6.0,
            velocity_decay   = 0.75,
            still_threshold  = 2.5,
            still_damping    = 0.05,
            outlier_gate     = 120.0,
            max_missing      = 12,
        )
        tracker = MultiFootTracker(cfg)
        tracker_name = "AdaptiveKalman"
    else:
        tracker = BasicMultiTracker()
        tracker_name = "BasicKalman"
    print(f"[Tracker] {tracker_name}")

    csv_f = csv_w = None
    if save_csv_flag:
        csv_f = open(DETECT_CSV, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","pid","side",
                         "raw_x","raw_y","smooth_x","smooth_y",
                         "conf","state"])

    print(f"\n=== MIVR-CEIQ Foot Detector v3.3 [{tag}] ===")
    print("  K=AKF on/off  D=debug overlay  T=trail  B/b=bright  C/c=contrast\n")

    conf_th   = CONF_THRESHOLD
    alpha     = 1.4
    beta      = 40
    kalman_on = True
    debug_on  = False
    trail_on  = True

    ALPHA_DEF, BETA_DEF = 1.4, 40

    trails: dict = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    fps_buf   = []
    frame_id  = 0
    black_n   = 0

    while True:
        t0 = time.time()
        ret, raw = cap.read()
        if not ret or raw is None:
            time.sleep(0.05); continue

        if raw.mean() < 2.0:
            black_n += 1
            if black_n % 15 == 0:
                print(f"[WARN] {black_n} black frames")
            frame_id += 1; continue
        black_n = 0

        frame = cv2.convertScaleAbs(raw, alpha=alpha, beta=beta)

        res = model(frame, conf=conf_th, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)

        h, w   = frame.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(frame, (dw, dh))
        sx_s   = dw / w
        sy_s   = dh / h
        feet   = 0
        seen   = set()

        if res and len(res) > 0:
            r = res[0]

            # ── Pose mode ────────────────────────────
            if use_pose and r.keypoints is not None:
                kpts = r.keypoints.data

                for pid, pk in enumerate(kpts):
                    for side, aid, kid, col in [
                        ("L", ANKLE_L, KNEE_L, (0, 220, 110)),
                        ("R", ANKLE_R, KNEE_R, (30, 140, 255)),
                    ]:
                        ac = float(pk[aid][2])
                        if ac < conf_th: continue

                        raw_x = float(pk[aid][0])
                        raw_y = float(pk[aid][1])
                        kx    = float(pk[kid][0])
                        ky    = float(pk[kid][1])

                        key = (pid, side)
                        seen.add(key)

                        # Smooth
                        if kalman_on:
                            sx_k, sy_k = tracker.update(pid, side, raw_x, raw_y)
                            state_info = tracker.get_info(pid, side)
                        else:
                            sx_k, sy_k = raw_x, raw_y
                            state_info = "raw"

                        # Display coords
                        dax = int(sx_k * sx_s); day = int(sy_k * sy_s)
                        dkx = int(kx  * sx_s); dky = int(ky  * sy_s)

                        trails[key].append((dax, day))
                        feet += 1

                        # Trail
                        if trail_on:
                            draw_trail(disp, trails[key], col)

                        # Raw dot (grey, small) when debug on
                        if debug_on and kalman_on:
                            rdx = int(raw_x * sx_s)
                            rdy = int(raw_y * sy_s)
                            cv2.circle(disp, (rdx, rdy), 4, (80, 80, 80), -1)
                            cv2.line(disp, (rdx,rdy), (dax,day), (60,60,60), 1)

                        # Ankle
                        cv2.circle(disp, (dax, day), 11, col, -1)
                        cv2.circle(disp, (dax, day), 16, col,  2)

                        # Knee + leg
                        if float(pk[kid][2]) > 0.3:
                            cv2.circle(disp, (dkx, dky), 6, col, 2)
                            cv2.line(disp, (dkx, dky), (dax, day), col, 2)

                        # Label
                        label = f"P{pid}{side}"
                        if debug_on:
                            label += f" {state_info}"
                        cv2.putText(disp, label,
                                    (dax+14, day-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

                        if csv_w:
                            csv_w.writerow([
                                f"{time.time():.3f}", pid, side,
                                f"{raw_x:.1f}", f"{raw_y:.1f}",
                                f"{sx_k:.1f}", f"{sy_k:.1f}",
                                f"{ac:.3f}", state_info
                            ])

                # Person boxes
                if r.boxes is not None:
                    for b in r.boxes:
                        x1,y1,x2,y2 = map(float, b.xyxy[0])
                        cv2.rectangle(disp,
                            (int(x1*sx_s),int(y1*sy_s)),
                            (int(x2*sx_s),int(y2*sy_s)),
                            (45,45,45), 1)

            # ── Custom model mode ─────────────────────
            elif not use_pose and r.boxes is not None:
                for i, b in enumerate(r.boxes):
                    cf = float(b.conf[0])
                    x1,y1,x2,y2 = map(float, b.xyxy[0])
                    cx,cy = (x1+x2)/2,(y1+y2)/2

                    key = (i,"F")
                    seen.add(key)

                    if kalman_on:
                        scx,scy = tracker.update(i,"F",cx,cy)
                        si = tracker.get_info(i,"F")
                    else:
                        scx,scy = cx,cy; si = "raw"

                    ddx,ddy = int(scx*sx_s), int(scy*sy_s)
                    trails[key].append((ddx,ddy))
                    feet += 1

                    if trail_on: draw_trail(disp, trails[key], (0,220,110))

                    dx1,dy1=int(x1*sx_s),int(y1*sy_s)
                    dx2,dy2=int(x2*sx_s),int(y2*sy_s)
                    cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,220,110),2)
                    cv2.circle(disp,(ddx,ddy),7,(0,220,110),-1)
                    lbl = f"foot{i}" + (f" {si}" if debug_on else "")
                    cv2.putText(disp,lbl,(dx1,dy1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,220,110),1)

        # Predict missing
        if kalman_on:
            tracker.predict_missing(seen)

        # Clean up trails for dead trackers
        dead_trails = [k for k in trails if k not in seen
                       and len(trails[k]) > 0
                       and frame_id % 30 == 0]
        for k in dead_trails:
            trails[k].clear()

        # FPS & HUD
        fps_buf.append(time.time()-t0)
        if len(fps_buf)>30: fps_buf.pop(0)
        fps = 1.0/(sum(fps_buf)/len(fps_buf))

        draw_hud(disp, feet, fps, conf_th, alpha, beta,
                 kalman_on, debug_on, trail_on, tag, dw, dh)

        cv2.imshow("MIVR-CEIQ Foot Detection v3.3", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break

        elif key in (ord('s'),ord('S')):
            ts = int(time.time())
            cv2.imwrite(f"{DATA_DIR}/shot_raw_{ts}.jpg",    raw)
            cv2.imwrite(f"{DATA_DIR}/shot_bright_{ts}.jpg", frame)
            print(f"  [shot] {ts}")

        elif key in (ord('+'),ord('=')):
            conf_th=min(conf_th+0.05,0.95); print(f"  conf->{conf_th:.2f}")
        elif key==ord('-'):
            conf_th=max(conf_th-0.05,0.05); print(f"  conf->{conf_th:.2f}")

        elif key==ord('B'): beta =min(beta+5, 150);            print(f"  bright->{beta}")
        elif key==ord('b'): beta =max(beta-5, -50);            print(f"  bright->{beta}")
        elif key==ord('C'): alpha=min(round(alpha+0.1,1),3.0); print(f"  contrast->{alpha}")
        elif key==ord('c'): alpha=max(round(alpha-0.1,1),0.5); print(f"  contrast->{alpha}")

        elif key in (ord('k'),ord('K')):
            kalman_on = not kalman_on
            if not kalman_on:
                tracker.reset_all()
                for t in trails.values(): t.clear()
            print(f"  AKF: {'ON' if kalman_on else 'OFF'}")

        elif key in (ord('d'),ord('D')):
            debug_on = not debug_on
            print(f"  Debug: {'ON' if debug_on else 'OFF'}")

        elif key in (ord('t'),ord('T')):
            trail_on = not trail_on
            print(f"  Trail: {'ON' if trail_on else 'OFF'}")

        elif key in (ord('r'),ord('R')):
            alpha, beta = ALPHA_DEF, BETA_DEF
            tracker.reset_all()
            for t in trails.values(): t.clear()
            print(f"  Reset all")

        frame_id += 1

    cap.release(); cv2.destroyAllWindows()
    if csv_f: csv_f.close(); print(f"[CSV] {DETECT_CSV}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v3.3")
    p.add_argument("--camera", type=int, default=0, help="-1=auto scan")
    p.add_argument("--no-csv", action="store_true")
    a = p.parse_args()
    print("="*58)
    print("  MIVR-CEIQ Foot Detector v3.3  [Adaptive Kalman]")
    print("="*58)
    detect(a.camera, not a.no_csv)

if __name__ == "__main__":
    main()
