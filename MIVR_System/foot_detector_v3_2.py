"""
MIVR-CEIQ Foot Detector v3.2
Fix: keypoint jitter -> Kalman Filter per ankle point
  Each ankle (person_id, side) gets its own Kalman tracker.
  Raw YOLOv8 output is smoothed before display & CSV output.

Controls:
  B/b  brightness +/-5      (default +40)
  C/c  contrast   +/-0.1    (default x1.4)
  K    toggle Kalman on/off  (default ON)
  T    toggle trail lines    (default ON)
  R    reset all             (brightness/contrast/Kalman states)
  +/-  confidence threshold
  S    screenshot
  Q    quit
"""

import cv2
import numpy as np
import argparse, time, os, csv
from pathlib import Path
from collections import defaultdict

DATA_DIR      = "data/foot_detection"
RAW_DIR       = f"{DATA_DIR}/raw_images"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections.csv"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

ANKLE_L, ANKLE_R = 15, 16
KNEE_L,  KNEE_R  = 13, 14

# Trail history length (frames)
TRAIL_LEN = 20


# ─────────────────────────────────────────────────────
# Kalman Filter wrapper for a single 2D point
# ─────────────────────────────────────────────────────

class KalmanPoint:
    """
    Constant-velocity Kalman filter for one (x, y) point.
    State: [x, y, vx, vy]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0   # per-frame time step

        # Transition matrix  (x' = x + vx*dt, etc.)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32)

        # Measurement matrix  (observe x, y only)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)

        # Process noise  (how much we trust the motion model)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Measurement noise  (how noisy YOLOv8 keypoints are)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.missing_frames = 0
        self.MAX_MISSING = 10   # drop tracker after N frames without detection

    def update(self, x: float, y: float):
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePre = np.array(
                [[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        self.kf.predict()
        smoothed = self.kf.correct(meas)   # shape (4, 1)
        self.missing_frames = 0
        return float(smoothed[0][0]), float(smoothed[1][0])

    def predict_only(self):
        """Called when detection is missing this frame."""
        self.missing_frames += 1
        pred = self.kf.predict()           # shape (4, 1)
        return float(pred[0][0]), float(pred[1][0])

    @property
    def alive(self):
        return self.missing_frames < self.MAX_MISSING


# ─────────────────────────────────────────────────────
# Camera helpers (same as v3.1)
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
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"OK  {w}x{h}")
                return cap
            time.sleep(0.1)
        print("no frame"); cap.release()
    return None


def find_camera():
    for i in range(5):
        c = open_camera(i)
        if c:
            print(f"[Camera] Using index {i}")
            return c
    return None


# ─────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────

def draw_hud(disp, feet, fps, conf, alpha, beta, kalman_on, trail_on, tag, dw, dh):
    hc = (0, 220, 110) if feet > 0 else (60, 60, 60)
    cv2.rectangle(disp, (0, 0), (dw, 52), (0, 0, 0), -1)
    cv2.putText(disp,
                f"Feet:{feet}  FPS:{fps:.1f}  Conf:{conf:.2f}  "
                f"Bright:+{beta}  Contrast:x{alpha:.1f}  "
                f"Kalman:{'ON' if kalman_on else 'OFF'}  "
                f"Trail:{'ON' if trail_on else 'OFF'}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, hc, 2)
    cv2.putText(disp,
                f"B/b=bright  C/c=contrast  K=kalman  T=trail  "
                f"R=reset  +/-=conf  S=shot  Q=quit  [{tag}]",
                (10, dh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 120), 1)


def draw_trail(disp, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i-1] and pts[i]:
            alpha_fade = i / len(pts)
            c = tuple(int(v * alpha_fade) for v in color)
            cv2.line(disp, pts[i-1], pts[i], c, 2)


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

    csv_f = csv_w = None
    if save_csv_flag:
        csv_f = open(DETECT_CSV, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","pid","side","raw_x","raw_y",
                         "smooth_x","smooth_y","conf"])

    print(f"\n=== MIVR-CEIQ Foot Detector v3.2  [{tag}] ===")
    print("  B/b=bright  C/c=contrast  K=kalman  T=trail  R=reset  Q=quit\n")

    # Runtime state
    conf_th    = CONF_THRESHOLD
    alpha      = 1.4
    beta       = 40
    kalman_on  = True
    trail_on   = True

    ALPHA_DEF, BETA_DEF = 1.4, 40

    # Kalman trackers: key = (person_id, side)
    trackers: dict = defaultdict(KalmanPoint)

    # Trail history: key = (person_id, side) -> deque of (dx, dy) pixel coords
    from collections import deque
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

        # Brightness / contrast
        frame = cv2.convertScaleAbs(raw, alpha=alpha, beta=beta)

        # Inference
        res = model(frame, conf=conf_th, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)

        h, w   = frame.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(frame, (dw, dh))
        sx, sy = dw / w, dh / h
        feet   = 0

        # Track which keys were seen this frame (to predict missing)
        seen_keys = set()

        if res and len(res) > 0:
            r = res[0]

            if use_pose and r.keypoints is not None:
                kpts = r.keypoints.data

                for pid, pk in enumerate(kpts):
                    for side, aid, kid, col in [
                        ("L", ANKLE_L, KNEE_L, (0, 220, 110)),
                        ("R", ANKLE_R, KNEE_R, (30, 140, 255)),
                    ]:
                        ac = float(pk[aid][2])
                        if ac < conf_th:
                            continue

                        # Raw keypoint position
                        raw_x = float(pk[aid][0])
                        raw_y = float(pk[aid][1])
                        kx    = float(pk[kid][0])
                        ky    = float(pk[kid][1])

                        key = (pid, side)
                        seen_keys.add(key)

                        # Kalman smooth
                        if kalman_on:
                            sx_k, sy_k = trackers[key].update(raw_x, raw_y)
                        else:
                            sx_k, sy_k = raw_x, raw_y

                        # Pixel coords for display
                        dax = int(sx_k * sx); day = int(sy_k * sy)
                        dkx = int(kx * sx);   dky = int(ky * sy)

                        # Trail
                        trails[key].append((dax, day))

                        feet += 1

                        # Draw trail
                        if trail_on:
                            draw_trail(disp, trails[key], col)

                        # Draw ankle
                        cv2.circle(disp, (dax, day), 11, col, -1)
                        cv2.circle(disp, (dax, day), 16, col,  2)

                        # Draw knee + leg line
                        if float(pk[kid][2]) > 0.3:
                            cv2.circle(disp, (dkx, dky), 6, col, 2)
                            cv2.line(disp, (dkx, dky), (dax, day), col, 2)

                        # Draw raw position (small dot) if Kalman is on
                        if kalman_on:
                            rdx = int(raw_x * sx); rdy = int(raw_y * sy)
                            cv2.circle(disp, (rdx, rdy), 4, (60, 60, 60), -1)

                        cv2.putText(disp, f"P{pid}{side} {ac:.2f}",
                                    (dax + 14, day - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

                        if csv_w:
                            csv_w.writerow([
                                f"{time.time():.3f}", pid, side,
                                f"{raw_x:.1f}", f"{raw_y:.1f}",
                                f"{sx_k:.1f}", f"{sy_k:.1f}",
                                f"{ac:.3f}"
                            ])

                # Person boxes (dim)
                if r.boxes is not None:
                    for b in r.boxes:
                        x1,y1,x2,y2 = map(float, b.xyxy[0])
                        cv2.rectangle(disp,
                                      (int(x1*sx), int(y1*sy)),
                                      (int(x2*sx), int(y2*sy)),
                                      (45, 45, 45), 1)

            elif not use_pose and r.boxes is not None:
                for i, b in enumerate(r.boxes):
                    cf = float(b.conf[0])
                    x1,y1,x2,y2 = map(float, b.xyxy[0])
                    cx,cy = (x1+x2)/2, (y1+y2)/2

                    key = (i, "F")
                    seen_keys.add(key)

                    if kalman_on:
                        scx, scy = trackers[key].update(cx, cy)
                    else:
                        scx, scy = cx, cy

                    ddx,ddy = int(scx*sx), int(scy*sy)
                    trails[key].append((ddx,ddy))
                    feet += 1

                    if trail_on:
                        draw_trail(disp, trails[key], (0,220,110))

                    dx1,dy1=int(x1*sx),int(y1*sy)
                    dx2,dy2=int(x2*sx),int(y2*sy)
                    cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,220,110),2)
                    cv2.circle(disp,(ddx,ddy),6,(0,220,110),-1)
                    cv2.putText(disp,f"foot{i} {cf:.2f}",
                                (dx1,dy1-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,110),1)

        # Predict-only for unseen trackers (keeps trail alive briefly)
        if kalman_on:
            dead = []
            for key, tkr in trackers.items():
                if key not in seen_keys:
                    tkr.predict_only()
                    if not tkr.alive:
                        dead.append(key)
            for key in dead:
                del trackers[key]
                trails.pop(key, None)

        # FPS & HUD
        fps_buf.append(time.time() - t0)
        if len(fps_buf) > 30: fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf))

        draw_hud(disp, feet, fps, conf_th, alpha, beta,
                 kalman_on, trail_on, tag, dw, dh)

        cv2.imshow("MIVR-CEIQ Foot Detection v3.2", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break

        elif key in (ord('s'),ord('S')):
            ts = int(time.time())
            cv2.imwrite(f"{DATA_DIR}/shot_raw_{ts}.jpg",    raw)
            cv2.imwrite(f"{DATA_DIR}/shot_bright_{ts}.jpg", frame)
            print(f"  [shot] {ts}")

        elif key in (ord('+'),ord('=')):
            conf_th = min(conf_th+0.05, 0.95); print(f"  conf->{conf_th:.2f}")
        elif key == ord('-'):
            conf_th = max(conf_th-0.05, 0.05); print(f"  conf->{conf_th:.2f}")

        elif key == ord('B'): beta  = min(beta+5, 150);         print(f"  bright->{beta}")
        elif key == ord('b'): beta  = max(beta-5, -50);         print(f"  bright->{beta}")
        elif key == ord('C'): alpha = min(round(alpha+0.1,1),3.0); print(f"  contrast->{alpha}")
        elif key == ord('c'): alpha = max(round(alpha-0.1,1),0.5); print(f"  contrast->{alpha}")

        elif key in (ord('k'),ord('K')):
            kalman_on = not kalman_on
            if not kalman_on:
                trackers.clear()
                for t in trails.values(): t.clear()
            print(f"  Kalman: {'ON' if kalman_on else 'OFF'}")

        elif key in (ord('t'),ord('T')):
            trail_on = not trail_on
            print(f"  Trail: {'ON' if trail_on else 'OFF'}")

        elif key in (ord('r'),ord('R')):
            alpha, beta = ALPHA_DEF, BETA_DEF
            trackers.clear()
            for t in trails.values(): t.clear()
            print(f"  Reset: bright={beta} contrast={alpha} Kalman states cleared")

        frame_id += 1

    cap.release(); cv2.destroyAllWindows()
    if csv_f: csv_f.close(); print(f"[CSV] {DETECT_CSV}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v3.2")
    p.add_argument("--camera", type=int, default=0, help="-1=auto scan")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-kalman", action="store_true", help="disable Kalman at start")
    a = p.parse_args()
    print("="*55)
    print("  MIVR-CEIQ Foot Detector v3.2  [Kalman smoothing]")
    print("="*55)
    detect(a.camera, not a.no_csv)


if __name__ == "__main__":
    main()
