"""
MIVR-CEIQ Foot Detector v4  [Bio-Fusion]
foot_detector_v4.py

新增：整合 bio_fusion_tracker.py
  - 17 點全骨架可見度
  - 手腕/手肘輔助腳踝定位
  - 腳踝遮擋時自動切換 extrapolated / predicted
  - 步態節律 HUD 指示器

需要的檔案（同目錄）：
  adaptive_kalman.py
  bio_fusion_tracker.py

Controls:
  B/b  brightness  C/c  contrast
  K    Kalman/BioFusion on/off
  A    show/hide arm landmarks
  D    debug overlay
  T    trail
  R    reset   S  screenshot   Q  quit   +/-  conf
"""

import cv2, numpy as np, argparse, time, os, csv
from pathlib import Path
from collections import deque, defaultdict

try:
    from adaptive_kalman_v3 import AdaptiveKalmanPoint, AKFConfig, MultiFootTracker
    from bio_fusion_tracker import (
        MultiBioFusionTracker, AKFConfig,
        draw_ankle_estimate, draw_gait_indicator,
        KP, CONF_VISIBLE, SOURCE_COLORS
    )
    BIO_AVAILABLE = True
except ImportError as e:
    BIO_AVAILABLE = False
    print(f"[WARN] bio_fusion_tracker not found ({e}), using basic Kalman")

DATA_DIR      = "data/foot_detection"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections_v4.csv"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640
TRAIL_LEN      = 30


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
        if c: print(f"[Camera] index {i}"); return c
    return None


# ─────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────

# 骨架連線定義（YOLOv8-Pose COCO）
SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),    # 上半身
    (5,11),(6,12),(11,12),              # 軀幹
    (11,13),(13,15),(12,14),(14,16),    # 下半身
]

ARM_KP_IDS  = [5,6,7,8,9,10]   # 肩/肘/腕
LEG_KP_IDS  = [11,12,13,14,15,16]

ARM_COLOR   = (0, 165, 255)     # 橙：手臂
LEG_COLOR   = (80, 80, 200)     # 暗藍：腿部骨架（腳踝用 BioFusion 顏色覆蓋）
TORSO_COLOR = (60,  60,  60)    # 灰：軀幹


def draw_skeleton(disp, kpts, sx, sy, show_arms=True, conf_th=0.3):
    """繪製半透明骨架輔助線"""
    for a, b in SKELETON:
        ca = float(kpts[a][2]); cb = float(kpts[b][2])
        if ca < conf_th or cb < conf_th: continue
        ax = int(float(kpts[a][0]) * sx); ay = int(float(kpts[a][1]) * sy)
        bx = int(float(kpts[b][0]) * sx); by = int(float(kpts[b][1]) * sy)

        if a in ARM_KP_IDS and b in ARM_KP_IDS:
            if not show_arms: continue
            col = ARM_COLOR
        elif a in LEG_KP_IDS and b in LEG_KP_IDS:
            col = LEG_COLOR
        else:
            col = TORSO_COLOR

        cv2.line(disp, (ax,ay), (bx,by), col, 1)

    # 手腕點（橙圓）
    if show_arms:
        for wid in [KP["l_wrist"], KP["r_wrist"]]:
            wc = float(kpts[wid][2])
            if wc > conf_th:
                wx = int(float(kpts[wid][0]) * sx)
                wy = int(float(kpts[wid][1]) * sy)
                cv2.circle(disp, (wx,wy), 5, ARM_COLOR, -1)


def draw_trail(disp, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i-1] and pts[i]:
            fade = i / len(pts)
            c = tuple(int(v*fade) for v in color)
            cv2.line(disp, pts[i-1], pts[i], c, 2)


def draw_hud(disp, feet, fps, conf, alpha, beta,
             bio_on, show_arms, debug_on, tag, dw, dh):
    hc = (0,220,110) if feet > 0 else (60,60,60)
    cv2.rectangle(disp,(0,0),(dw,52),(0,0,0),-1)
    cv2.putText(disp,
                f"Feet:{feet}  FPS:{fps:.1f}  Conf:{conf:.2f}  "
                f"Bright:+{beta}  Cx{alpha:.1f}  "
                f"BioFusion:{'ON' if bio_on else 'OFF'}  "
                f"Arms:{'ON' if show_arms else 'OFF'}",
                (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.57,hc,2)
    cv2.putText(disp,
                "B/b=bright  C/c=contrast  K=bio  A=arms  "
                "D=debug  T=trail  R=reset  +/-=conf  S  Q"
                f"  [{tag}]",
                (10,dh-9),cv2.FONT_HERSHEY_SIMPLEX,0.35,(70,70,120),1)

    # 圖例（右下角）
    legend = [
        ("D=direct",       SOURCE_COLORS["direct"]),
        ("F=fused",        SOURCE_COLORS["low_conf_fused"]),
        ("E=extrapolated", SOURCE_COLORS["extrapolated"]),
        ("P=predicted",    SOURCE_COLORS["predicted"]),
    ]
    lx, ly = dw - 160, dh - 80
    for i, (txt, col) in enumerate(legend):
        cv2.circle(disp,(lx,ly+i*16),4,col,-1)
        cv2.putText(disp,txt,(lx+10,ly+i*16+4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,col,1)


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

    model    = YOLO("yolov8n-pose.pt"); use_pose = True
    tag      = "YOLOv8n-Pose+BioFusion" if BIO_AVAILABLE else "YOLOv8n-Pose"
    print(f"[Model] {tag}")

    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None:
        print("[ERROR] No camera."); return

    # BioFusion tracker
    if BIO_AVAILABLE:
        akf_cfg = AKFConfig(
            q_base=0.5, q_max=800.0, r_noise_base=8.0,
            innovation_window=5, innovation_gain=6.0,
            velocity_decay=0.75, still_threshold=2.5,
            still_damping=0.05, outlier_gate=120.0, max_missing=12
        )
        bio_tracker = MultiBioFusionTracker(akf_cfg)
    else:
        bio_tracker = None

    csv_f = csv_w = None
    if save_csv_flag:
        csv_f = open(DETECT_CSV,"w",newline="",encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","pid","side","x","y","conf","source"])

    print(f"\n=== Foot Detector v4 [{tag}] ===")
    print("  K=bio  A=arms  D=debug  T=trail  B/b  C/c  R  S  Q\n")

    conf_th   = CONF_THRESHOLD
    alpha, beta = 1.4, 40
    bio_on    = BIO_AVAILABLE
    show_arms = True
    debug_on  = False
    trail_on  = True
    ALPHA_DEF, BETA_DEF = 1.4, 40

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    fps_buf = []; frame_id = 0; black_n = 0

    while True:
        t0 = time.time()
        ret, raw = cap.read()
        if not ret or raw is None: time.sleep(0.05); continue
        if raw.mean() < 2.0:
            black_n += 1
            if black_n % 15 == 0: print(f"[WARN] {black_n} black frames")
            frame_id += 1; continue
        black_n = 0

        frame = cv2.convertScaleAbs(raw, alpha=alpha, beta=beta)
        res   = model(frame, conf=conf_th, iou=IOU_THRESHOLD,
                      imgsz=IMG_SIZE, verbose=False)

        h, w   = frame.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(frame,(dw,dh))
        sx_s   = dw/w; sy_s = dh/h
        feet   = 0

        if res and len(res) > 0:
            r = res[0]

            if r.keypoints is not None:
                kpts_all = r.keypoints.data   # (N, 17, 3)

                for pid, kpts in enumerate(kpts_all):
                    kpts_np = kpts.cpu().numpy() if hasattr(kpts,'cpu') else np.array(kpts)

                    # 骨架
                    draw_skeleton(disp, kpts_np, sx_s, sy_s,
                                  show_arms=show_arms, conf_th=conf_th)

                    if bio_on and bio_tracker is not None:
                        # BioFusion 處理
                        estimates = bio_tracker.process_person(pid, kpts_np, h, w)

                        for side, est in estimates.items():
                            if est.confidence < 0.05: continue
                            feet += 1
                            col = SOURCE_COLORS.get(est.source,(100,100,100))
                            key = (pid, side)

                            # Trail
                            dax = int(est.x * sx_s)
                            day = int(est.y * sy_s)
                            trails[key].append((dax, day))
                            if trail_on:
                                draw_trail(disp, trails[key], col)

                            # 繪製估算點
                            draw_ankle_estimate(
                                disp, est, pid, side,
                                sx_s, sy_s, debug_on,
                                kpts_np if debug_on else None, w, h
                            )

                            if csv_w:
                                csv_w.writerow([
                                    f"{time.time():.3f}", pid, side,
                                    f"{est.x:.1f}", f"{est.y:.1f}",
                                    f"{est.confidence:.3f}", est.source
                                ])

                        # 步態指示器（左上角）
                        gait = bio_tracker.get_gait_info(pid)
                        if gait:
                            draw_gait_indicator(disp, gait, pid,
                                                10 + pid*200, dh-30)

                    else:
                        # Fallback：直接顯示腳踝點
                        for side, aid, kid, col in [
                            ("L",15,13,(0,220,110)),
                            ("R",16,14,(30,140,255))
                        ]:
                            ac = float(kpts_np[aid][2])
                            if ac < conf_th: continue
                            ax = float(kpts_np[aid][0])
                            ay = float(kpts_np[aid][1])
                            dax=int(ax*sx_s); day=int(ay*sy_s)
                            key=(pid,side)
                            trails[key].append((dax,day))
                            if trail_on: draw_trail(disp,trails[key],col)
                            cv2.circle(disp,(dax,day),11,col,-1)
                            cv2.circle(disp,(dax,day),16,col,2)
                            feet += 1

                # Person boxes
                if r.boxes is not None:
                    for b in r.boxes:
                        x1,y1,x2,y2=map(float,b.xyxy[0])
                        cv2.rectangle(disp,
                            (int(x1*sx_s),int(y1*sy_s)),
                            (int(x2*sx_s),int(y2*sy_s)),(40,40,40),1)

        fps_buf.append(time.time()-t0)
        if len(fps_buf)>30: fps_buf.pop(0)
        fps = 1.0/(sum(fps_buf)/len(fps_buf))

        draw_hud(disp, feet, fps, conf_th, alpha, beta,
                 bio_on, show_arms, debug_on, tag, dw, dh)

        cv2.imshow("MIVR-CEIQ Foot Detector v4", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break
        elif key in (ord('s'),ord('S')):
            ts=int(time.time())
            cv2.imwrite(f"{DATA_DIR}/shot_{ts}.jpg",frame)
            print(f"  [shot] {ts}")
        elif key in (ord('+'),ord('=')): conf_th=min(conf_th+0.05,0.95); print(f"  conf->{conf_th:.2f}")
        elif key==ord('-'):              conf_th=max(conf_th-0.05,0.05); print(f"  conf->{conf_th:.2f}")
        elif key==ord('B'): beta =min(beta+5,150);             print(f"  bright->{beta}")
        elif key==ord('b'): beta =max(beta-5,-50);             print(f"  bright->{beta}")
        elif key==ord('C'): alpha=min(round(alpha+0.1,1),3.0); print(f"  contrast->{alpha}")
        elif key==ord('c'): alpha=max(round(alpha-0.1,1),0.5); print(f"  contrast->{alpha}")
        elif key in (ord('k'),ord('K')):
            bio_on = not bio_on
            if bio_tracker and not bio_on: bio_tracker.reset_all()
            for t in trails.values(): t.clear()
            print(f"  BioFusion: {'ON' if bio_on else 'OFF'}")
        elif key in (ord('a'),ord('A')):
            show_arms = not show_arms; print(f"  Arms: {'ON' if show_arms else 'OFF'}")
        elif key in (ord('d'),ord('D')):
            debug_on = not debug_on; print(f"  Debug: {'ON' if debug_on else 'OFF'}")
        elif key in (ord('t'),ord('T')):
            trail_on = not trail_on; print(f"  Trail: {'ON' if trail_on else 'OFF'}")
        elif key in (ord('r'),ord('R')):
            alpha,beta=ALPHA_DEF,BETA_DEF
            if bio_tracker: bio_tracker.reset_all()
            for t in trails.values(): t.clear()
            print("  Reset all")
        frame_id += 1

    cap.release(); cv2.destroyAllWindows()
    if csv_f: csv_f.close(); print(f"[CSV] {DETECT_CSV}")
    print("Done.")


def main():
    p=argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v4")
    # 將 default 改為 -1，當你直接按下播放鍵時，它會自動從 0 掃描到 4 號相機
    p.add_argument("--camera", type=int, default=-1, help="-1=auto scan") 
    p.add_argument("--no-csv", action="store_true")
    a=p.parse_args()
    
    print("="*58)
    print("  MIVR-CEIQ Foot Detector v4  [Bio-Fusion]")
    print("="*58)
    detect(a.camera, not a.no_csv)

if __name__=="__main__":
    main()
