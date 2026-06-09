"""
MIVR-CEIQ Foot Detector v6  [即時追蹤 + 快速移動修正版]
foot_detector_v5.py

修正 v4 的四個漏洞：
  [Bug-1] import 改用 adaptive_kalman_v3（最新優化版）
  [Bug-2] AKFConfig 改用 v3 欄位名（r_noise_base / outlier_gate_min 等）
  [Bug-3] person_id 改用 YOLOv8 ByteTrack 穩定 ID（model.track）
          → 根治多人靠近時 ID 互換、軌跡錯亂
  [Bug-4] 傳遞 norm_y = ankle_y / frame_height 給 AKF
          → 啟用距離自適應量測噪音

需要的檔案（同目錄）：
  adaptive_kalman_v3.py
  bio_fusion_tracker_v2.py   （見下方，已改 import v3）

Controls:
  B/b brightness  C/c contrast  K bio  A arms  D debug  T trail
  R reset  S screenshot  +/- conf  Q quit
"""

import cv2, numpy as np, argparse, time, os, csv
from pathlib import Path
from collections import deque, defaultdict

try:
    from adaptive_kalman_v4 import AKFConfig
    from bio_fusion_tracker_v3 import (
        MultiBioFusionTracker,
        draw_ankle_estimate, draw_gait_indicator,
        KP, CONF_VISIBLE, SOURCE_COLORS
    )
    BIO_AVAILABLE = True
except ImportError as e:
    BIO_AVAILABLE = False
    print(f"[WARN] bio_fusion_tracker_v3 not found ({e})")

DATA_DIR      = "data/foot_detection"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections_v5.csv"
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

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
ARM_KP_IDS  = [5,6,7,8,9,10]
LEG_KP_IDS  = [11,12,13,14,15,16]
ARM_COLOR   = (0, 165, 255)
LEG_COLOR   = (80, 80, 200)
TORSO_COLOR = (60, 60, 60)


def draw_skeleton(disp, kpts, sx, sy, show_arms=True, conf_th=0.3):
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
    if show_arms:
        for wid in [KP["l_wrist"], KP["r_wrist"]]:
            if float(kpts[wid][2]) > conf_th:
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
             bio_on, show_arms, debug_on, track_on, tag, dw, dh):
    hc = (0,220,110) if feet > 0 else (60,60,60)
    cv2.rectangle(disp,(0,0),(dw,52),(0,0,0),-1)
    cv2.putText(disp,
                f"Feet:{feet}  FPS:{fps:.1f}  Conf:{conf:.2f}  "
                f"Bright:+{beta}  Cx{alpha:.1f}  "
                f"Bio:{'ON' if bio_on else 'OFF'}  "
                f"Track:{'ByteTrack' if track_on else 'enum'}",
                (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.54,hc,2)
    cv2.putText(disp,
                "B/b bright  C/c contrast  K bio  A arms  D debug  "
                "T trail  R reset  +/- conf  S  Q"
                f"  [{tag}]",
                (10,dh-9),cv2.FONT_HERSHEY_SIMPLEX,0.34,(70,70,120),1)
    legend = [
        ("D direct",       SOURCE_COLORS["direct"]),
        ("F fused",        SOURCE_COLORS["low_conf_fused"]),
        ("E extrapolated", SOURCE_COLORS["extrapolated"]),
        ("P predicted",    SOURCE_COLORS["predicted"]),
    ]
    lx, ly = dw - 165, dh - 80
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

    model = YOLO("yolov8n-pose.pt")
    tag   = "Pose+BioFusion+ByteTrack" if BIO_AVAILABLE else "Pose+ByteTrack"
    print(f"[Model] {tag}")

    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None:
        print("[ERROR] No camera."); return

    # 用 v4 欄位建立 config（已移除 EWMA/互補濾波參數；改用 DWNA Q + StationaryLock）
    if BIO_AVAILABLE:
        akf_cfg = AKFConfig(
            # ── 量測噪音：資料驅動，零相機假設（與 v3 相同精神）──
            r_noise_base=5.0, r_noise_max=60.0,
            r_residual_window=15, r_residual_gain=0.6,
            r_dist_scale=0.0,            # 0 = 不用畫面位置假設
            # ── 靜止判定 / 閘門（v4 已調校預設，這裡顯式列出重點）──
            still_threshold=6.0,         # px/幀：低於此暫判靜止（半窗均值差，抗噪音）
            reacquire_after=2,           # 連續同向遠離 2 幀 → 重新鎖定（破 v3 deadlock）
            outlier_gate_min=45.0, outlier_gate_vel=3.5,
            # 其餘（DWNA q_base/q_max、pos_noise、StationaryLock 各門檻）採 v4 調校預設
        )
        bio_tracker = MultiBioFusionTracker(akf_cfg)
    else:
        bio_tracker = None

    csv_f = csv_w = None
    if save_csv_flag:
        csv_f = open(DETECT_CSV,"w",newline="",encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","track_id","side","x","y","conf","source"])

    print(f"\n=== Foot Detector v6 [{tag}] ===")
    print("  K bio  A arms  D debug  T trail  B/b  C/c  R  S  Q\n")

    conf_th     = CONF_THRESHOLD
    alpha, beta = 1.4, 40
    bio_on      = BIO_AVAILABLE
    show_arms   = True
    debug_on    = False
    trail_on    = True
    track_on    = True   # ByteTrack 穩定 ID
    clahe_on    = False  # [v6] YOLO 輸入用 CLAHE（低光才開；預設給 YOLO 乾淨原圖）
    ALPHA_DEF, BETA_DEF = 1.4, 40
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

        # ── [v6 核心修正] 偵測輸入 與 顯示影像 分離 ──
        # v5 把 convertScaleAbs(α=1.4,β=40) 後的影像直接餵給 YOLO，
        # 過度提亮/拉對比會削峰、扭曲 → YOLO 關鍵點偵測變差、追不到人。
        # v6：YOLO 看「乾淨原圖」(或低光時的 CLAHE)，提亮只用於人眼顯示。
        if clahe_on:
            ycc = cv2.cvtColor(raw, cv2.COLOR_BGR2YCrCb)
            ycc[:, :, 0] = _clahe.apply(ycc[:, :, 0])    # 只均衡亮度通道
            det_img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        else:
            det_img = raw                                  # YOLO 吃原圖
        disp_src = cv2.convertScaleAbs(raw, alpha=alpha, beta=beta)  # 僅顯示用

        # 用 track() 取得穩定 ID（內建 ByteTrack）
        if track_on:
            res = model.track(det_img, conf=conf_th, iou=IOU_THRESHOLD,
                              imgsz=IMG_SIZE, persist=True,
                              tracker="bytetrack.yaml", verbose=False)
        else:
            res = model(det_img, conf=conf_th, iou=IOU_THRESHOLD,
                        imgsz=IMG_SIZE, verbose=False)

        h, w   = det_img.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(disp_src,(dw,dh))
        sx_s   = dw/w; sy_s = dh/h
        feet   = 0

        if res and len(res) > 0:
            r = res[0]

            if r.keypoints is not None:
                kpts_all = r.keypoints.data   # (N, 17, 3)

                # [Bug-3 修正] 取得穩定 track ID 列表
                if track_on and r.boxes is not None and r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int).tolist()
                else:
                    track_ids = list(range(len(kpts_all)))   # fallback

                for idx, kpts in enumerate(kpts_all):
                    kpts_np = kpts.cpu().numpy() if hasattr(kpts,'cpu') else np.array(kpts)
                    # 用穩定 track_id 而非 enumerate 順序
                    pid = track_ids[idx] if idx < len(track_ids) else idx

                    draw_skeleton(disp, kpts_np, sx_s, sy_s,
                                  show_arms=show_arms, conf_th=conf_th)

                    if bio_on and bio_tracker is not None:
                        # [Bug-4 修正] 傳 frame_h 讓 bio_fusion 算 norm_y
                        estimates = bio_tracker.process_person(pid, kpts_np, h, w)

                        for side, est in estimates.items():
                            if est.confidence < 0.05: continue
                            feet += 1
                            col = SOURCE_COLORS.get(est.source,(100,100,100))
                            key = (pid, side)
                            dax = int(est.x * sx_s); day = int(est.y * sy_s)
                            trails[key].append((dax, day))
                            if trail_on: draw_trail(disp, trails[key], col)
                            draw_ankle_estimate(
                                disp, est, pid, side, sx_s, sy_s, debug_on,
                                kpts_np if debug_on else None, w, h)
                            if csv_w:
                                csv_w.writerow([f"{time.time():.3f}", pid, side,
                                    f"{est.x:.1f}", f"{est.y:.1f}",
                                    f"{est.confidence:.3f}", est.source])

                        gait = bio_tracker.get_gait_info(pid)
                        if gait:
                            draw_gait_indicator(disp, gait, pid,
                                                10 + (idx%4)*200, dh-30)
                    else:
                        for side, aid, kid, col in [
                            ("L",15,13,(0,220,110)),("R",16,14,(30,140,255))]:
                            ac = float(kpts_np[aid][2])
                            if ac < conf_th: continue
                            ax = float(kpts_np[aid][0]); ay = float(kpts_np[aid][1])
                            dax=int(ax*sx_s); day=int(ay*sy_s)
                            key=(pid,side)
                            trails[key].append((dax,day))
                            if trail_on: draw_trail(disp,trails[key],col)
                            cv2.circle(disp,(dax,day),11,col,-1)
                            cv2.circle(disp,(dax,day),16,col,2)
                            feet += 1

                # Person boxes + track ID 標籤
                if r.boxes is not None:
                    for bi, b in enumerate(r.boxes):
                        x1,y1,x2,y2=map(float,b.xyxy[0])
                        tid = track_ids[bi] if bi < len(track_ids) else bi
                        cv2.rectangle(disp,
                            (int(x1*sx_s),int(y1*sy_s)),
                            (int(x2*sx_s),int(y2*sy_s)),(40,40,40),1)
                        cv2.putText(disp, f"ID:{tid}",
                            (int(x1*sx_s),int(y1*sy_s)-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,120,180),1)

        fps_buf.append(time.time()-t0)
        if len(fps_buf)>30: fps_buf.pop(0)
        fps = 1.0/(sum(fps_buf)/len(fps_buf))

        # [Bug-5 修正] 週期性清理 trail dict，防止記憶體無限增長
        if frame_id % 60 == 0 and len(trails) > 64:
            # 移除已不在畫面、且 trail 已空的 key
            active_keys = set()
            if bio_tracker is not None:
                for pid in list(bio_tracker._trackers.keys()):
                    active_keys.add((pid, "L"))
                    active_keys.add((pid, "R"))
            stale = [k for k in trails if k not in active_keys]
            for k in stale:
                del trails[k]

        draw_hud(disp, feet, fps, conf_th, alpha, beta,
                 bio_on, show_arms, debug_on, track_on, tag, dw, dh)
        cv2.imshow("MIVR-CEIQ Foot Detector v6", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break
        elif key in (ord('s'),ord('S')):
            ts=int(time.time()); cv2.imwrite(f"{DATA_DIR}/shot_{ts}.jpg",disp_src)
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
            print(f"  Bio: {'ON' if bio_on else 'OFF'}")
        elif key in (ord('a'),ord('A')):
            show_arms = not show_arms; print(f"  Arms: {'ON' if show_arms else 'OFF'}")
        elif key in (ord('d'),ord('D')):
            debug_on = not debug_on; print(f"  Debug: {'ON' if debug_on else 'OFF'}")
        elif key in (ord('t'),ord('T')):
            trail_on = not trail_on; print(f"  Trail: {'ON' if trail_on else 'OFF'}")
        elif key in (ord('g'),ord('G')):
            clahe_on = not clahe_on
            print(f"  CLAHE(YOLO輸入): {'ON (低光增強)' if clahe_on else 'OFF (原圖)'}")
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
    p=argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v6")
    p.add_argument("--camera",type=int,default=0,help="-1=auto scan")
    p.add_argument("--no-csv",action="store_true")
    a=p.parse_args()
    print("="*58)
    print("  MIVR-CEIQ Foot Detector v6  [realtime + fast-move fixed]")
    print("="*58)
    detect(a.camera,not a.no_csv)

if __name__=="__main__":
    main()
