"""
MIVR-CEIQ Foot Detector v3.1
Patch: live brightness/contrast adjustment
  B / b  ->  brightness  +5 / -5   (default +40)
  C / c  ->  contrast    +0.1/-0.1  (default x1.4)
  R      ->  reset to default
  Q      ->  quit
  S      ->  screenshot (saves raw + brightened)
  +/-    ->  detection confidence threshold
"""

import cv2, numpy as np, argparse, time, os, csv
from pathlib import Path

# ── same paths as v3 ──────────────────────────────────
DATA_DIR      = "data/foot_detection"
RAW_DIR       = f"{DATA_DIR}/raw_images"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections.csv"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640
ANKLE_L, ANKLE_R, KNEE_L, KNEE_R = 15, 16, 13, 14
# ─────────────────────────────────────────────────────


def open_camera(index):
    for backend, name in [(cv2.CAP_DSHOW,"DSHOW"),
                          (cv2.CAP_MSMF,"MSMF"),
                          (cv2.CAP_ANY,"ANY")]:
        print(f"  cam {index} [{name}]...", end=" ", flush=True)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened(): print("fail"); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(20):
            ret, f = cap.read()
            if ret and f is not None and f.size > 0 and f.mean() > 2.0:
                print(f"OK {int(cap.get(3))}x{int(cap.get(4))}")
                return cap
            time.sleep(0.1)
        print("no frame"); cap.release()
    return None


def find_camera():
    for i in range(5):
        c = open_camera(i)
        if c: return c
    return None


def draw_hud(disp, feet, fps, conf, alpha, beta, tag, dw, dh):
    """Single-line top HUD + bottom hint, English only."""
    hc = (0, 220, 110) if feet > 0 else (70, 70, 70)
    cv2.rectangle(disp, (0, 0), (dw, 50), (0, 0, 0), -1)
    cv2.putText(disp,
                f"Feet:{feet}  FPS:{fps:.1f}  "
                f"Conf:{conf:.2f}  "
                f"Bright:+{beta}  Contrast:x{alpha:.1f}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.63, hc, 2)
    cv2.putText(disp,
                "B/b=bright  C/c=contrast  R=reset  +/-=conf  S=shot  Q=quit"
                f"  [{tag}]",
                (10, dh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 130), 1)


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
        print(f"[Model] {tag}  (auto-download ~6MB first run)")

    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None:
        print("[ERROR] No camera. Try --camera -1 or check USB mode (PC Remote).")
        return

    csv_f = csv_w = None
    if save_csv_flag:
        csv_f = open(DETECT_CSV, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","pid","side","cx","cy","conf","kx","ky"])

    print(f"\n=== MIVR-CEIQ Foot Detector v3.1  [{tag}] ===")
    print("  B/b=bright+/-5  C/c=contrast+/-0.1  R=reset")
    print("  +/-=conf  S=screenshot  Q=quit\n")

    conf_th  = CONF_THRESHOLD
    alpha    = 1.4    # contrast multiplier  (raw * alpha + beta)
    beta     = 40     # brightness offset
    fps_buf  = []
    frame_id = 0
    black_n  = 0

    # default saved for reset
    ALPHA_DEF, BETA_DEF = 1.4, 40

    while True:
        t0 = time.time()
        ret, raw = cap.read()
        if not ret or raw is None:
            time.sleep(0.05); continue
        if raw.mean() < 2.0:
            black_n += 1
            if black_n % 15 == 0:
                print(f"[WARN] {black_n} black frames — check A7R IV USB mode")
            frame_id += 1; continue
        black_n = 0

        # ── brightness / contrast ──
        frame = cv2.convertScaleAbs(raw, alpha=alpha, beta=beta)

        # ── inference ──
        res = model(frame, conf=conf_th, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)

        h, w   = frame.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(frame, (dw, dh))
        sx, sy = dw/w, dh/h
        feet   = 0

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
                        if ac < conf_th: continue
                        ax,ay = float(pk[aid][0]), float(pk[aid][1])
                        kx,ky = float(pk[kid][0]), float(pk[kid][1])
                        dax,day = int(ax*sx), int(ay*sy)
                        dkx,dky = int(kx*sx), int(ky*sy)
                        feet += 1
                        cv2.circle(disp, (dax,day), 11, col, -1)
                        cv2.circle(disp, (dax,day), 16, col,  2)
                        if float(pk[kid][2]) > 0.3:
                            cv2.circle(disp,(dkx,dky),6,col,2)
                            cv2.line(disp,(dkx,dky),(dax,day),col,2)
                        cv2.putText(disp, f"P{pid}{side} {ac:.2f}",
                                    (dax+14,day-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
                        if csv_w:
                            csv_w.writerow([f"{time.time():.3f}",pid,side,
                                            f"{ax:.1f}",f"{ay:.1f}",f"{ac:.3f}",
                                            f"{kx:.1f}",f"{ky:.1f}"])
                if r.boxes is not None:
                    for b in r.boxes:
                        x1,y1,x2,y2 = map(float,b.xyxy[0])
                        cv2.rectangle(disp,
                            (int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),
                            (45,45,45),1)

            elif not use_pose and r.boxes is not None:
                for i, b in enumerate(r.boxes):
                    cf = float(b.conf[0])
                    x1,y1,x2,y2 = map(float,b.xyxy[0])
                    cx,cy = (x1+x2)/2,(y1+y2)/2
                    dx1,dy1=int(x1*sx),int(y1*sy)
                    dx2,dy2=int(x2*sx),int(y2*sy)
                    feet += 1
                    cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,220,110),2)
                    cv2.circle(disp,(int(cx*sx),int(cy*sy)),5,(0,220,110),-1)
                    cv2.putText(disp,f"foot{i} {cf:.2f}",
                                (dx1,dy1-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,110),1)
                    if csv_w:
                        csv_w.writerow([f"{time.time():.3f}",i,"?",
                                        f"{cx:.1f}",f"{cy:.1f}",f"{cf:.3f}","",""])

        # ── FPS & HUD ──
        fps_buf.append(time.time()-t0)
        if len(fps_buf)>30: fps_buf.pop(0)
        fps = 1.0/(sum(fps_buf)/len(fps_buf))
        draw_hud(disp, feet, fps, conf_th, alpha, beta, tag, dw, dh)

        cv2.imshow("MIVR-CEIQ Foot Detection v3.1", disp)

        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('q'),ord('Q')): break
        elif key in (ord('s'),ord('S')):
            ts = int(time.time())
            p1 = f"{DATA_DIR}/shot_raw_{ts}.jpg"
            p2 = f"{DATA_DIR}/shot_bright_{ts}.jpg"
            cv2.imwrite(p1, raw)
            cv2.imwrite(p2, frame)
            print(f"  [shot] raw={p1}  bright={p2}")
        elif key in (ord('+'),ord('=')):
            conf_th=min(conf_th+0.05,0.95); print(f"  conf->{conf_th:.2f}")
        elif key == ord('-'):
            conf_th=max(conf_th-0.05,0.05); print(f"  conf->{conf_th:.2f}")
        elif key == ord('B'):                          # B  = brighter
            beta=min(beta+5,150); print(f"  brightness->{beta}")
        elif key == ord('b'):                          # b  = darker
            beta=max(beta-5,-50); print(f"  brightness->{beta}")
        elif key == ord('C'):                          # C  = more contrast
            alpha=min(round(alpha+0.1,1),3.0); print(f"  contrast->{alpha}")
        elif key == ord('c'):                          # c  = less contrast
            alpha=max(round(alpha-0.1,1),0.5); print(f"  contrast->{alpha}")
        elif key in (ord('r'),ord('R')):               # R  = reset
            alpha,beta = ALPHA_DEF,BETA_DEF
            print(f"  reset -> bright={beta} contrast={alpha}")

        frame_id += 1

    cap.release(); cv2.destroyAllWindows()
    if csv_f: csv_f.close(); print(f"[CSV] {DETECT_CSV}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v3.1")
    p.add_argument("--camera", type=int, default=0, help="-1=auto scan")
    p.add_argument("--no-csv", action="store_true")
    a = p.parse_args()
    print("="*52)
    print("  MIVR-CEIQ Foot Detector v3.1  [brightness fix]")
    print("="*52)
    detect(a.camera, not a.no_csv)

if __name__ == "__main__":
    main()
