"""
MIVR-CEIQ Foot Detector v3
Fixes:
  [1] Chinese garbled text -> English HUD only
  [2] Black screen -> tries DirectShow / MSMF / Auto backends
  [3] No mediapipe dependency -> YOLOv8n-pose only
"""

import cv2
import numpy as np
import argparse
import time
import os
import csv
import shutil
from pathlib import Path

DATA_DIR      = "data/foot_detection"
RAW_DIR       = f"{DATA_DIR}/raw_images"
LABEL_DIR     = f"{DATA_DIR}/labels"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections.csv"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

# YOLOv8-Pose COCO keypoint indices
ANKLE_L = 15   # left_ankle
ANKLE_R = 16   # right_ankle
KNEE_L  = 13   # left_knee
KNEE_R  = 14   # right_knee


def ensure_dirs():
    for d in [RAW_DIR, LABEL_DIR, MODEL_DIR, "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────

def open_camera(index: int):
    """Try multiple backends; return working VideoCapture or None."""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF,  "MediaFoundation"),
        (cv2.CAP_ANY,   "Any"),
    ]
    for backend, name in backends:
        print(f"  Camera {index} [{name}] ...", end=" ", flush=True)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print("open failed"); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(20):           # wait up to ~2s for first frame
            ret, f = cap.read()
            if ret and f is not None and f.size > 0 and f.mean() > 2.0:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"OK  {w}x{h}")
                return cap
            time.sleep(0.1)
        print("no valid frame"); cap.release()
    return None


def find_camera():
    print("[Camera] Scanning index 0-4 ...")
    for i in range(5):
        cap = open_camera(i)
        if cap:
            print(f"[Camera] Using index {i}")
            return cap
    return None


# ─────────────────────────────────────────────────────
# Detect
# ─────────────────────────────────────────────────────

def detect_mode(camera_index: int, save_csv_flag: bool):
    ensure_dirs()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics"); return

    if os.path.exists(TRAINED_MODEL):
        model    = YOLO(TRAINED_MODEL)
        use_pose = False
        tag      = "Custom foot model"
    else:
        model    = YOLO("yolov8n-pose.pt")   # auto-download ~6 MB
        use_pose = True
        tag      = "YOLOv8n-Pose (ankle kpts)"
        print(f"[Model] {tag}  (first run downloads ~6 MB)\n")

    print("[Camera] Opening ...")
    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None:
        print("\n[ERROR] Cannot open camera.")
        print("  Tips:")
        print("  - A7R IV USB mode -> 'PC Remote (Still Img)'")
        print("  - Try --camera 1  or  --camera -1  (auto scan)")
        print("  - Restart Imaging Edge / Sony driver")
        return

    csv_f, csv_w = None, None
    if save_csv_flag:
        csv_f = open(DETECT_CSV, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["ts","person","side","cx","cy","conf","kx","ky"])

    print(f"\n=== Foot Detection [{tag}] ===")
    print("  Q=quit  S=screenshot  +/-=threshold\n")

    conf_th   = CONF_THRESHOLD
    fps_buf   = []
    frame_id  = 0
    black_cnt = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.05); continue

        if frame.mean() < 2.0:
            black_cnt += 1
            if black_cnt % 10 == 0:
                print(f"[WARN] Black frame x{black_cnt} - check camera USB mode")
            frame_id += 1; continue
        black_cnt = 0

        res = model(frame, conf=conf_th, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)

        h, w   = frame.shape[:2]
        dw, dh = 1280, 720
        disp   = cv2.resize(frame, (dw, dh))
        sx     = dw / w
        sy     = dh / h
        feet   = 0

        if res and len(res) > 0:
            r = res[0]

            # ── pose mode ──────────────────────────────
            if use_pose and r.keypoints is not None:
                kpts = r.keypoints.data          # (N, 17, 3)

                for pid, pk in enumerate(kpts):
                    for side, aid, kid, col in [
                        ("L", ANKLE_L, KNEE_L, (0, 220, 110)),
                        ("R", ANKLE_R, KNEE_R, (30, 140, 255)),
                    ]:
                        ac = float(pk[aid][2])
                        if ac < conf_th: continue

                        ax = float(pk[aid][0]); ay = float(pk[aid][1])
                        kx = float(pk[kid][0]); ky = float(pk[kid][1])
                        dax = int(ax*sx); day = int(ay*sy)
                        dkx = int(kx*sx); dky = int(ky*sy)

                        feet += 1
                        cv2.circle(disp, (dax, day), 11, col, -1)
                        cv2.circle(disp, (dax, day), 16, col,  2)
                        if float(pk[kid][2]) > 0.3:
                            cv2.circle(disp, (dkx, dky), 6, col, 2)
                            cv2.line(disp, (dkx, dky), (dax, day), col, 2)
                        cv2.putText(disp, f"P{pid} {side} {ac:.2f}",
                                    (dax+14, day-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
                        if csv_w:
                            csv_w.writerow([f"{time.time():.3f}", pid, side,
                                            f"{ax:.1f}", f"{ay:.1f}", f"{ac:.3f}",
                                            f"{kx:.1f}", f"{ky:.1f}"])

                if r.boxes is not None:
                    for b in r.boxes:
                        x1,y1,x2,y2 = map(float, b.xyxy[0])
                        cv2.rectangle(disp,
                                      (int(x1*sx), int(y1*sy)),
                                      (int(x2*sx), int(y2*sy)),
                                      (50, 50, 50), 1)

            # ── custom model mode ───────────────────────
            elif not use_pose and r.boxes is not None:
                for i, b in enumerate(r.boxes):
                    cf = float(b.conf[0])
                    x1,y1,x2,y2 = map(float, b.xyxy[0])
                    cx,cy = (x1+x2)/2, (y1+y2)/2
                    dx1,dy1 = int(x1*sx), int(y1*sy)
                    dx2,dy2 = int(x2*sx), int(y2*sy)
                    feet += 1
                    cv2.rectangle(disp,(dx1,dy1),(dx2,dy2),(0,220,110),2)
                    cv2.circle(disp,(int(cx*sx),int(cy*sy)),5,(0,220,110),-1)
                    cv2.putText(disp,f"foot{i} {cf:.2f}",
                                (dx1,dy1-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,110),1)
                    if csv_w:
                        csv_w.writerow([f"{time.time():.3f}",i,"?",
                                        f"{cx:.1f}",f"{cy:.1f}",f"{cf:.3f}","",""])

        # FPS
        fps_buf.append(time.time()-t0)
        if len(fps_buf) > 30: fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf)/len(fps_buf))

        # HUD (English only)
        hc = (0,220,110) if feet > 0 else (70,70,70)
        cv2.rectangle(disp, (0,0),(dw,46),(0,0,0),-1)
        cv2.putText(disp,
                    f"Feet: {feet}  |  FPS: {fps:.1f}  "
                    f"|  Conf: {conf_th:.2f}  |  Frame #{frame_id}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, hc, 2)
        hint = ("Custom model" if not use_pose else
                "Pose mode  Green=L-Ankle  Blue=R-Ankle  "
                "|  Train custom model: --mode collect -> label -> train")
        cv2.putText(disp, hint, (10, dh-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80,80,120), 1)

        cv2.imshow("MIVR-CEIQ Foot Detection", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),ord('Q')): break
        elif key in (ord('s'),ord('S')):
            p = f"{DATA_DIR}/shot_{int(time.time())}.jpg"
            cv2.imwrite(p, frame); print(f"  [shot] {p}")
        elif key in (ord('+'),ord('=')):
            conf_th = min(conf_th+0.05, 0.95); print(f"  conf -> {conf_th:.2f}")
        elif key == ord('-'):
            conf_th = max(conf_th-0.05, 0.05); print(f"  conf -> {conf_th:.2f}")

        frame_id += 1

    cap.release(); cv2.destroyAllWindows()
    if csv_f: csv_f.close(); print(f"[CSV] {DETECT_CSV}")
    print("Done.")


# ─────────────────────────────────────────────────────
# Collect
# ─────────────────────────────────────────────────────

def collect_mode(camera_index: int):
    ensure_dirs()
    cap = open_camera(camera_index) if camera_index >= 0 else find_camera()
    if cap is None: print("[ERROR] No camera."); return

    count = len(list(Path(RAW_DIR).glob("*.jpg")))
    auto  = False; last = 0.0

    print("\n=== Data Collection ===  SPACE=save  A=auto(0.5s)  Q=quit")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue
        disp = cv2.resize(frame,(1280,720)); now = time.time()
        if auto and now-last >= 0.5:
            cv2.imwrite(f"{RAW_DIR}/frame_{count:04d}.jpg", frame)
            count += 1; last = now
        col = (0,80,255) if auto else (180,180,180)
        cv2.rectangle(disp,(0,0),(disp.shape[1],44),(0,0,0),-1)
        cv2.putText(disp,
                    f"{'[AUTO]' if auto else '[MANUAL]'}  {count}/300  "
                    "SPACE  A=auto  Q=quit",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
        cv2.rectangle(disp,(0,42),(int(disp.shape[1]*min(count/300,1)),44),(0,220,110),-1)
        cv2.imshow("MIVR-CEIQ Collect", disp)
        k = cv2.waitKey(30)&0xFF
        if k in (ord('q'),ord('Q')): break
        elif k == ord(' '):
            cv2.imwrite(f"{RAW_DIR}/frame_{count:04d}.jpg",frame)
            count += 1; print(f"  saved {count}")
        elif k in (ord('a'),ord('A')):
            auto = not auto; print(f"  auto {'ON' if auto else 'OFF'}")
    cap.release(); cv2.destroyAllWindows()
    print(f"Done: {count} images -> {RAW_DIR}/")
    print("Next: python foot_detector_v3.py --mode label")


# ─────────────────────────────────────────────────────
# Label
# ─────────────────────────────────────────────────────

def label_mode():
    ensure_dirs()
    imgs = sorted(Path(RAW_DIR).glob("*.jpg"))
    todo = [i for i in imgs if not (Path(LABEL_DIR)/(i.stem+".txt")).exists()]
    if not todo: print(f"All {len(imgs)} labeled."); return

    print(f"\n=== Labeler  {len(todo)}/{len(imgs)} remaining ===")
    print("  Drag box | ENTER=save | R=redo | S=skip | Q=quit\n")

    st = {"d":False,"s":(-1,-1),"e":(-1,-1),"bx":[],"img":None}

    def cb(ev,x,y,fl,_):
        if ev==cv2.EVENT_LBUTTONDOWN: st["d"]=True; st["s"]=st["e"]=(x,y)
        elif ev==cv2.EVENT_MOUSEMOVE and st["d"]:
            st["e"]=(x,y); tmp=st["img"].copy()
            cv2.rectangle(tmp,st["s"],st["e"],(0,220,110),2)
            cv2.imshow("MIVR-CEIQ Labeler",tmp)
        elif ev==cv2.EVENT_LBUTTONUP:
            st["d"]=False; st["e"]=(x,y)
            if abs(x-st["s"][0])>10 and abs(y-st["s"][1])>10:
                st["bx"].append((st["s"],st["e"]))

    cv2.namedWindow("MIVR-CEIQ Labeler")
    cv2.setMouseCallback("MIVR-CEIQ Labeler",cb)

    for img_path in todo:
        orig=cv2.imread(str(img_path)); oh,ow=orig.shape[:2]
        sc=min(1280/ow,720/oh)
        st["img"]=cv2.resize(orig,(int(ow*sc),int(oh*sc)))
        st["bx"]=[]; dh,dw=st["img"].shape[:2]
        while True:
            s=st["img"].copy()
            for a,b in st["bx"]: cv2.rectangle(s,a,b,(0,180,255),2)
            cv2.rectangle(s,(0,0),(dw,38),(0,0,0),-1)
            cv2.putText(s,f"{img_path.name} | {len(st['bx'])} boxes | "
                        "ENTER=save R=redo S=skip Q=quit",
                        (8,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            cv2.imshow("MIVR-CEIQ Labeler",s)
            k=cv2.waitKey(30)&0xFF
            if k==13:
                if st["bx"]:
                    lines=[]
                    for (x1d,y1d),(x2d,y2d) in st["bx"]:
                        x1=min(x1d,x2d)/sc; x2=max(x1d,x2d)/sc
                        y1=min(y1d,y2d)/sc; y2=max(y1d,y2d)/sc
                        cx=((x1+x2)/2)/ow; cy=((y1+y2)/2)/oh
                        bw=(x2-x1)/ow; bh=(y2-y1)/oh
                        cx,cy=max(0,min(1,cx)),max(0,min(1,cy))
                        bw,bh=max(.001,min(1,bw)),max(.001,min(1,bh))
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    with open(Path(LABEL_DIR)/(img_path.stem+".txt"),"w") as f:
                        f.write("\n".join(lines))
                    print(f"  saved {img_path.name} ({len(st['bx'])} boxes)")
                break
            elif k in (ord('r'),ord('R')): st["bx"]=[]
            elif k in (ord('s'),ord('S')): print(f"  skip"); break
            elif k in (ord('q'),ord('Q')): cv2.destroyAllWindows(); return

    cv2.destroyAllWindows()
    done=len(list(Path(LABEL_DIR).glob("*.txt")))
    print(f"\nDone: {done}/{len(imgs)}")
    if done>=50: print("Ready: python foot_detector_v3.py --mode train")


# ─────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────

def train_mode(epochs,batch):
    ensure_dirs()
    try: from ultralytics import YOLO
    except ImportError: print("[ERROR] pip install ultralytics"); return

    lbls=list(Path(LABEL_DIR).glob("*.txt"))
    print(f"Labels: {len(lbls)}")
    if len(lbls)<50:
        if input("< 50 labels. Continue? (y/N): ").lower()!='y': return

    paired=[(Path(RAW_DIR)/(l.stem+".jpg"),l)
            for l in lbls if (Path(RAW_DIR)/(l.stem+".jpg")).exists()]
    np.random.shuffle(paired); sp=int(len(paired)*.8)

    for d in [f"{DATA_DIR}/dataset/images/train",f"{DATA_DIR}/dataset/labels/train",
              f"{DATA_DIR}/dataset/images/val",  f"{DATA_DIR}/dataset/labels/val"]:
        Path(d).mkdir(parents=True,exist_ok=True)

    ti,tl=f"{DATA_DIR}/dataset/images/train",f"{DATA_DIR}/dataset/labels/train"
    vi,vl=f"{DATA_DIR}/dataset/images/val",  f"{DATA_DIR}/dataset/labels/val"
    for img,lbl in paired[:sp]: shutil.copy2(img,ti); shutil.copy2(lbl,tl)
    for img,lbl in paired[sp:]: shutil.copy2(img,vi); shutil.copy2(lbl,vl)

    yp=f"{DATA_DIR}/dataset/foot.yaml"
    Path(yp).write_text(
        f"path: {str(Path(DATA_DIR+'/dataset').resolve()).replace(chr(92),'/')}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames:\n  0: foot\n")

    YOLO("yolov8n.pt").train(
        data=yp,epochs=epochs,batch=batch,imgsz=IMG_SIZE,
        device=0,patience=15,project=MODEL_DIR,name="foot_detection",exist_ok=True,
        degrees=45.0,fliplr=0.5,flipud=0.5,scale=0.3)

    best=Path(MODEL_DIR)/"foot_detection"/"weights"/"best.pt"
    if best.exists():
        shutil.copy2(best,TRAINED_MODEL)
        print(f"[OK] Model: {TRAINED_MODEL}")
        print("Run: python foot_detector_v3.py --mode detect")


# ─────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser(description="MIVR-CEIQ Foot Detector v3")
    p.add_argument("--mode",choices=["detect","collect","label","train"],default="detect")
    p.add_argument("--camera",type=int,default=0,help="-1=auto scan")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch", type=int,default=16)
    p.add_argument("--no-csv",action="store_true")
    a=p.parse_args()
    print("="*50)
    print("  MIVR-CEIQ Foot Detector v3")
    print("="*50)
    if   a.mode=="detect":  detect_mode(a.camera,not a.no_csv)
    elif a.mode=="collect": collect_mode(a.camera)
    elif a.mode=="label":   label_mode()
    elif a.mode=="train":   train_mode(a.epochs,a.batch)

if __name__=="__main__":
    main()
