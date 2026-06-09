"""
MIVR-CEIQ — 腳部偵測（YOLOv8-Pose 版）
foot_detector.py  ← 修正版，移除 mediapipe.solutions

修正說明：
  MediaPipe 0.10+ 已移除 mp.solutions API
  改用 YOLOv8n-pose 偵測腳踝關節點（更穩定，不需額外安裝）

使用方式：
  # 即時腳部偵測（自動下載 yolov8n-pose.pt，約 6MB）
  python foot_detector.py --mode detect

  # 收集訓練數據
  python foot_detector.py --mode collect

  # 內建標注工具
  python foot_detector.py --mode label

  # 訓練自訂腳部模型
  python foot_detector.py --mode train --epochs 50
"""

import cv2
import numpy as np
import argparse
import time
import os
import csv
import shutil
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────
DATA_DIR      = "data/foot_detection"
RAW_DIR       = f"{DATA_DIR}/raw_images"
LABEL_DIR     = f"{DATA_DIR}/labels"
TRAIN_IMG_DIR = f"{DATA_DIR}/dataset/images/train"
VAL_IMG_DIR   = f"{DATA_DIR}/dataset/images/val"
TRAIN_LBL_DIR = f"{DATA_DIR}/dataset/labels/train"
VAL_LBL_DIR   = f"{DATA_DIR}/dataset/labels/val"
YAML_PATH     = f"{DATA_DIR}/dataset/foot.yaml"
MODEL_DIR     = "models"
TRAINED_MODEL = f"{MODEL_DIR}/foot_yolov8.pt"
DETECT_CSV    = "data/foot_detections.csv"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

# YOLOv8-Pose COCO 關節點索引
# 15=left_knee 16=right_knee 15,16 提供更穩定的下肢追蹤
ANKLE_L  = 15   # left_ankle
ANKLE_R  = 16   # right_ankle
KNEE_L   = 13   # left_knee   (備用，腳踝被遮時用膝蓋)
KNEE_R   = 14   # right_knee
# ─────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]


def ensure_dirs():
    for d in [RAW_DIR, LABEL_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR,
              TRAIN_LBL_DIR, VAL_LBL_DIR, MODEL_DIR,
              "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════
# MODE：即時偵測（YOLOv8-Pose）
# ══════════════════════════════════════════════════════

def detect_mode(camera_index: int, save_csv: bool = True):
    """
    用 YOLOv8n-pose 偵測腳踝關節點
    左腳踝 [15]、右腳踝 [16]
    信心值 > 0.3 才顯示
    """
    ensure_dirs()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 請先安裝：pip install ultralytics")
        return

    # 優先用自訓練模型，否則用 pose 模型
    if os.path.exists(TRAINED_MODEL):
        model = YOLO(TRAINED_MODEL)
        mode_name = "自訓練腳部模型"
        use_pose  = False
    else:
        model = YOLO("yolov8n-pose.pt")   # 第一次執行會自動下載
        mode_name = "YOLOv8n-Pose（腳踝關節點）"
        use_pose  = True
        print(f"[INFO] 使用 {mode_name}")
        print("  首次執行會自動下載 yolov8n-pose.pt（約 6MB）\n")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # DSHOW 失敗時嘗試預設後端
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟相機 index={camera_index}")
        print("  請確認 A7R IV 已連接且驅動安裝完成")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # CSV 輸出
    csv_f, csv_w = None, None
    if save_csv:
        csv_f = open(DETECT_CSV, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["timestamp", "person_id", "side",
                         "cx_px", "cy_px", "confidence",
                         "knee_cx", "knee_cy"])

    print(f"\n=== MIVR-CEIQ 腳部偵測 ({mode_name}) ===")
    print("  按 Q 結束  |  S 截圖  |  +/- 調整信心閾值")
    print(f"  信心閾值：{CONF_THRESHOLD}")
    print(f"  CSV 輸出：{DETECT_CSV if save_csv else '關閉'}\n")

    conf_th  = CONF_THRESHOLD
    fps_buf  = []
    frame_id = 0

    while True:
        t0  = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 讀取相機失敗，重試...")
            time.sleep(0.05)
            continue

        # ── YOLOv8 推理 ──
        results = model(frame,
                        conf    = conf_th,
                        iou     = IOU_THRESHOLD,
                        imgsz   = IMG_SIZE,
                        verbose = False)

        h, w = frame.shape[:2]
        display = cv2.resize(frame, (1280, 720))
        dh, dw  = display.shape[:2]
        sx, sy  = dw / w, dh / h

        foot_count  = 0
        foot_points = []   # [(person_id, side, cx, cy, conf)]

        if results and len(results) > 0:
            r = results[0]

            if use_pose and r.keypoints is not None:
                # ── Pose 模式：取腳踝關節點 ──
                kpts = r.keypoints.data   # shape: (N_persons, 17, 3) [x, y, conf]

                for person_id, person_kpts in enumerate(kpts):

                    for side, ank_id, kne_id, color in [
                        ("L", ANKLE_L, KNEE_L,  (0, 245, 180)),   # 左腳 綠
                        ("R", ANKLE_R, KNEE_R,  (255, 160, 0)),    # 右腳 橙
                    ]:
                        ank = person_kpts[ank_id]   # [x, y, conf]
                        kne = person_kpts[kne_id]

                        ank_conf = float(ank[2])
                        if ank_conf < conf_th:
                            continue

                        ax = float(ank[0]); ay = float(ank[1])
                        kx = float(kne[0]); ky = float(kne[1])

                        # 顯示座標
                        dax = int(ax * sx); day = int(ay * sy)
                        dkx = int(kx * sx); dky = int(ky * sy)

                        foot_points.append((person_id, side, ax, ay, ank_conf))
                        foot_count += 1

                        # 腳踝大圓
                        cv2.circle(display, (dax, day), 10, color, -1)
                        cv2.circle(display, (dax, day), 14, color, 2)

                        # 膝蓋小圓（提供下肢方向參考）
                        if float(kne[2]) > 0.3:
                            cv2.circle(display, (dkx, dky), 6, color, 2)
                            cv2.line(display, (dkx, dky), (dax, day), color, 2)

                        # 標籤
                        label = f"P{person_id} {side}-Ankle {ank_conf:.2f}"
                        cv2.putText(display, label,
                                    (dax + 12, day - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                        # CSV
                        if csv_w:
                            csv_w.writerow([
                                f"{time.time():.4f}",
                                person_id, side,
                                f"{ax:.1f}", f"{ay:.1f}",
                                f"{ank_conf:.3f}",
                                f"{kx:.1f}", f"{ky:.1f}"
                            ])

                # 人體框（淡色）
                if r.boxes is not None:
                    for box in r.boxes:
                        x1,y1,x2,y2 = map(float, box.xyxy[0])
                        cv2.rectangle(display,
                                      (int(x1*sx), int(y1*sy)),
                                      (int(x2*sx), int(y2*sy)),
                                      (60, 60, 60), 1)

            elif not use_pose and r.boxes is not None:
                # ── 自訓練腳部模型模式 ──
                for i, box in enumerate(r.boxes):
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(float, box.xyxy[0])
                    cx = (x1+x2)/2; cy = (y1+y2)/2

                    foot_points.append((i, "?", cx, cy, conf))
                    foot_count += 1

                    dx1=int(x1*sx); dy1=int(y1*sy)
                    dx2=int(x2*sx); dy2=int(y2*sy)
                    dcx=int(cx*sx); dcy=int(cy*sy)

                    cv2.rectangle(display,(dx1,dy1),(dx2,dy2),(0,245,180),2)
                    cv2.circle(display,(dcx,dcy),5,(0,245,180),-1)
                    cv2.putText(display, f"foot {i} {conf:.2f}",
                                (dx1, dy1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,245,180),1)

                    if csv_w:
                        csv_w.writerow([f"{time.time():.4f}", i, "?",
                                        f"{cx:.1f}", f"{cy:.1f}",
                                        f"{conf:.3f}", "", ""])

        # ── FPS 計算 ──
        fps_buf.append(time.time() - t0)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf))

        # ── HUD ──
        cv2.rectangle(display, (0, 0), (dw, 48), (0, 0, 0), -1)
        status_color = (0, 245, 180) if foot_count > 0 else (80, 80, 80)
        cv2.putText(display,
                    f"偵測到 {foot_count} 個腳踝點  "
                    f"| FPS {fps:.1f}  "
                    f"| 信心閾值 {conf_th:.2f}  "
                    f"| 幀 #{frame_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # 底部說明
        tip = ("自訓練模型" if not use_pose
               else "Pose模式：綠=左腳踝  橙=右腳踝  | 訓練專屬模型後精度更高")
        cv2.putText(display, tip,
                    (10, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 150), 1)

        cv2.imshow("MIVR-CEIQ — Foot Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            p = f"{DATA_DIR}/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(p, frame)
            print(f"  [SCREENSHOT] {p}")
        elif key in (ord('+'), ord('=')):
            conf_th = min(conf_th + 0.05, 0.95)
            print(f"  信心閾值 → {conf_th:.2f}")
        elif key in (ord('-'),):
            conf_th = max(conf_th - 0.05, 0.05)
            print(f"  信心閾值 → {conf_th:.2f}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    if csv_f:
        csv_f.close()
        print(f"\n偵測數據已儲存：{DETECT_CSV}")
    print("偵測結束")


# ══════════════════════════════════════════════════════
# MODE：數據收集
# ══════════════════════════════════════════════════════

def collect_mode(camera_index: int):
    ensure_dirs()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟相機 index={camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 30)

    count     = len(list(Path(RAW_DIR).glob("*.jpg")))
    auto_mode = False
    last_auto = 0

    print("\n=== 腳部訓練數據收集 ===")
    print("  SPACE → 手動擷取 | A → 自動連拍(0.5s) | Q → 結束")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = cv2.resize(frame, (1280, 720))
        now = time.time()

        if auto_mode and (now - last_auto) >= 0.5:
            cv2.imwrite(f"{RAW_DIR}/frame_{count:04d}.jpg", frame)
            count    += 1
            last_auto = now

        color = (0, 80, 255) if auto_mode else (180, 180, 180)
        cv2.rectangle(display, (0, 0), (display.shape[1], 44), (0,0,0), -1)
        cv2.putText(display,
                    f"{'🔴 自動' if auto_mode else '⚪ 手動'}  "
                    f"已收集：{count}/300 張  |  SPACE手動  A自動  Q結束",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        bar = int(display.shape[1] * min(count/300, 1.0))
        cv2.rectangle(display, (0, 42), (bar, 44), (0,245,180), -1)

        cv2.imshow("MIVR-CEIQ — Data Collection", display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            cv2.imwrite(f"{RAW_DIR}/frame_{count:04d}.jpg", frame)
            count += 1
            print(f"  [SAVED] frame_{count:04d}.jpg")
        elif key in (ord('a'), ord('A')):
            auto_mode = not auto_mode
            print(f"  自動連拍：{'開啟' if auto_mode else '關閉'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n完成：{count} 張圖片 → {RAW_DIR}/")
    print(f"下一步：python foot_detector.py --mode label")


# ══════════════════════════════════════════════════════
# MODE：內建標注工具
# ══════════════════════════════════════════════════════

def label_mode():
    ensure_dirs()
    images = sorted(Path(RAW_DIR).glob("*.jpg"))
    unlabeled = [i for i in images
                 if not (Path(LABEL_DIR) / (i.stem+".txt")).exists()]

    if not unlabeled:
        print(f"所有 {len(images)} 張已標注完成")
        return

    print(f"\n=== 快速標注工具 ===")
    print(f"  待標注：{len(unlabeled)}/{len(images)} 張")
    print("  拖曳畫框 | ENTER確認 | R重畫 | S跳過 | Q結束\n")

    state = {"drawing": False, "s": (-1,-1), "e": (-1,-1), "boxes": [], "img": None}

    def mouse_cb(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["s"] = state["e"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["e"] = (x, y)
            tmp = state["img"].copy()
            cv2.rectangle(tmp, state["s"], state["e"], (0,245,180), 2)
            cv2.imshow("MIVR-CEIQ — Labeler", tmp)
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["e"] = (x, y)
            if (abs(state["e"][0]-state["s"][0]) > 10 and
                abs(state["e"][1]-state["s"][1]) > 10):
                state["boxes"].append((state["s"], state["e"]))

    cv2.namedWindow("MIVR-CEIQ — Labeler")
    cv2.setMouseCallback("MIVR-CEIQ — Labeler", mouse_cb)

    for img_path in unlabeled:
        orig  = cv2.imread(str(img_path))
        h, w  = orig.shape[:2]
        scale = min(1280/w, 720/h)
        state["img"]   = cv2.resize(orig, (int(w*scale), int(h*scale)))
        state["boxes"] = []
        dh, dw = state["img"].shape[:2]

        while True:
            show = state["img"].copy()
            for s, e in state["boxes"]:
                cv2.rectangle(show, s, e, (0,200,255), 2)
            cv2.rectangle(show, (0,0), (dw, 40), (0,0,0), -1)
            cv2.putText(show,
                        f"{img_path.name} | {len(state['boxes'])}框 | "
                        "ENTER確認 R重畫 S跳過 Q結束",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("MIVR-CEIQ — Labeler", show)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # ENTER
                if state["boxes"]:
                    _save_label(img_path.stem, state["boxes"], w, h, scale)
                    print(f"  ✅ {img_path.name}  {len(state['boxes'])} 框")
                break
            elif key in (ord('r'), ord('R')):
                state["boxes"] = []
            elif key in (ord('s'), ord('S')):
                print(f"  ⏭ 跳過 {img_path.name}")
                break
            elif key in (ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    done = len(list(Path(LABEL_DIR).glob("*.txt")))
    print(f"\n標注完成：{done}/{len(images)} 張")
    if done >= 50:
        print(f"執行訓練：python foot_detector.py --mode train")


def _save_label(stem, boxes, orig_w, orig_h, scale):
    lines = []
    for (x1d, y1d), (x2d, y2d) in boxes:
        x1 = min(x1d, x2d) / scale; x2 = max(x1d, x2d) / scale
        y1 = min(y1d, y2d) / scale; y2 = max(y1d, y2d) / scale
        cx = ((x1+x2)/2) / orig_w
        cy = ((y1+y2)/2) / orig_h
        bw = (x2-x1) / orig_w
        bh = (y2-y1) / orig_h
        cx,cy = max(0,min(1,cx)), max(0,min(1,cy))
        bw,bh = max(0.001,min(1,bw)), max(0.001,min(1,bh))
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(Path(LABEL_DIR)/(stem+".txt"), "w") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════
# MODE：訓練
# ══════════════════════════════════════════════════════

def train_mode(epochs: int, batch: int):
    ensure_dirs()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        return

    raw_imgs = list(Path(RAW_DIR).glob("*.jpg"))
    lbls     = list(Path(LABEL_DIR).glob("*.txt"))
    print(f"圖片：{len(raw_imgs)} | 標注：{len(lbls)}")

    if len(lbls) < 50:
        print(f"[WARN] 標注不足 50，偵測效果可能較差")
        if input("繼續？(y/N): ").lower() != 'y':
            return

    paired = []
    for l in lbls:
        img = Path(RAW_DIR)/(l.stem+".jpg")
        if img.exists():
            paired.append((img, l))

    np.random.shuffle(paired)
    sp = int(len(paired)*0.8)

    for pairs, id_, il in [
        (paired[:sp], TRAIN_IMG_DIR, TRAIN_LBL_DIR),
        (paired[sp:], VAL_IMG_DIR,   VAL_LBL_DIR),
    ]:
        for img, lbl in pairs:
            shutil.copy2(img, id_); shutil.copy2(lbl, il)

    yaml = (f"path: {str(Path(DATA_DIR+'/dataset').resolve()).replace(chr(92),'/')}\n"
            f"train: images/train\nval: images/val\nnc: 1\nnames:\n  0: foot\n")
    Path(YAML_PATH).write_text(yaml)

    model = YOLO("yolov8n.pt")
    model.train(data=YAML_PATH, epochs=epochs, batch=batch,
                imgsz=IMG_SIZE, device=0, patience=15,
                project=MODEL_DIR, name="foot_detection", exist_ok=True,
                degrees=45.0, fliplr=0.5, flipud=0.5, scale=0.3)

    best = Path(MODEL_DIR)/"foot_detection"/"weights"/"best.pt"
    if best.exists():
        shutil.copy2(best, TRAINED_MODEL)
        print(f"\n✅ 訓練完成：{TRAINED_MODEL}")
        print(f"執行偵測：python foot_detector.py --mode detect")


# ══════════════════════════════════════════════════════
# 入口點
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MIVR-CEIQ 腳部偵測")
    parser.add_argument("--mode",   choices=["detect","collect","label","train"],
                        default="detect")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  MIVR-CEIQ Foot Detection System  [修正版]")
    print("=" * 55)

    if   args.mode == "detect":  detect_mode(args.camera, not args.no_csv)
    elif args.mode == "collect": collect_mode(args.camera)
    elif args.mode == "label":   label_mode()
    elif args.mode == "train":   train_mode(args.epochs, args.batch)


if __name__ == "__main__":
    main()
