"""
MIVR-CEIQ Phase 2 — 相機校正腳本
camera_calibration.py

功能：
  1. 互動式拍攝模式：按空白鍵即時從相機擷取棋盤格校正圖片
  2. 離線校正模式：讀取已有圖片計算校正矩陣
  3. 儲存結果至 config/camera_calibration.json

使用方式：
  # 互動式拍攝（推薦，按 SPACE 拍照，Q 結束並校正）
  python src/positioning/camera_calibration.py --mode capture --camera 0

  # 離線模式（已有圖片）
  python src/positioning/camera_calibration.py --mode offline

  # 驗證校正結果
  python src/positioning/camera_calibration.py --mode verify

環境：Windows / Python 3.14 / OpenCV
"""

import cv2
import numpy as np
import json
import os
import glob
import argparse
import time
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────
CHESSBOARD_SIZE = (9, 6)        # 內角點數量 (欄, 列)
SQUARE_SIZE_M   = 0.025         # 棋盤格實體邊長（公尺），列印後請量測修正
CALIB_DIR       = "config/calibration"
OUTPUT_JSON     = "config/camera_calibration.json"
MIN_IMAGES      = 5
RECOMMENDED     = 10
# ─────────────────────────────────────────────────────


def ensure_dirs():
    Path(CALIB_DIR).mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(parents=True, exist_ok=True)


def capture_mode(camera_index: int):
    """
    互動式拍攝模式
    SPACE → 擷取並儲存
    R     → 刪除最後一張
    Q     → 結束並執行校正
    """
    ensure_dirs()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟相機 index={camera_index}")
        print("  請確認 A7R IV 已透過 USB 連接，並安裝 Imaging Edge 或 Direct Show 驅動")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    saved_paths = []
    frame_count = 0

    print("\n=== 相機校正拍攝模式 ===")
    print(f"  棋盤格規格：{CHESSBOARD_SIZE[0]}×{CHESSBOARD_SIZE[1]} 內角點")
    print(f"  目標張數：{RECOMMENDED} 張（最少 {MIN_IMAGES} 張）")
    print("  操作：")
    print("    SPACE → 擷取目前畫面（偵測到棋盤格才會儲存）")
    print("    R     → 刪除最後一張")
    print("    Q     → 完成拍攝並執行校正\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取相機畫面，重試中...")
            time.sleep(0.1)
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 即時偵測棋盤格（加速用較低精度）
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
            status_color = (0, 255, 0)
            status_text  = f"棋盤格已偵測到！按 SPACE 擷取 [{len(saved_paths)}/{RECOMMENDED}]"
        else:
            status_color = (0, 100, 255)
            status_text  = f"尋找棋盤格中... [{len(saved_paths)}/{RECOMMENDED}]"

        # HUD 覆蓋
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("MIVR-CEIQ Camera Calibration — Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord(' ') and found:
            # 高精度角點細化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            fname = os.path.join(CALIB_DIR, f"calib_{len(saved_paths):03d}.jpg")
            cv2.imwrite(fname, frame)
            saved_paths.append(fname)
            print(f"  [SAVED] {fname}  ({len(saved_paths)}/{RECOMMENDED})")

            # 閃爍回饋
            flash = frame.copy()
            cv2.rectangle(flash, (0,0), (flash.shape[1], flash.shape[0]), (255,255,255), 20)
            cv2.imshow("MIVR-CEIQ Camera Calibration — Capture", flash)
            cv2.waitKey(150)

        elif key == ord('r') or key == ord('R'):
            if saved_paths:
                removed = saved_paths.pop()
                os.remove(removed)
                print(f"  [REMOVED] {removed}")

    cap.release()
    cv2.destroyAllWindows()

    if len(saved_paths) < MIN_IMAGES:
        print(f"\n[WARN] 僅有 {len(saved_paths)} 張，建議至少 {MIN_IMAGES} 張才能校正")
        return False

    print(f"\n已擷取 {len(saved_paths)} 張，開始執行校正...")
    return run_calibration()


def run_calibration() -> bool:
    """從 config/calibration/ 目錄讀取圖片執行校正"""
    ensure_dirs()

    images = (glob.glob(os.path.join(CALIB_DIR, "*.jpg")) +
              glob.glob(os.path.join(CALIB_DIR, "*.png")))

    if len(images) < MIN_IMAGES:
        print(f"[ERROR] config/calibration/ 中僅有 {len(images)} 張圖片（需 >= {MIN_IMAGES}）")
        print(f"  請先執行：python src/positioning/camera_calibration.py --mode capture")
        return False

    print(f"\n=== 執行相機校正 ({len(images)} 張圖片) ===")

    # 3D 世界座標（棋盤格角點）
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                            0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_M

    obj_pts, img_pts = [], []
    valid_count = 0
    img_shape   = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, fname in enumerate(sorted(images)):
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners_refined)
            valid_count += 1
            print(f"  [OK] [{i+1:02d}/{len(images)}] {os.path.basename(fname)}")
        else:
            print(f"  [FAIL] [{i+1:02d}/{len(images)}] {os.path.basename(fname)}（未偵測到棋盤格，跳過）")

    if valid_count < MIN_IMAGES:
        print(f"\n[ERROR] 有效圖片僅 {valid_count} 張，校正失敗")
        return False

    print(f"\n有效圖片：{valid_count}/{len(images)}，計算校正矩陣...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_shape, None, None
    )

    # 計算重投影誤差
    total_error = 0.0
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        total_error += cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
    mean_reproj = total_error / len(obj_pts)

    result = {
        "camera_matrix":    camera_matrix.tolist(),
        "dist_coeffs":      dist_coeffs.tolist(),
        "image_size":       list(img_shape),
        "square_size_m":    SQUARE_SIZE_M,
        "chessboard_size":  list(CHESSBOARD_SIZE),
        "valid_images":     valid_count,
        "rms_error":        round(rms, 6),
        "mean_reproj_error":round(mean_reproj, 6),
        "pass":             rms < 1.0
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== 校正結果 ===")
    print(f"  RMS 誤差          : {rms:.4f} px（目標 < 1.0）")
    print(f"  平均重投影誤差    : {mean_reproj:.4f} px")
    print(f"  焦距 fx           : {camera_matrix[0][0]:.1f} px")
    print(f"  焦距 fy           : {camera_matrix[1][1]:.1f} px")
    print(f"  主點 cx, cy       : {camera_matrix[0][2]:.1f}, {camera_matrix[1][2]:.1f}")
    print(f"  失真係數 k1       : {dist_coeffs[0][0]:.6f}")
    print(f"  Pass / Fail       : {'[PASS] — 已儲存至 ' + OUTPUT_JSON if rms < 1.0 else '[WARN] RMS > 1.0，建議重新拍攝更多角度圖片'}")

    return rms < 1.0


def verify_mode(camera_index: int):
    """驗證校正結果：即時顯示去畸變效果"""
    if not os.path.exists(OUTPUT_JSON):
        print(f"[ERROR] 找不到 {OUTPUT_JSON}，請先執行校正")
        return

    with open(OUTPUT_JSON) as f:
        calib = json.load(f)

    mtx  = np.array(calib["camera_matrix"])
    dist = np.array(calib["dist_coeffs"])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟相機 index={camera_index}")
        return

    print("\n=== 校正驗證模式（按 Q 結束）===")
    print(f"  RMS 誤差: {calib['rms_error']} | 去畸變前後對比")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted   = cv2.undistort(frame, mtx, dist, None, new_mtx)

        # 左右對比顯示
        combined = np.hstack([frame, undistorted])
        scale    = 900 / combined.shape[1]
        combined = cv2.resize(combined, (0,0), fx=scale, fy=scale)

        mid = combined.shape[1] // 2
        cv2.putText(combined, "原始",      (10, 28),    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,200,255), 2)
        cv2.putText(combined, "去畸變後",  (mid+10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,100), 2)
        cv2.line(combined, (mid, 0), (mid, combined.shape[0]), (255,255,255), 1)

        cv2.imshow("MIVR-CEIQ — Calibration Verify", combined)
        if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MIVR-CEIQ 相機校正工具")
    parser.add_argument("--mode",   choices=["capture", "offline", "verify"],
                        default="capture", help="執行模式")
    parser.add_argument("--camera", type=int, default=0,
                        help="相機 index（預設 0）")
    args = parser.parse_args()

    if args.mode == "capture":
        success = capture_mode(args.camera)
    elif args.mode == "offline":
        success = run_calibration()
    elif args.mode == "verify":
        verify_mode(args.camera)
        success = True

    if args.mode != "verify":
        print("\n" + ("[PASS] Phase 2 相機校正完成，可繼續 Phase 3" if success
                      else "[FAIL] 校正未通過，請重新拍攝"))


if __name__ == "__main__":
    main()
