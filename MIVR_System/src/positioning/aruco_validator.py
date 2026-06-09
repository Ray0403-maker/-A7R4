import cv2
import numpy as np
import argparse
import csv
import os

def load_calibration(calibration_file, width, height):
    """
    載入相機校正參數。如果檔案不存在，則使用預設估算參數。
    """
    if calibration_file and os.path.exists(calibration_file):
        try:
            import json
            with open(calibration_file, 'r') as f:
                calib = json.load(f)
            mtx = np.array(calib['camera_matrix'], dtype=np.float32)
            dist = np.array(calib['dist_coeffs'], dtype=np.float32)
            print(f"[OK] 成功載入相機校正檔案: {calibration_file}")
            print(f"   RMS 誤差: {calib.get('rms_error', 'N/A')}")
            return mtx, dist
        except Exception as e:
            print(f"[WARN] 載入校正檔案時發生錯誤: {e}，將使用預設參數。")
    
    # 預設參數 (使用解析度估算相機矩陣)
    print("[WARN] 未找到有效的相機校正檔。使用預設(Heuristic)相機參數進行估算。")
    focal_length = max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    mtx = np.array([[focal_length, 0, cx],
                    [0, focal_length, cy],
                    [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32) # 無畸變假設
    return mtx, dist

def estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """
    估算 ArUco Marker 的位姿，相容不同版本的 OpenCV。
    """
    if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        return rvecs, tvecs
    else:
        # solvePnP 備份方案
        obj_pts = np.array([
            [-marker_size/2,  marker_size/2, 0],
            [ marker_size/2,  marker_size/2, 0],
            [ marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
        rvecs, tvecs = [], []
        for c in corners:
            ret, rvec, tvec = cv2.solvePnP(obj_pts, c[0], camera_matrix, dist_coeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)
        return np.array(rvecs), np.array(tvecs)

def main():
    parser = argparse.ArgumentParser(description="ArUco 即時識別與靜態定位精度驗證")
    parser.add_argument("--camera", type=int, default=0, help="相機裝置索引 (預設 0)")
    parser.add_argument("--marker-size", type=float, default=0.08, help="ArUco Marker 實際尺寸 (公尺，預設 0.08)")
    parser.add_argument("--output-csv", type=str, default="data/static_accuracy_test.csv", help="輸出 CSV 路徑")
    parser.add_argument("--calibration", type=str, default="config/camera_calibration.json", help="相機校正檔路徑")
    args = parser.parse_args()

    # 開啟相機
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟相機，裝置 ID: {args.camera}")
        return

    # 取得相機解析度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 相機已開啟。解析度: {width}x{height}, FPS: {fps}")

    # 載入校正參數
    camera_matrix, dist_coeffs = load_calibration(args.calibration, width, height)

    # 建立輸出資料夾
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # 設定 ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # 支援 OpenCV 不同版本的 detector 宣告
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        def detect_markers(img):
            return detector.detectMarkers(img)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        def detect_markers(img):
            return cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # 用來紀錄每個 Marker 的歷史座標，計算靜態精度 (抖動/誤差)
    # 這裡我們用「所有量測點的平均值」作為靜態的地面真值 (Ground Truth)
    marker_history = {} # marker_id -> list of tvecs

    # 開啟 CSV 檔案寫入
    csv_file = open(args.output_csv, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'marker_id', 'x_m', 'y_m', 'z_m', 'error_cm'])

    print(f"[INFO] 數據將即時寫入: {args.output_csv}")
    print("[INFO] 請將 Marker 固定在相機視野內。")
    print("[INFO] 按 'Q' 鍵退出並儲存數據。")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 讀取影像失敗")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect_markers(gray)

        if ids is not None:
            # 畫出檢測到的 Marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 估算位姿
            rvecs, tvecs = estimate_marker_pose(corners, args.marker_size, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                tvec = tvecs[i].flatten() # [x, y, z] 以公尺為單位
                rvec = rvecs[i].flatten()

                # 畫出 3D 軸線
                if hasattr(cv2.drawFrameAxes, '__call__') or hasattr(cv2, 'drawFrameAxes'):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.marker_size * 0.5)
                elif hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, args.marker_size * 0.5)

                # 紀錄歷史軌跡以計算基準點 (Ground Truth)
                if marker_id not in marker_history:
                    marker_history[marker_id] = []
                marker_history[marker_id].append(tvec)

                # 計算基準點 (以目前為止的平均座標當作真值)
                history = np.array(marker_history[marker_id])
                ref_tvec = np.mean(history, axis=0)

                # 計算目前的抖動誤差 (目前位置與平均位置的歐幾里得距離，單位：公分)
                error_cm = np.linalg.norm(tvec - ref_tvec) * 100.0

                # 寫入 CSV
                csv_writer.writerow([frame_idx, marker_id, tvec[0], tvec[1], tvec[2], f"{error_cm:.4f}"])

                # 在畫面疊加資訊
                dist_m = np.linalg.norm(tvec)
                text = f"ID: {marker_id} Dist: {dist_m:.2f}m Jitter: {error_cm:.2f}cm"
                # 取得該 marker 邊角位置來顯示文字
                c = corners[i][0]
                text_pos = (int(c[0][0]), int(c[0][1]) - 10)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 疊加系統資訊
        cv2.putText(frame, f"Frame: {frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Q: Save & Exit", (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("MIVR-CEIQ ArUco Static Accuracy Validator", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("[INFO] 驗證結束，CSV 檔案已儲存。")

if __name__ == "__main__":
    main()
