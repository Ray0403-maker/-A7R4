import cv2
import numpy as np
import os

# 已經確認 Sony A7R4 是裝置 0
DEVICE_ID = 1  

def main():
    # 使用預設後端讀取（移除 CAP_DSHOW 以避免部分虛擬驅動衝突）
    cap = cv2.VideoCapture(DEVICE_ID)  

    if not cap.isOpened():
        print("❌ 無法開啟相機，請確認沒有其他軟體（如 Remote）佔用相機。")
        return

    # 確認實際取得的解析度 (應為 1024 x 576)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Sony A7R4 已成功開啟！")
    print(f"   目前解析度：{actual_w} x {actual_h}")
    print(f"   FPS：{actual_fps}")
    print("   提示：按 Q 鍵退出系統 ｜ 按 S 鍵儲存相機校正截圖")

    frame_count = 0
    screenshot_count = 0

    # 確保校正資料夾存在
    os.makedirs("config/calibration", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  讀取影格失敗，正在重試...")
            continue

        frame_count += 1
        
        # 複製未疊加 HUD 文字的原始畫面（用於相機校正，避免文字遮擋棋盤格角落）
        raw_frame = frame.copy()

        # ── 畫面資訊疊加 (HUD) ────────────────────────
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Sony A7R4  {w}x{h}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Saved Calib Pics: {screenshot_count}",
                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Q:Quit  S:Capture Calib Pic",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        # 顯示畫面
        cv2.imshow("Sony A7R4 - OpenCV Live View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"config/calibration/calib_{screenshot_count:02d}.jpg"
            # 儲存無文字遮擋的原始圖片
            cv2.imwrite(filename, raw_frame)
            print(f"📸 原始校正截圖已儲存至：{filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("🔴 Live View 已結束")

if __name__ == "__main__":
    main()
