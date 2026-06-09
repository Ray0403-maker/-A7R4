"""
MIVR-CEIQ Optimized Adaptive Kalman Filter v2
adaptive_kalman_v2.py

專家十五、十六審查後的改進：
  [E16-Fix1] 距離自適應量測噪音 r(y)
  [E16-Fix2] 真實 dt 驅動 transitionMatrix
  [E16-Fix3] 互補濾波前處理（去高頻抖動）
  [E15-Fix1] 速度依賴的 innovation_gain
"""

import numpy as np
import cv2
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AKFConfig:
    # 過程噪音
    q_base:            float = 0.5
    q_max:             float = 800.0

    # 量測噪音基準（像素²，會被距離縮放）
    r_noise_base:      float = 6.0
    # [E16-Fix1] 距離自適應：r = r_base * (1 + dist_scale * norm_y²)
    r_dist_scale:      float = 4.0

    # 新息自適應
    innovation_window: int   = 5
    innovation_gain:   float = 5.0

    # 速度依賴增益（快速移動時更激進）
    # [E15-Fix1] gain 隨速度非線性增大
    speed_gain_exp:    float = 1.4

    # 速度衰減
    velocity_decay:    float = 0.78

    # 靜止判定
    still_threshold:   float = 2.0
    still_damping:     float = 0.04

    # 異常值閘門
    outlier_gate:      float = 100.0

    # 最大消失幀數
    max_missing:       int   = 12

    # [E16-Fix3] 互補濾波截止頻率（Hz）
    comp_filter_cutoff: float = 4.0


class ComplementaryFilter1D:
    """
    [E16-Fix3] 一維互補濾波器
    低通：保留真實運動（<cutoff Hz）
    高通截止：消除 YOLOv8 輸出的高頻抖動
    alpha = dt / (dt + 1/(2π*fc))
    """

    def __init__(self, cutoff_hz: float = 4.0):
        self.cutoff = cutoff_hz
        self._prev_filtered: Optional[float] = None
        self._prev_raw:      Optional[float] = None

    def filter(self, raw: float, dt: float) -> float:
        if dt <= 0:
            dt = 1.0 / 30.0
        tau   = 1.0 / (2.0 * np.pi * self.cutoff)
        alpha = dt / (dt + tau)    # 低通係數

        if self._prev_filtered is None:
            self._prev_filtered = raw
            self._prev_raw      = raw
            return raw

        # 低通濾波
        filtered = alpha * raw + (1.0 - alpha) * self._prev_filtered

        self._prev_filtered = filtered
        self._prev_raw      = raw
        return filtered

    def reset(self):
        self._prev_filtered = None
        self._prev_raw      = None


class AdaptiveKalmanPoint:
    """
    2D 自適應卡爾曼濾波器 v2
    狀態：[x, y, vx, vy]
    改進：真實 dt、距離自適應 r、互補前濾波
    """

    def __init__(self, cfg: AKFConfig = None):
        self.cfg  = cfg or AKFConfig()
        self._kf  = None
        self.initialized   = False
        self.missing_frames = 0

        # 新息歷史
        self._innov_buf: deque = deque(maxlen=self.cfg.innovation_window)
        self._last_smooth: Optional[Tuple[float, float]] = None

        # [E16-Fix2] 時間追蹤
        self._last_time: Optional[float] = None

        # [E16-Fix3] 互補濾波器（x, y 各一個）
        self._cf_x = ComplementaryFilter1D(self.cfg.comp_filter_cutoff)
        self._cf_y = ComplementaryFilter1D(self.cfg.comp_filter_cutoff)

        # 診斷資訊
        self.current_q   = self.cfg.q_base
        self.current_r   = self.cfg.r_noise_base
        self.current_dt  = 1.0 / 30.0
        self.is_still    = True
        self.was_outlier = False

    # ── KF 初始化 ────────────────────────────────────

    def _init_kf(self, x: float, y: float, dt: float):
        kf = cv2.KalmanFilter(4, 2)
        self._update_transition(kf, dt)
        kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        r = np.float32(self.cfg.r_noise_base)
        kf.measurementNoiseCov = np.array(
            [[r,0],[0,r]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * np.float32(self.cfg.q_base)
        kf.errorCovPost    = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array([[x],[y],[0.0],[0.0]], dtype=np.float32)
        kf.statePre  = kf.statePost.copy()
        return kf

    @staticmethod
    def _update_transition(kf: cv2.KalmanFilter, dt: float):
        """[E16-Fix2] 用真實 dt 更新狀態轉移矩陣"""
        dt = np.float32(max(dt, 1e-4))
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

    # ── 距離自適應 r ─────────────────────────────────

    def _compute_r(self, norm_y: float) -> float:
        """
        [E16-Fix1] 根據腳踝在畫面中的垂直位置調整量測噪音。
        norm_y = ankle_y / frame_height  (0=頂部, 1=底部)
        底部（近）→ r 小；頂部（遠）→ r 大
        """
        # 遠端誤差放大：r_dist_scale 倍（線性插值）
        dist_factor = 1.0 + self.cfg.r_dist_scale * (1.0 - norm_y) ** 2
        r = self.cfg.r_noise_base * dist_factor
        self.current_r = r
        return r

    # ── 自適應 Q ─────────────────────────────────────

    def _compute_q(self, raw_x: float, raw_y: float) -> float:
        if self._last_smooth is None:
            return self.cfg.q_base

        dx   = raw_x - self._last_smooth[0]
        dy   = raw_y - self._last_smooth[1]
        dist = float(np.sqrt(dx*dx + dy*dy))
        self._innov_buf.append(dist)

        # 靜止判定：即時距離
        self.is_still = dist < self.cfg.still_threshold

        if self.is_still:
            q = self.cfg.q_base * self.cfg.still_damping
        else:
            # [E15-Fix1] 速度依賴的非線性增益
            max_innov    = max(self._innov_buf)
            effective    = max(dist, max_innov * 0.5)
            speed_factor = (effective * self.cfg.innovation_gain) ** self.cfg.speed_gain_exp
            q = self.cfg.q_base + speed_factor
            q = min(q, self.cfg.q_max)

        self.current_q = q
        return q

    # ── 異常值閘門 ───────────────────────────────────

    def _is_outlier(self, x: float, y: float) -> bool:
        if not self.initialized or self._last_smooth is None:
            return False
        dx = x - self._last_smooth[0]
        dy = y - self._last_smooth[1]
        return float(np.sqrt(dx*dx + dy*dy)) > self.cfg.outlier_gate

    # ── 主更新 ───────────────────────────────────────

    def update(self, raw_x: float, raw_y: float,
               norm_y: float = 0.5) -> Tuple[float, float]:
        """
        norm_y: ankle_y / frame_height，用於距離自適應噪音
        """
        now = time.time()

        # 計算真實 dt
        if self._last_time is not None:
            dt = float(np.clip(now - self._last_time, 1e-4, 0.5))
        else:
            dt = 1.0 / 30.0
        self._last_time   = now
        self.current_dt   = dt

        # [E16-Fix3] 互補前濾波
        fx = self._cf_x.filter(raw_x, dt)
        fy = self._cf_y.filter(raw_y, dt)

        # 第一次初始化
        if not self.initialized:
            self._kf = self._init_kf(fx, fy, dt)
            self.initialized    = True
            self._last_smooth   = (fx, fy)
            self.missing_frames = 0
            return fx, fy

        # 異常值閘門（對原始值做判斷，不對濾波後做）
        if self._is_outlier(raw_x, raw_y):
            self.was_outlier = True
            self._update_transition(self._kf, dt)
            pred = self._kf.predict()
            sx, sy = float(pred[0][0]), float(pred[1][0])
            return sx, sy
        self.was_outlier = False

        # 計算自適應參數
        q = self._compute_q(fx, fy)
        r = self._compute_r(norm_y)

        # 更新 KF 矩陣
        self._update_transition(self._kf, dt)
        self._kf.processNoiseCov  = np.eye(4, dtype=np.float32) * np.float32(q)
        self._kf.measurementNoiseCov = np.array(
            [[r, 0],[0, r]], dtype=np.float32)

        # 速度衰減（靜止時）
        if self.is_still:
            decay = np.float32(self.cfg.velocity_decay * self.cfg.still_damping * 8)
            self._kf.statePost[2][0] = float(
                np.float32(self._kf.statePost[2][0]) * min(decay, np.float32(self.cfg.velocity_decay))
            )
            self._kf.statePost[3][0] = float(
                np.float32(self._kf.statePost[3][0]) * min(decay, np.float32(self.cfg.velocity_decay))
            )

        # Kalman predict + correct
        self._kf.predict()
        meas     = np.array([[fx],[fy]], dtype=np.float32)
        smoothed = self._kf.correct(meas)

        sx = float(smoothed[0][0])
        sy = float(smoothed[1][0])
        self._last_smooth   = (sx, sy)
        self.missing_frames = 0
        return sx, sy

    # ── 預測（遮擋）─────────────────────────────────

    def predict_only(self) -> Tuple[float, float]:
        self.missing_frames += 1
        if not self.initialized:
            return 0.0, 0.0
        now = time.time()
        dt  = float(np.clip(now - self._last_time, 1e-4, 0.2)) if self._last_time else 1/30
        self._last_time = now
        self._update_transition(self._kf, dt)
        # 速度衰減
        d = np.float32(self.cfg.velocity_decay)
        self._kf.statePost[2][0] = float(np.float32(self._kf.statePost[2][0]) * d)
        self._kf.statePost[3][0] = float(np.float32(self._kf.statePost[3][0]) * d)
        pred = self._kf.predict()
        sx, sy = float(pred[0][0]), float(pred[1][0])
        self._last_smooth = (sx, sy)
        return sx, sy

    def reset(self):
        self._kf = None
        self.initialized    = False
        self.missing_frames = 0
        self._innov_buf.clear()
        self._last_smooth   = None
        self._last_time     = None
        self._cf_x.reset()
        self._cf_y.reset()
        self.is_still    = True
        self.was_outlier = False

    @property
    def alive(self) -> bool:
        return self.missing_frames < self.cfg.max_missing

    @property
    def state_info(self) -> str:
        s = "still" if self.is_still else "moving"
        if self.was_outlier: s = "outlier"
        if not self.alive:   s = "lost"
        return f"Q={self.current_q:.0f} R={self.current_r:.1f} dt={self.current_dt*1000:.0f}ms [{s}]"


# ─────────────────────────────────────────────────────
# 多目標管理器
# ─────────────────────────────────────────────────────

class MultiFootTracker:
    def __init__(self, cfg: AKFConfig = None):
        self.cfg = cfg or AKFConfig()
        self._trackers: dict = {}

    def update(self, person_id: int, side: str,
               raw_x: float, raw_y: float,
               norm_y: float = 0.5) -> Tuple[float, float]:
        key = (person_id, side)
        if key not in self._trackers:
            self._trackers[key] = AdaptiveKalmanPoint(self.cfg)
        return self._trackers[key].update(raw_x, raw_y, norm_y)

    def predict_missing(self, seen: set):
        dead = []
        for key, t in self._trackers.items():
            if key not in seen:
                t.predict_only()
                if not t.alive:
                    dead.append(key)
        for k in dead:
            del self._trackers[k]

    def get_info(self, person_id: int, side: str) -> str:
        t = self._trackers.get((person_id, side))
        return t.state_info if t else ""

    def reset_all(self):
        self._trackers.clear()


# ─────────────────────────────────────────────────────
# 單元測試
# ─────────────────────────────────────────────────────

def _run_tests():
    print("=== AdaptiveKalmanPoint v2 Unit Tests ===\n")
    cfg = AKFConfig()

    # Test 1: 靜止收斂
    print("Test 1: 靜止收斂")
    kp = AdaptiveKalmanPoint(cfg)
    rng = np.random.default_rng(42)
    for _ in range(40):
        nx = 100.0 + rng.normal(0, 3)
        ny = 200.0 + rng.normal(0, 3)
        sx, sy = kp.update(nx, ny, norm_y=0.7)
        time.sleep(0.001)
    err = np.sqrt((sx-100)**2 + (sy-200)**2)
    print(f"  smooth=({sx:.2f},{sy:.2f})  err={err:.2f}px  Q={kp.current_q:.3f}")
    print(f"  {'✅ PASS' if err < 5 else '❌ FAIL'}\n")

    # Test 2: 移動響應（每幀 20px）
    print("Test 2: 連續移動響應")
    kp2 = AdaptiveKalmanPoint(cfg)
    for _ in range(25):
        kp2.update(100.0, 200.0, 0.7); time.sleep(0.033)
    tx = 100.0
    for i in range(10):
        tx += 20.0
        sx, sy = kp2.update(tx, 200.0, 0.7); time.sleep(0.033)
        lag = abs(sx - tx)
        print(f"  frame{i+1:02d}: target={tx:.0f} smooth={sx:.1f} lag={lag:.1f}px Q={kp2.current_q:.0f}")
    print(f"  {'✅ PASS' if lag < 40 else '❌ FAIL'}\n")

    # Test 3: 距離自適應噪音
    print("Test 3: 距離自適應 R（分兩個獨立 tracker）")
    # tracker A：在畫面底部（近端）
    kp_near = AdaptiveKalmanPoint(cfg)
    for _ in range(5):
        kp_near.update(500.0, 580.0, norm_y=0.9); time.sleep(0.033)
    r_near = kp_near.current_r

    # tracker B：在畫面頂部（遠端）
    kp_far = AdaptiveKalmanPoint(cfg)
    for _ in range(5):
        kp_far.update(500.0, 80.0, norm_y=0.1); time.sleep(0.033)
    r_far = kp_far.current_r

    print(f"  R_near(y=0.9)={r_near:.2f}  R_far(y=0.1)={r_far:.2f}  ratio={r_far/r_near:.2f}x")
    print(f"  {'✅ PASS' if r_far > r_near * 1.5 else '❌ FAIL'}\n")

    # Test 4: 真實 dt 驅動
    print("Test 4: 真實 dt（模擬 FPS 波動）")
    kp4 = AdaptiveKalmanPoint(cfg)
    kp4.update(200.0, 300.0, 0.5); time.sleep(0.033)
    kp4.update(200.0, 300.0, 0.5); time.sleep(0.100)  # 模擬卡頓
    kp4.update(220.0, 300.0, 0.5)
    print(f"  dt_recorded={kp4.current_dt*1000:.0f}ms（上幀卡頓後應 >30ms）")
    print(f"  {'✅ PASS' if kp4.current_dt > 0.03 else '❌ FAIL'}\n")

    # Test 5: 異常值閘門
    print("Test 5: 異常值閘門")
    kp5 = AdaptiveKalmanPoint(cfg)
    for _ in range(20):
        kp5.update(100.0, 100.0, 0.5); time.sleep(0.001)
    bx, by = kp5._last_smooth
    kp5.update(9999.0, 9999.0, 0.5)
    drift = np.sqrt((kp5._last_smooth[0]-bx)**2 + (kp5._last_smooth[1]-by)**2)
    print(f"  outlier 後飄移={drift:.2f}px  outlier_flag={kp5.was_outlier}")
    print(f"  {'✅ PASS' if drift < 15 else '❌ FAIL'}\n")

    # Test 6: 遮擋預測不飄移
    print("Test 6: 遮擋預測")
    kp6 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp6.update(200.0, 150.0, 0.6); time.sleep(0.001)
    bx, by = kp6._last_smooth
    for _ in range(10):
        kp6.predict_only(); time.sleep(0.033)
    drift = np.sqrt((kp6._last_smooth[0]-bx)**2 + (kp6._last_smooth[1]-by)**2)
    print(f"  10幀預測後飄移={drift:.2f}px")
    print(f"  {'✅ PASS' if drift < 15 else '❌ FAIL'}\n")

    print("=== 所有測試完成 ===")


if __name__ == "__main__":
    _run_tests()
