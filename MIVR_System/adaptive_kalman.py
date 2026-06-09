"""
MIVR-CEIQ Adaptive Kalman Filter for Foot Tracking
adaptive_kalman.py

設計目標：
  1. 靜止時：消除抖動，點完全穩定
  2. 移動時：幾乎零延遲跟上，不拖尾
  3. 不飄移：持續修正，長時間使用仍準確

核心機制：
  - Innovation-based Q adaptation（新息自適應過程噪音）
    偵測到大位移 → 升高 Q → Kalman 更相信量測值 → 快速跟上
    偵測到小位移 → 降低 Q → Kalman 更相信模型   → 消除抖動
  - Velocity decay（速度衰減）
    靜止時速度自動歸零，防止慣性導致飄移
  - Outlier gate（異常值閘門）
    極端跳點（遮擋誤偵測）直接忽略，不讓 Kalman 被污染
  - Re-init on long absence（長時間消失重置）
    人離開視野後回來，直接重初始化，不繼承舊狀態
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# 可調參數
# ─────────────────────────────────────────────────────

@dataclass
class AKFConfig:
    # 過程噪音基準（靜止時）
    q_base: float = 0.5

    # 過程噪音上限（高速移動時）
    q_max: float = 800.0

    # 量測噪音（YOLOv8 關節點誤差，像素²）
    r_noise: float = 8.0

    # 新息視窗（用來計算移動量的幀數）
    innovation_window: int = 5

    # 新息放大係數（移動速度 → Q 的增益）
    innovation_gain: float = 6.0

    # 速度衰減係數（0.0=立刻停, 1.0=無衰減）
    velocity_decay: float = 0.75

    # 靜止判定閾值（像素/幀，低於此認為靜止）
    still_threshold: float = 2.5

    # 靜止時 Q 額外壓低倍率
    still_damping: float = 0.05

    # 異常值閘門（像素，超過此距離的量測直接跳過）
    outlier_gate: float = 120.0

    # 最大允許消失幀數（超過則重初始化）
    max_missing: int = 12


# ─────────────────────────────────────────────────────
# 自適應卡爾曼濾波器
# ─────────────────────────────────────────────────────

class AdaptiveKalmanPoint:
    """
    2D 自適應卡爾曼濾波器
    狀態向量：[x, y, vx, vy]
    量測向量：[x, y]
    """

    def __init__(self, cfg: AKFConfig = None):
        self.cfg = cfg or AKFConfig()
        self._kf  = None
        self.initialized = False
        self.missing_frames = 0

        # 新息歷史（最近 N 幀的量測殘差大小）
        self._innovation_buf: deque = deque(maxlen=self.cfg.innovation_window)

        # 上一幀平滑後的位置（用於速度估算）
        self._last_smooth: Optional[Tuple[float, float]] = None

        # 統計（除錯用）
        self.current_q   = self.cfg.q_base
        self.is_still    = True
        self.was_outlier = False

    # ── 初始化 KalmanFilter ──────────────────────────

    def _init_kf(self, x: float, y: float):
        kf = cv2.KalmanFilter(4, 2)   # 4 狀態，2 量測

        # 狀態轉移矩陣（等速模型）
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # 量測矩陣（只觀測 x, y）
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # 量測噪音協方差（固定）
        r = float(self.cfg.r_noise)
        kf.measurementNoiseCov = np.array([[r, 0],[0, r]], dtype=np.float32)

        # 過程噪音（動態，先設初始值）
        q = float(self.cfg.q_base)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * q

        # 後驗誤差協方差
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        # 初始狀態（位置已知，速度為零）
        kf.statePost = np.array(
            [[x], [y], [0.0], [0.0]], dtype=np.float32
        )
        kf.statePre = kf.statePost.copy()

        return kf

    # ── 計算自適應 Q ─────────────────────────────────

    def _compute_adaptive_q(self, raw_x: float, raw_y: float) -> float:
        """
        根據最近幾幀的量測新息（innovation）動態調整過程噪音 Q。
        新息大 → 正在移動 → Q 升高 → 更信任量測值
        新息小 → 靜止中   → Q 降低 → 更強力平滑

        關鍵：用單幀距離判斷「是否剛開始移動」，
        確保加速初期立即響應，不等視窗填滿。
        """
        if self._last_smooth is None:
            return self.cfg.q_base

        dx = raw_x - self._last_smooth[0]
        dy = raw_y - self._last_smooth[1]
        dist = np.sqrt(dx*dx + dy*dy)

        self._innovation_buf.append(dist)
        mean_innov = np.mean(self._innovation_buf)

        # 靜止判定：用當前幀距離（即時），而非視窗均值
        # 這樣一移動馬上脫離靜止模式
        self.is_still = dist < self.cfg.still_threshold

        if self.is_still:
            q = self.cfg.q_base * self.cfg.still_damping
        else:
            # 取當前幀距離（即時響應）與視窗最大值（持續移動）中的較大值
            max_innov = max(self._innovation_buf)
            effective  = max(dist, max_innov * 0.5)
            speed_factor = effective * self.cfg.innovation_gain
            q = self.cfg.q_base + speed_factor ** 1.5
            q = min(q, self.cfg.q_max)

        self.current_q = q
        return q

    # ── 異常值閘門 ───────────────────────────────────

    def _is_outlier(self, raw_x: float, raw_y: float) -> bool:
        if not self.initialized or self._last_smooth is None:
            return False
        dx = raw_x - self._last_smooth[0]
        dy = raw_y - self._last_smooth[1]
        dist = np.sqrt(dx*dx + dy*dy)
        return dist > self.cfg.outlier_gate

    # ── 更新（有量測值）─────────────────────────────

    def update(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        # 第一次初始化
        if not self.initialized:
            self._kf = self._init_kf(raw_x, raw_y)
            self.initialized = True
            self._last_smooth = (raw_x, raw_y)
            self.missing_frames = 0
            return raw_x, raw_y

        # 異常值閘門
        if self._is_outlier(raw_x, raw_y):
            self.was_outlier = True
            # 跳過此量測，僅做預測
            pred = self._kf.predict()
            sx, sy = float(pred[0][0]), float(pred[1][0])
            return sx, sy
        self.was_outlier = False

        # 計算自適應 Q
        q = self._compute_adaptive_q(raw_x, raw_y)

        # 更新過程噪音
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * np.float32(q)

        # 速度衰減（靜止時讓速度歸零，防飄移）
        if self.is_still:
            decay = self.cfg.velocity_decay * self.cfg.still_damping * 10
            self._kf.statePost[2][0] *= min(decay, self.cfg.velocity_decay)
            self._kf.statePost[3][0] *= min(decay, self.cfg.velocity_decay)

        # Kalman predict + correct
        self._kf.predict()
        meas = np.array([[raw_x], [raw_y]], dtype=np.float32)
        smoothed = self._kf.correct(meas)

        sx = float(smoothed[0][0])
        sy = float(smoothed[1][0])

        self._last_smooth = (sx, sy)
        self.missing_frames = 0
        return sx, sy

    # ── 預測（無量測值，目標短暫遮擋）───────────────

    def predict_only(self) -> Tuple[float, float]:
        self.missing_frames += 1

        if not self.initialized:
            return 0.0, 0.0

        # 遮擋時讓速度衰減，防止預測飄移
        self._kf.statePost[2][0] *= self.cfg.velocity_decay
        self._kf.statePost[3][0] *= self.cfg.velocity_decay

        pred = self._kf.predict()
        sx = float(pred[0][0])
        sy = float(pred[1][0])

        self._last_smooth = (sx, sy)
        return sx, sy

    # ── 重置（長時間消失後回來）─────────────────────

    def reset(self):
        self._kf = None
        self.initialized = False
        self.missing_frames = 0
        self._innovation_buf.clear()
        self._last_smooth = None
        self.is_still = True
        self.was_outlier = False

    @property
    def alive(self) -> bool:
        return self.missing_frames < self.cfg.max_missing

    @property
    def state_info(self) -> str:
        """HUD 除錯字串"""
        state = "still" if self.is_still else "moving"
        if self.was_outlier:
            state = "outlier"
        if not self.alive:
            state = "lost"
        return f"Q={self.current_q:.1f} [{state}]"


# ─────────────────────────────────────────────────────
# 多目標管理器
# ─────────────────────────────────────────────────────

class MultiFootTracker:
    """
    管理所有 (person_id, side) 的 AdaptiveKalmanPoint 實例。
    自動建立、重置、清除追蹤器。
    """

    def __init__(self, cfg: AKFConfig = None):
        self.cfg = cfg or AKFConfig()
        self._trackers: dict[tuple, AdaptiveKalmanPoint] = {}

    def update(self, person_id: int, side: str,
               raw_x: float, raw_y: float) -> Tuple[float, float]:
        key = (person_id, side)
        if key not in self._trackers:
            self._trackers[key] = AdaptiveKalmanPoint(self.cfg)
        return self._trackers[key].update(raw_x, raw_y)

    def predict_missing(self, seen_keys: set):
        """對本幀未被偵測到的追蹤器做預測，並清理已消失過久的。"""
        dead = []
        for key, tkr in self._trackers.items():
            if key not in seen_keys:
                tkr.predict_only()
                if not tkr.alive:
                    dead.append(key)
        for key in dead:
            del self._trackers[key]

    def get_info(self, person_id: int, side: str) -> str:
        key = (person_id, side)
        t = self._trackers.get(key)
        return t.state_info if t else ""

    def reset_all(self):
        for t in self._trackers.values():
            t.reset()
        self._trackers.clear()

    @property
    def active_count(self) -> int:
        return len(self._trackers)


# ─────────────────────────────────────────────────────
# 單元測試（直接執行此檔案時跑）
# ─────────────────────────────────────────────────────

def _run_tests():
    print("=== AdaptiveKalmanPoint Unit Tests ===\n")
    cfg = AKFConfig()
    kp  = AdaptiveKalmanPoint(cfg)

    # Test 1: 靜止收斂
    print("Test 1: 靜止收斂（輸入 (100,200) 加雜訊，應收斂到約 (100,200)）")
    rng = np.random.default_rng(42)
    pos = (100.0, 200.0)
    for i in range(30):
        nx = pos[0] + rng.normal(0, 3)
        ny = pos[1] + rng.normal(0, 3)
        sx, sy = kp.update(nx, ny)
    err = np.sqrt((sx-pos[0])**2 + (sy-pos[1])**2)
    print(f"  最終平滑位置：({sx:.2f}, {sy:.2f})  誤差：{err:.2f}px")
    assert err < 5.0, f"靜止收斂誤差過大：{err:.2f}"
    print("  ✅ PASS\n")

    # Test 2: 移動響應（連續移動，非瞬間跳躍）
    print("Test 2: 連續移動響應（每幀移動20px，10幀後應跟上）")
    kp2 = AdaptiveKalmanPoint(cfg)
    for _ in range(20):
        kp2.update(100.0, 200.0)   # 先靜止穩定
    # 每幀移動 20px（模擬快步行走）
    target_x = 100.0
    for i in range(10):
        target_x += 20.0           # 每幀前進 20px
        sx, sy = kp2.update(target_x, 200.0)
        err = abs(sx - target_x)
        print(f"  幀{i+1:02d}: target={target_x:.0f}  smooth={sx:.1f}  lag={err:.1f}px")
    print(f"  {'✅ PASS' if err < 40 else '❌ FAIL'}  (最終lag={err:.1f}px)\n")

    # Test 3: 異常值閘門
    print("Test 3: 異常值閘門（突然出現超遠點，不應大幅影響位置）")
    kp3 = AdaptiveKalmanPoint(cfg)
    for _ in range(20):
        kp3.update(100.0, 100.0)
    sx_before, sy_before = kp3._last_smooth
    sx_out, sy_out = kp3.update(9999.0, 9999.0)   # 極端異常值
    drift = np.sqrt((sx_out-sx_before)**2 + (sy_out-sy_before)**2)
    print(f"  異常值前：({sx_before:.1f}, {sy_before:.1f})")
    print(f"  輸入異常值 (9999, 9999) 後：({sx_out:.1f}, {sy_out:.1f})")
    print(f"  飄移量：{drift:.2f}px")
    print(f"  {'✅ PASS' if drift < 20 else '❌ FAIL'}\n")

    # Test 4: 消失重預測不飄移
    print("Test 4: 遮擋預測（靜止後消失10幀，位置不應飄移）")
    kp4 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp4.update(200.0, 150.0)
    base_x, base_y = kp4._last_smooth
    for i in range(10):
        px, py = kp4.predict_only()
    drift = np.sqrt((px-base_x)**2 + (py-base_y)**2)
    print(f"  靜止基準：({base_x:.1f}, {base_y:.1f})")
    print(f"  10幀預測後：({px:.1f}, {py:.1f})  飄移：{drift:.2f}px")
    print(f"  {'✅ PASS' if drift < 15 else '❌ FAIL'}\n")

    print("=== 所有測試完成 ===")


if __name__ == "__main__":
    _run_tests()
