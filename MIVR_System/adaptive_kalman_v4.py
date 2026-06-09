"""
MIVR-CEIQ Adaptive Kalman Filter v4
adaptive_kalman_v4.py

第一步目標：相機骨架定位「不飄移、不抖動、高精度、低延遲」。

相對 v3 的結構性改動（依專家會議結論）：

  [v4-A] 砍掉前置濾波鏈中的 EWMA 與互補濾波（Complementary Filter）
         原因：兩級低通在移動時帶來數十 ms 群延遲，且讓量測噪音
               「染色」（時間上相關），破壞 Kalman 的白噪音假設、
               造成過度自信與「卡住」病灶。
         保留：狀態感知中位數（靜止去跳點 / 移動直通，零延遲）
               + 速度感知閘門（離群剔除）。
         平滑工作交還給「調好的 KF（adaptive R）+ StationaryLock」。

  [v4-B] 過程噪音改用物理正確的離散白噪音加速度（DWNA）矩陣
         取代 v3 的 eye(4)*q（位置/速度同噪音且不耦合）。
         遮擋預測時的位置不確定度才正確。

  [v4-C] dt 改用 time.perf_counter()（單調時鐘）
         避免 time.time() 因 NTP 校時倒退造成 dt 為負 / 跳動。

  [v4-D] 新增 StationaryLock（遲滯狀態機 / ZUPT 思路）
         靜止時凍結輸出 → σ→0（零抖動、零漂移）；
         移動時放行 → 零延遲跟隨。
         用「進入靜止 / 離開靜止」兩個不同門檻形成遲滯，避免邊界顫動。
         鎖定時同步把 KF 速度歸零、位置貼齊鎖定點，避免解鎖瞬跳。

  [v4-E] KF 改用純 NumPy 實作（移除對 cv2.KalmanFilter 的依賴），
         完整掌控 DWNA Q 與協方差更新。

API 與 v3 相容：AKFConfig / AdaptiveKalmanPoint / MultiFootTracker
update(raw_x, raw_y, norm_y) 簽章不變 → 可直接被
bio_fusion_tracker_v2.py 與 foot_detector_v5/v6 匯入沿用。

注意：本版閾值（still_threshold / gate / lock）皆為「像素」單位。
      未來導入地板 homography 改公尺座標後，把這些換成公分即可，
      閾值將變成尺度不變（這正是後續第二、三步的基礎）。
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────

@dataclass
class AKFConfig:
    # ── 過程噪音（DWNA：q 代表加速度噪音 PSD，單位 px^2/s^3 量級）──
    # 注意：語意已和 v3 不同（v3 的 q 是直接乘 eye(4)）。
    q_base:             float = 6.0      # 靜止/低速基準（小 = 信任模型 = 平滑）
    q_max:              float = 9.0e4    # 高速上限（大 = 信任量測 = 跟得上）
    innovation_window:  int   = 6
    innovation_gain:    float = 9.0      # 位移 → q 的增益
    speed_gain_exp:     float = 2.0

    # ── 量測噪音（資料驅動，零相機假設）──
    r_noise_base:       float = 4.0      # R 下限（px^2，最信任量測時）
    r_noise_max:        float = 80.0     # R 上限（px^2，防過度平滑）
    r_residual_window:  int   = 15
    r_residual_gain:    float = 0.6
    r_dist_scale:       float = 0.0      # 0 = 完全不用畫面位置假設

    # ── 速度 / 衰減 ──
    velocity_decay:     float = 0.80     # 遮擋預測時速度衰減

    # ── 靜止判定（用「半窗均值差」偵測移動，抗噪音又能即時起步）──
    move_win:           int   = 6        # 移動偵測窗（分前後兩半取均值差）
    still_threshold:    float = 6.0      # px/幀：估計速度低於此 → 暫判靜止
    still_damping:      float = 0.04     # 靜止時 q 的縮放（更強平滑）

    # ── 移動時的位置過程噪音（讓 KF 能「追上」凍結期累積的位置缺口）──
    # DWNA 的 q 只放大「速度」不確定度，位置增益太低 → 追不上凍結缺口；
    # 故移動時額外加一個隨速度增長的位置噪音項。
    pos_noise_gain:     float = 2.0      # q_pos = (spd_est)^2 * gain
    pos_noise_max:      float = 2500.0   # px^2 上限

    # ── 速度感知閘門（離群剔除）──
    outlier_gate_min:   float = 45.0     # 絕對最小閘門（靜止時）px
    outlier_gate_vel:   float = 3.5      # 速度容差倍率
    outlier_max_angle:  float = 110.0    # 方向突變角度閾值（度）
    outlier_hist_len:   int   = 5
    reacquire_after:    int   = 2        # 連續同向遠離幾幀 → 重新鎖定（破 deadlock）

    # ── 中位數（離群前處理；移動時直通，零延遲）──
    median_window:      int   = 3        # 奇數

    # ── [v4-D] StationaryLock 遲滯鎖定（錨點偏離 + debounce）──
    lock_enter:         float = 4.0      # px：窗內淨位移低於此 → 進入靜止
    lock_exit_net:      float = 6.0      # px：偏離錨點高於此（連續）→ 解鎖
    lock_exit_jump:     float = 20.0     # px：單幀位移高於此 → 立即解鎖
    lock_hold:          int   = 3        # 進入靜止所需連續幀
    lock_win:           int   = 5        # 淨位移統計窗長
    lock_exit_debounce: int   = 2        # 偏離錨點需連續幾幀才解鎖（抗噪音尖峰）

    # ── 生命週期 ──
    max_missing:        int   = 15


# ─────────────────────────────────────────────────────
# Level 1：狀態感知中位數（保留）
# ─────────────────────────────────────────────────────

class MedianFilter1D:
    """靜止時視窗中位數去跳點；移動時直通（零延遲）。"""
    def __init__(self, window: int = 3):
        assert window % 2 == 1, "視窗必須為奇數"
        self._buf: deque = deque(maxlen=window)

    def filter(self, x: float, is_still: bool = True) -> float:
        self._buf.append(x)
        if not is_still:
            return x
        return float(np.median(self._buf))

    def reset(self):
        self._buf.clear()


# ─────────────────────────────────────────────────────
# 速度感知閘門（保留 v3 設計）
# ─────────────────────────────────────────────────────

class VelocityAwareGate:
    """
    以「近期實際移動距離」為半徑的動態閘門。

    [v4 關鍵修正] 根治 v3 的「快速移動 deadlock」：
      v3 的閘門在拒絕離群點時，不會把該位移記進歷史 →
      站著突然快走（單幀位移 > gate_min）會被判離群 → 拒絕 →
      歷史不更新 → 閘門永遠停在 gate_min → 每一幀都被拒 →
      永久卡住、完全追不到人（已實測重現：輸出凍結、落後 600px）。

      修法：連續 reacquire_after 幀都「遠離」且方向一致 →
            判定為真實快速移動 → 重新鎖定（snap 到最新量測 + 估速度）。
            單幀孤立跳點（雜訊）仍被當離群剔除（不滿足連續條件）。
    """
    def __init__(self, gate_min=45.0, vel_factor=3.5, max_angle=110.0,
                 history_len=5, reacquire_after=2):
        self._gate_min = gate_min
        self._vel_factor = vel_factor
        self._max_angle = max_angle
        self._dist_hist: deque = deque(maxlen=history_len)
        self._prev_move_dir: Optional[Tuple[float, float]] = None
        self._reacquire_after = reacquire_after
        self._reject_count = 0
        self._reject_pts: deque = deque(maxlen=3)

    def check(self, raw_x, raw_y, last_pos) -> Tuple[str, str]:
        """回傳 (action, reason)；action ∈ {'ok','reject','reacquire'}"""
        if last_pos is None:
            return "ok", "init"
        sx, sy = last_pos
        dx, dy = raw_x - sx, raw_y - sy
        dist = float(np.hypot(dx, dy))

        recent_max = max(self._dist_hist) if self._dist_hist else 0.0
        gate_r = max(self._gate_min, recent_max * self._vel_factor)

        if dist > gate_r:
            # 方向一致性：與前一個被拒位移同向才算「持續快速移動」
            consistent = True
            if self._reject_pts:
                px, py = (raw_x - self._reject_pts[-1][0],
                          raw_y - self._reject_pts[-1][1])
                # 與整體偏離方向(dx,dy)比對
                if np.hypot(px, py) > 1.0 and dist > 1.0:
                    cos_a = (px * dx + py * dy) / (np.hypot(px, py) * dist)
                    consistent = cos_a > 0.3
            self._reject_count = self._reject_count + 1 if consistent else 1
            self._reject_pts.append((raw_x, raw_y))

            if self._reject_count >= self._reacquire_after:
                # 連續同向遠離 = 真實快速移動 → 重新鎖定
                self._reject_count = 0
                self._dist_hist.clear()
                self._dist_hist.append(dist)          # 閘門立即變寬
                self._prev_move_dir = (dx, dy)
                return "reacquire", f"reacquire dist={dist:.0f}"
            return "reject", f"dist={dist:.0f}>gate={gate_r:.0f}"

        # 方向突變（誤偵測特徵：移動中突然反向）
        if (self._prev_move_dir is not None and
                dist > self._gate_min * 0.3 and recent_max > self._gate_min * 0.3):
            px, py = self._prev_move_dir
            pl = float(np.hypot(px, py))
            if pl > 3.0 and dist > 3.0:
                cos_a = float(np.clip((px*dx + py*dy) / (pl*dist), -1, 1))
                if float(np.degrees(np.arccos(cos_a))) > self._max_angle:
                    return "reject", f"angle"

        self._reject_count = 0
        self._dist_hist.append(dist)
        self._prev_move_dir = (dx, dy)
        return "ok", "ok"

    def reacquire_velocity(self, dt: float) -> Tuple[float, float]:
        """從近期被拒位置估算再鎖定時的初速度（px/s）。"""
        if len(self._reject_pts) >= 2:
            (x0, y0), (x1, y1) = self._reject_pts[-2], self._reject_pts[-1]
            return (x1 - x0) / max(dt, 1e-4), (y1 - y0) / max(dt, 1e-4)
        return 0.0, 0.0

    def reset(self):
        self._dist_hist.clear()
        self._prev_move_dir = None
        self._reject_count = 0
        self._reject_pts.clear()


# ─────────────────────────────────────────────────────
# [v4-D] StationaryLock 遲滯鎖定
# ─────────────────────────────────────────────────────

class StationaryLock:
    """
    靜止時凍結輸出（零抖動、零漂移），移動時放行（零延遲）。

    鎖定中的解鎖判斷（兩條路）：
      1. 單幀大跳 inst > exit_jump → 立即解鎖（快速移動，零延遲）
      2. 偏離錨點 d_anchor > exit_net 連續 debounce 幀 → 解鎖（持續移動）
         → 單一噪音尖峰（1 幀）不會解鎖，真實位移才會。
    未鎖定 → 鎖定：窗內淨位移 < enter 連續 hold 幀（零均值噪音淨位移會抵消）。
    """
    def __init__(self, enter=4.0, exit_net=6.0, exit_jump=20.0,
                 hold=3, win=5, exit_debounce=2):
        self.enter = enter
        self.exit_net = exit_net
        self.exit_jump = exit_jump
        self.hold = hold
        self.exit_debounce = exit_debounce
        self.is_still = False
        self.locked: Optional[Tuple[float, float]] = None
        self._hist: deque = deque(maxlen=win)
        self._prev_kf: Optional[Tuple[float, float]] = None
        self._cnt = 0
        self._exit_cnt = 0

    def step(self, x: float, y: float,
             ref: Optional[Tuple[float, float]]) -> Tuple[float, float, bool]:
        self._hist.append((x, y))
        inst = 0.0 if self._prev_kf is None \
            else float(np.hypot(x - self._prev_kf[0], y - self._prev_kf[1]))
        self._prev_kf = (x, y)

        if self.locked is None and not self.is_still:
            if ref is None:
                return x, y, False

        if self.is_still:
            d_anchor = float(np.hypot(x - self.locked[0], y - self.locked[1]))
            if inst > self.exit_jump:                  # 快速移動 → 立即解鎖
                self.is_still = False; self._exit_cnt = 0; self.locked = None
                return x, y, False
            if d_anchor > self.exit_net:               # 偏離錨點
                self._exit_cnt += 1
                if self._exit_cnt >= self.exit_debounce:
                    self.is_still = False; self._exit_cnt = 0; self.locked = None
                    return x, y, False
            else:
                self._exit_cnt = 0
            return self.locked[0], self.locked[1], True   # 維持凍結
        else:
            ox0, oy0 = self._hist[0]
            net = float(np.hypot(x - ox0, y - oy0))    # 窗內淨位移
            if net < self.enter:
                self._cnt += 1
                if self._cnt >= self.hold:
                    self.is_still = True
                    self.locked = (x, y)
                    self._exit_cnt = 0
                    return x, y, True
            else:
                self._cnt = 0
            return x, y, False

    def reset(self):
        self.is_still = False
        self.locked = None
        self._hist.clear()
        self._prev_kf = None
        self._cnt = 0
        self._exit_cnt = 0


# ─────────────────────────────────────────────────────
# [v4-B] DWNA 過程噪音矩陣
# ─────────────────────────────────────────────────────

def dwna_Q(dt: float, q: float) -> np.ndarray:
    """離散白噪音加速度模型的過程噪音（位置/速度耦合）。"""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    return q * np.array([
        [dt4 / 4, 0,       dt3 / 2, 0],
        [0,       dt4 / 4, 0,       dt3 / 2],
        [dt3 / 2, 0,       dt2,     0],
        [0,       dt3 / 2, 0,       dt2],
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────
# 主類別：AdaptiveKalmanPoint v4（純 NumPy KF）
# ─────────────────────────────────────────────────────

class AdaptiveKalmanPoint:
    """
    狀態向量 [x, y, vx, vy]。
    管線：NaN守衛 → 速度閘門(raw) → 暫定靜止判定 → 中位數(raw)
          → adaptive Q/R → KF predict/update → StationaryLock(輸出)
    """

    def __init__(self, cfg: AKFConfig = None):
        self.cfg = cfg or AKFConfig()
        self.initialized = False
        self.missing_frames = 0

        # KF 狀態
        self._x = np.zeros((4, 1), dtype=np.float64)        # state
        self._P = np.eye(4, dtype=np.float64) * 50.0         # covariance
        self._H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)

        # Level 1 中位數
        self._med_x = MedianFilter1D(self.cfg.median_window)
        self._med_y = MedianFilter1D(self.cfg.median_window)

        # 閘門 + 鎖定
        self._gate = VelocityAwareGate(self.cfg.outlier_gate_min,
                                       self.cfg.outlier_gate_vel,
                                       self.cfg.outlier_max_angle,
                                       self.cfg.outlier_hist_len,
                                       self.cfg.reacquire_after)
        self._lock = StationaryLock(self.cfg.lock_enter,
                                    self.cfg.lock_exit_net,
                                    self.cfg.lock_exit_jump,
                                    self.cfg.lock_hold,
                                    self.cfg.lock_win,
                                    self.cfg.lock_exit_debounce)

        # 自適應統計
        self._innov_buf: deque = deque(maxlen=self.cfg.innovation_window)
        self._residual_buf: deque = deque(maxlen=self.cfg.r_residual_window)
        self._raw_hist: deque = deque(maxlen=self.cfg.move_win)  # 移動偵測

        # 上一「輸出」位置（Lock 後）；閘門 / Q 都以它為參考
        self._last_out: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None

        # 診斷
        self.current_q = self.cfg.q_base
        self.current_r = self.cfg.r_noise_base
        self.current_dt = 1.0 / 30.0
        self.is_still = True
        self.is_locked = False
        self.outlier_reason = ""

    # ── KF 基本運算 ──────────────────────────────────

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

    def _predict(self, dt: float, q: float, q_pos: float = 0.0):
        F = self._F(dt)
        Q = dwna_Q(dt, q)
        if q_pos > 0.0:                      # 移動時加位置噪音 → 能追上缺口
            Q[0, 0] += q_pos
            Q[1, 1] += q_pos
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def _update(self, mx: float, my: float, r: float):
        z = np.array([[mx], [my]], dtype=np.float64)
        R = np.array([[r, 0], [0, r]], dtype=np.float64)
        y = z - self._H @ self._x                       # 新息
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P

    # ── 自適應 Q ─────────────────────────────────────

    def _compute_q(self, move_metric: float, provisional_still: bool) -> float:
        """move_metric: px/幀（取 KF 速度與 raw 位移的較大者，移動時快速跟隨）。"""
        self._innov_buf.append(move_metric)
        if provisional_still:
            return self.cfg.q_base * self.cfg.still_damping
        max_i = max(self._innov_buf)
        eff = max(move_metric, max_i * 0.5)
        q = self.cfg.q_base + (eff * self.cfg.innovation_gain) ** self.cfg.speed_gain_exp
        return float(min(q, self.cfg.q_max))

    # ── 資料驅動 R（MAD，零相機假設）─────────────────

    def _compute_r(self, norm_y: float = 0.5) -> float:
        if len(self._residual_buf) < 3:
            r = self.cfg.r_noise_base
        else:
            med_res = float(np.median(self._residual_buf))
            robust_std = med_res * 1.4826
            r = self.cfg.r_noise_base + (robust_std ** 2) * self.cfg.r_residual_gain
            r = float(np.clip(r, self.cfg.r_noise_base, self.cfg.r_noise_max))
        if self.cfg.r_dist_scale > 0.0:
            r = min(r * (1.0 + self.cfg.r_dist_scale * (1.0 - norm_y) ** 2),
                    self.cfg.r_noise_max)
        self.current_r = r
        return r

    def _record_residual(self, meas_x: float, meas_y: float):
        # 用 raw 對「predict 後的預測位置」算殘差 → 反映 YOLO 真實噪音
        pred_x = float(self._x[0, 0])
        pred_y = float(self._x[1, 0])
        self._residual_buf.append(float(np.hypot(meas_x - pred_x, meas_y - pred_y)))

    # ── 主更新 ───────────────────────────────────────

    def update(self, raw_x: float, raw_y: float,
               norm_y: float = 0.5,
               dt: Optional[float] = None) -> Tuple[float, float]:
        """dt 可由呼叫端傳入（建議用相機影格時間戳，比 wall clock 準）。
           dt=None 時退回 perf_counter() 單調時鐘自動量測。"""
        # NaN / Inf 守衛
        if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
            if self.initialized:
                return self.predict_only(dt)
            return 0.0, 0.0

        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        now = time.perf_counter()                       # [v4-C] 單調時鐘
        if dt is None:
            dt = 1.0 / 30.0 if self._last_time is None \
                else float(np.clip(now - self._last_time, 1e-4, 0.5))
        else:
            dt = float(np.clip(dt, 1e-4, 0.5))
        self._last_time = now
        self.current_dt = dt

        # 首次初始化
        if not self.initialized:
            self._med_x.filter(raw_x, True)
            self._med_y.filter(raw_y, True)
            self._x = np.array([[raw_x], [raw_y], [0.0], [0.0]], dtype=np.float64)
            self._P = np.eye(4, dtype=np.float64) * 50.0
            self.initialized = True
            self._last_out = (raw_x, raw_y)
            self.missing_frames = 0
            return raw_x, raw_y

        # 速度感知閘門（用 raw 對上一輸出判斷；在濾波之前）
        action, reason = self._gate.check(raw_x, raw_y, self._last_out)
        if action == "reject":
            self.outlier_reason = reason
            self._predict(dt, self.cfg.q_base)          # 只預測，不吃這個量測
            ox, oy = float(self._x[0, 0]), float(self._x[1, 0])
            self._last_out = (ox, oy)
            return ox, oy
        if action == "reacquire":
            # 連續快速移動 → 重新鎖定到最新量測 + 估初速度（破 deadlock）
            self.outlier_reason = reason
            vx, vy = self._gate.reacquire_velocity(dt)
            self._x = np.array([[raw_x], [raw_y], [vx], [vy]], dtype=np.float64)
            self._P = np.eye(4, dtype=np.float64) * 50.0
            self._lock.reset()
            self._med_x.reset(); self._med_y.reset()
            self._raw_hist.clear()
            self._last_out = (raw_x, raw_y)
            self.missing_frames = 0
            return raw_x, raw_y
        self.outlier_reason = ""

        # 暫定靜止判定：結合「KF 平滑速度」與「半窗均值差」兩個抗噪音指標。
        #   半窗均值差 = |最近半窗均值 - 較舊半窗均值|，零均值噪音會抵消、
        #   真實移動會累積；除以半窗長 → 估計速度(px/幀)，起步即可偵測。
        vx, vy = float(self._x[2, 0]), float(self._x[3, 0])     # 上一幀後驗速度
        kf_spd_pf = float(np.hypot(vx, vy)) * dt                # px/幀
        raw_dist = float(np.hypot(raw_x - self._last_out[0],
                                  raw_y - self._last_out[1]))

        self._raw_hist.append((raw_x, raw_y))
        win_spd = 0.0
        if len(self._raw_hist) >= 4:
            arr = np.array(self._raw_hist)
            h = len(arr) // 2
            recent = arr[h:].mean(axis=0)
            older = arr[:h].mean(axis=0)
            win_spd = float(np.hypot(*(recent - older))) / max(h, 1)

        spd_est = max(kf_spd_pf, win_spd)                       # px/幀（抗噪音）
        provisional_still = spd_est < self.cfg.still_threshold

        # Level 1 中位數（靜止去跳點 / 移動直通）
        m_x = self._med_x.filter(raw_x, provisional_still)
        m_y = self._med_y.filter(raw_y, provisional_still)

        # adaptive Q（移動量取估計速度與 raw 位移較大者 → 起步快）
        q = self._compute_q(max(raw_dist, spd_est), provisional_still)
        self.current_q = q

        # 移動時的位置過程噪音（追上凍結期缺口；靜止時為 0 → 不影響穩定）
        if provisional_still:
            q_pos = 0.0
        else:
            q_pos = float(min((spd_est ** 2) * self.cfg.pos_noise_gain,
                              self.cfg.pos_noise_max))

        # KF predict → 記殘差（raw vs 預測）→ 估 R → KF update
        self._predict(dt, q, q_pos)
        self._record_residual(raw_x, raw_y)
        r = self._compute_r(norm_y)
        self._update(m_x, m_y, r)

        sx = float(self._x[0, 0])
        sy = float(self._x[1, 0])

        # [v4-D] StationaryLock：最終輸出穩定化
        # 只凍結「輸出」，不動 KF 內部狀態 →
        #   靜止時 q 極小，KF 內部本來就幾乎不動（無內部漂移）；
        #   一旦真的移動，KF 速度可立即建立（解鎖無延遲）。
        ox, oy, locked = self._lock.step(sx, sy, self._last_out)
        self.is_locked = locked
        self.is_still = locked or provisional_still

        self._last_out = (ox, oy)
        self.missing_frames = 0
        return ox, oy

    # ── 遮擋預測 ─────────────────────────────────────

    def predict_only(self, dt: Optional[float] = None) -> Tuple[float, float]:
        self.missing_frames += 1
        if not self.initialized:
            return 0.0, 0.0
        now = time.perf_counter()
        if dt is None:
            dt = 1.0 / 30.0 if self._last_time is None \
                else float(np.clip(now - self._last_time, 1e-4, 0.2))
        else:
            dt = float(np.clip(dt, 1e-4, 0.2))
        self._last_time = now
        # 速度衰減後純預測
        self._x[2, 0] *= self.cfg.velocity_decay
        self._x[3, 0] *= self.cfg.velocity_decay
        self._predict(dt, self.cfg.q_base)
        ox, oy = float(self._x[0, 0]), float(self._x[1, 0])
        self._last_out = (ox, oy)
        return ox, oy

    def reset(self):
        self.initialized = False
        self.missing_frames = 0
        self._x = np.zeros((4, 1))
        self._P = np.eye(4) * 50.0
        self._innov_buf.clear()
        self._residual_buf.clear()
        self._raw_hist.clear()
        self._last_out = None
        self._last_time = None
        self._med_x.reset(); self._med_y.reset()
        self._gate.reset(); self._lock.reset()
        self.is_still = True
        self.is_locked = False
        self.outlier_reason = ""

    @property
    def alive(self) -> bool:
        return self.missing_frames < self.cfg.max_missing

    @property
    def state_info(self) -> str:
        s = "lock" if self.is_locked else ("still" if self.is_still else "move")
        if self.outlier_reason:
            s = f"gate[{self.outlier_reason}]"
        if not self.alive:
            s = "lost"
        return (f"Q={self.current_q:.0f} R={self.current_r:.1f} "
                f"dt={self.current_dt*1000:.0f}ms [{s}]")


# ─────────────────────────────────────────────────────
# 多目標管理器（API 與 v3 相容）
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
# 單元測試（把「不抖、不飄、低延遲」變成數字）
# ─────────────────────────────────────────────────────

def _run_tests():
    print("=== AdaptiveKalmanPoint v4 Unit Tests ===\n")
    cfg = AKFConfig()
    rng = np.random.default_rng(42)
    DT = 1.0 / 30.0          # 寫死真實影格 dt，測試確定性、免 sleep
    results = []

    # Test 1：靜止收斂 + 抖動抑制（核心指標：σ→0）
    print("Test 1: 靜止收斂 + 抖動抑制（輸入 ±4px 噪音）")
    kp = AdaptiveKalmanPoint(cfg)
    pos = []
    for _ in range(80):
        nx = 100.0 + rng.normal(0, 4)
        ny = 200.0 + rng.normal(0, 4)
        sx, sy = kp.update(nx, ny, 0.7, dt=DT)
        pos.append((sx, sy))
    arr = np.array(pos[-30:])
    std_x, std_y = float(np.std(arr[:, 0])), float(np.std(arr[:, 1]))
    err = float(np.hypot(sx - 100, sy - 200))
    ok = std_x < 0.5 and std_y < 0.5 and err < 5
    results.append(ok)
    print(f"  最終=({sx:.2f},{sy:.2f}) 誤差={err:.2f}px  locked={kp.is_locked}")
    print(f"  最後30幀 σx={std_x:.4f} σy={std_y:.4f} (目標<0.5px)")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 2：連續移動響應（20px/幀，lag）
    print("Test 2: 連續等速移動（20px/幀，12幀）lag")
    kp2 = AdaptiveKalmanPoint(cfg)
    for _ in range(20):
        kp2.update(100.0, 200.0, 0.7, dt=DT)
    tx, max_lag, steady_lag = 100.0, 0.0, 0.0
    for i in range(12):
        tx += 20.0
        sx, sy = kp2.update(tx, 200.0, 0.7, dt=DT)
        lag = abs(sx - tx)
        max_lag = max(max_lag, lag)
        if i >= 6:
            steady_lag = max(steady_lag, lag)   # 穩態 lag（速度建立後）
    ok = steady_lag < 30.0
    results.append(ok)
    print(f"  穩態 lag={steady_lag:.1f}px (目標<30=1.5幀) 最大={max_lag:.1f} Q={kp2.current_q:.0f}")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 3：快速移動不卡住（每幀 30px）
    print("Test 3: 快速移動不卡住（30px/幀）")
    kp3 = AdaptiveKalmanPoint(cfg)
    x = 100.0
    for _ in range(15):
        x += 30.0; kp3.update(x, 200.0, 0.7, dt=DT)
    before = kp3._last_out[0]
    x += 30.0
    sx, _ = kp3.update(x, 200.0, 0.7, dt=DT)
    moved = abs(sx - before)
    ok = moved > 15.0
    results.append(ok)
    print(f"  位移={moved:.1f}px (目標>15) outlier='{kp3.outlier_reason}'")
    print(f"  {'PASS (不卡住)' if ok else 'FAIL (卡住)'}\n")

    # Test 4：異常值閘門（靜止時遠跳點被攔）
    print("Test 4: 異常值閘門（靜止時 200px 遠跳點）")
    kp4 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp4.update(200.0, 300.0, 0.5, dt=DT)
    bx = kp4._last_out[0]
    kp4.update(400.0, 300.0, 0.5, dt=DT)
    drift = abs(kp4._last_out[0] - bx)
    ok = drift < 20
    results.append(ok)
    print(f"  飄移={drift:.1f}px gate='{kp4.outlier_reason}'")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 5：資料驅動 R（穩定點 R 小 / 抖動點 R 大）
    print("Test 5: 資料驅動 R")
    rng5 = np.random.default_rng(7)
    ks = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        ks.update(400.0 + rng5.normal(0, 1.5), 300.0, 0.5, dt=DT)
    r_stable = ks.current_r
    kj = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kj.update(400.0 + rng5.normal(0, 12.0), 300.0, 0.5, dt=DT)
    r_jitter = kj.current_r
    ok = r_jitter > r_stable * 1.3
    results.append(ok)
    print(f"  穩定 R={r_stable:.2f}  抖動 R={r_jitter:.2f}  比值={r_jitter/max(r_stable,1e-6):.2f}x")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 6：遮擋預測不飄移
    print("Test 6: 遮擋預測（10幀 predict_only）")
    kp6 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp6.update(300.0, 400.0, 0.6, dt=DT)
    bx, by = kp6._last_out
    for _ in range(10):
        kp6.predict_only(dt=DT)
    drift = float(np.hypot(kp6._last_out[0] - bx, kp6._last_out[1] - by))
    ok = drift < 15
    results.append(ok)
    print(f"  飄移={drift:.2f}px (目標<15) {'PASS' if ok else 'FAIL'}\n")

    # Test 7：抖動抑制率（輸入 ±8px）
    print("Test 7: 抖動抑制率（輸入 ±8px，含靜止鎖定）")
    rng7 = np.random.default_rng(2024)
    raw_v, sm_v = [], []
    kp7 = AdaptiveKalmanPoint(cfg)
    for i in range(100):
        raw = 400.0 + rng7.normal(0, 8)
        sx, _ = kp7.update(raw, 300.0, 0.6, dt=DT)
        if i >= 25:
            raw_v.append(raw); sm_v.append(sx)
    raw_std, sm_std = float(np.std(raw_v)), float(np.std(sm_v))
    reduction = (1 - sm_std / raw_std) * 100
    ok = reduction > 90
    results.append(ok)
    print(f"  輸入 σ={raw_std:.2f} 輸出 σ={sm_std:.3f} 抑制率={reduction:.1f}% (目標>90%)")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 8：靜止→移動（解鎖後快速追上，無大幅倒退）
    print("Test 8: 靜止→移動（解鎖追上 + 無倒退）")
    kp8 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp8.update(150.0 + rng.normal(0, 3), 250.0, 0.5, dt=DT)
    locked_before = kp8.is_locked
    xx, seq = 150.0, []
    for _ in range(8):
        xx += 18.0
        ox, _ = kp8.update(xx, 250.0, 0.5, dt=DT); seq.append(ox)
    advanced = seq[-1] - 150.0                 # 8幀後總前進
    min_out = min(seq)                          # 過程中最低點（檢查倒退）
    catch_lag = abs(seq[-1] - xx)               # 末幀落後目標
    ok = (locked_before and advanced > 80 and
          min_out > 150.0 - 10 and catch_lag < 35)
    results.append(ok)
    print(f"  鎖定={locked_before} 8幀後前進={advanced:.1f}px 末幀lag={catch_lag:.1f}px "
          f"最低={min_out:.1f}(應≈150不倒退)")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    # Test 9：站著→突然快速移動（60px/幀）不卡死（v3 deadlock 回歸測試）
    print("Test 9: 站立→突然快速移動 60px/幀（v3 deadlock 回歸）")
    kp9 = AdaptiveKalmanPoint(cfg)
    for _ in range(25):
        kp9.update(300.0 + rng.normal(0, 2), 300.0, 0.5, dt=DT)
    tx, lags = 300.0, []
    for _ in range(10):
        tx += 60.0
        sx, _ = kp9.update(tx, 300.0, 0.5, dt=DT)
        lags.append(tx - sx)
    final_lag = lags[-1]
    ok = final_lag < 40        # 應在數幀內重新鎖定追上（v3 此處落後 600px）
    results.append(ok)
    print(f"  末幀落後={final_lag:.1f}px (v3=600px卡死, 目標<40) 過程lag={[f'{l:.0f}' for l in lags]}")
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    print("=" * 50)
    print(f"  通過 {sum(results)}/{len(results)}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    _run_tests()
