"""
MIVR-CEIQ Adaptive Kalman Filter v3
adaptive_kalman_v3.py

修正問題：
  [Fix-A] 抖動：三級訊號清理鏈
    Level 1 — 中位數視窗（去除 YOLOv8 單幀跳點）
    Level 2 — 指數加權移動平均 EWMA（去除高頻噪音）
    Level 3 — AKF（最終平滑 + 物理預測）
    互補濾波截止頻率從 4Hz 降至 2Hz（更徹底去抖）

  [Fix-B] 快速移動後「卡住」：速度感知閘門
    舊方法：固定 outlier_gate=100px → 快速移動被誤判為異常值
    新方法：動態閘門 = max(固定下限, 預測位置 + 速度容差)
    概念：「Kalman 預測你會到哪裡，閘門以預測位置為圓心」
    → 快速移動時閘門跟著移動，不再卡住
    → 真正的異常值（方向突變）仍被攔截

  [Fix-C] 改善靜止穩定性
    靜止時降低互補濾波截止頻率至 1Hz（更強抑制）
    動態切換截止頻率：靜止 1Hz / 移動 2.5Hz
"""

import numpy as np
import cv2
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────

@dataclass
class AKFConfig:
    # ── 過程噪音 ──────────────────────────────────────
    q_base:             float = 0.3
    q_max:              float = 1200.0
    innovation_window:  int   = 6
    innovation_gain:    float = 4.5
    speed_gain_exp:     float = 1.5

    # ── 量測噪音（資料驅動，零相機假設）──────────────
    # 不再依賴「畫面位置 → 距離」的假設（會強迫使用者用特定角度）
    # 改為從「量測殘差」自己學每個點的真實噪音水平
    r_noise_base:       float = 5.0     # R 下限（最信任量測時）
    r_noise_max:        float = 60.0    # R 上限（防止過度平滑）
    r_residual_window:  int   = 15      # 殘差統計視窗
    r_residual_gain:    float = 0.6     # 殘差變異 → R 的增益

    # 相容性保留（設 0 = 完全不用畫面位置假設，預設關閉）
    r_dist_scale:       float = 0.0

    # ── 速度 / 衰減 ──────────────────────────────────
    velocity_decay:     float = 0.80

    # ── 靜止判定 ─────────────────────────────────────
    still_threshold:    float = 1.8     # 比 v2 更嚴格（減少誤判）
    still_damping:      float = 0.03

    # ── [Fix-B] 速度感知閘門 ──────────────────────────
    outlier_gate_min:   float = 40.0    # 絕對最小閘門（靜止時）
    outlier_gate_vel:   float = 3.5     # 速度容差倍率（px/幀 × 倍率）
    outlier_max_angle:  float = 110.0   # 方向突變角度閾值（度）

    # ── [Fix-A] 三級濾波 ─────────────────────────────
    median_window:      int   = 3       # 中位數視窗大小（奇數）
    ewma_alpha_move:    float = 0.85    # EWMA 移動時平滑係數（較大=更跟隨）
    ewma_alpha_still:   float = 0.20    # EWMA 靜止時平滑係數（較小=更穩定）
    cf_cutoff_move:     float = 2.5     # 互補濾波截止：移動時（Hz）
    cf_cutoff_still:    float = 1.0     # 互補濾波截止：靜止時（Hz）

    # ── 生命週期 ─────────────────────────────────────
    max_missing:        int   = 15


# ─────────────────────────────────────────────────────
# Level 1：中位數視窗濾波
# ─────────────────────────────────────────────────────

class MedianFilter1D:
    """
    [Fix-A L1] 動態中位數濾波
    靜止時：視窗 3 → 去除跳點
    移動時：視窗 1（直通）→ 零延遲跟隨
    初期（buffer < 2）返回原始值以避免過度平滑
    """
    def __init__(self, window: int = 3):
        assert window % 2 == 1, "視窗必須為奇數"
        self._win_max = window
        self._buf: deque = deque(maxlen=window)

    def filter(self, x: float, is_still: bool = True) -> float:
        self._buf.append(x)
        if not is_still:
            return x   # 移動時直通，零延遲
        # 初期直通，避免少量點做中位數產生過度平滑
        if len(self._buf) < 2:
            return x
        return float(np.median(list(self._buf)))

    def reset(self):
        self._buf.clear()


# ─────────────────────────────────────────────────────
# Level 2：動態 EWMA
# ─────────────────────────────────────────────────────

class DynamicEWMA1D:
    """
    [Fix-A L2] 動態指數加權移動平均
    靜止時 alpha 小 → 強平滑
    移動時 alpha 大 → 快速跟隨
    比 FIR 低通有更好的移動響應（無固定相位延遲）
    """
    def __init__(self, alpha_move: float = 0.55,
                 alpha_still: float = 0.20):
        self._a_move  = alpha_move
        self._a_still = alpha_still
        self._prev:   Optional[float] = None

    def filter(self, x: float, is_still: bool) -> float:
        if self._prev is None:
            self._prev = x
            return x
        alpha     = self._a_still if is_still else self._a_move
        filtered  = alpha * x + (1.0 - alpha) * self._prev
        self._prev = filtered
        return filtered

    def reset(self):
        self._prev = None


# ─────────────────────────────────────────────────────
# Level 3a：動態互補濾波（截止頻率隨狀態切換）
# ─────────────────────────────────────────────────────

class DynamicComplementaryFilter1D:
    """
    [Fix-A L3a] 互補低通濾波
    靜止 1.0Hz / 移動 2.5Hz 動態切換
    """
    def __init__(self, cutoff_still: float = 1.0,
                 cutoff_move:  float = 2.5):
        self._fc_still = cutoff_still
        self._fc_move  = cutoff_move
        self._prev:    Optional[float] = None

    def filter(self, x: float, dt: float, is_still: bool) -> float:
        dt  = max(dt, 1e-4)
        fc  = self._fc_still if is_still else self._fc_move
        tau = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (dt + tau)

        if self._prev is None:
            self._prev = x
            return x
        filtered   = alpha * x + (1.0 - alpha) * self._prev
        self._prev = filtered
        return filtered

    def reset(self):
        self._prev = None


# ─────────────────────────────────────────────────────
# [Fix-B] 速度感知閘門
# ─────────────────────────────────────────────────────

class VelocityAwareGate:
    """
    [Fix-B] 基於移動歷史的動態閘門

    設計原則：
      - 閘門圓心：上幀平滑位置（不依賴 KF 速度狀態，更可靠）
      - 閘門半徑：max(gate_min, 近期最大移動距離 × multiplier)
      - 近期移動距離由滑動視窗維護，比 KF vx 更快響應
      - 方向突變：連續移動中突然 90° 轉向 → 誤偵測特徵

    效果：
      快速移動（每幀 30px）→ history_max=30 → gate=30×3=90 → 不卡住
      靜止誤跳（突然 200px）→ history_max≈0 → gate=40 → 攔截
    """

    def __init__(self, gate_min: float = 40.0,
                 vel_factor: float = 3.5,
                 max_angle:  float = 110.0,
                 history_len: int  = 5):
        self._gate_min   = gate_min
        self._vel_factor = vel_factor
        self._max_angle  = max_angle
        self._dist_hist: deque = deque(maxlen=history_len)
        self._prev_move_dir: Optional[Tuple[float, float]] = None

    def is_outlier(self, raw_x: float, raw_y: float,
                   last_smooth: Tuple[float, float]) -> Tuple[bool, str]:
        """
        last_smooth: 上幀 Kalman 平滑後的位置
        回傳: (is_outlier, reason)
        """
        if last_smooth is None:
            return False, "ok"

        sx, sy = last_smooth
        dx   = raw_x - sx
        dy   = raw_y - sy
        dist = float(np.sqrt(dx*dx + dy*dy))

        # 動態閘門：基於近期實際移動距離
        if self._dist_hist:
            recent_max = max(self._dist_hist)
        else:
            recent_max = 0.0
        gate_r = max(self._gate_min, recent_max * self._vel_factor)

        if dist > gate_r:
            return True, f"dist={dist:.0f}>gate={gate_r:.0f}"

        # 方向突變（誤偵測特徵：位置突然跳到反方向）
        if (self._prev_move_dir is not None and
                dist > self._gate_min * 0.3 and
                recent_max > self._gate_min * 0.3):
            px, py = self._prev_move_dir
            prev_len = float(np.sqrt(px*px + py*py))
            curr_len = float(np.sqrt(dx*dx + dy*dy))
            if prev_len > 3.0 and curr_len > 3.0:
                pdir = np.array([px/prev_len, py/prev_len])
                cdir = np.array([dx/curr_len, dy/curr_len])
                cos_a = float(np.clip(np.dot(pdir, cdir), -1, 1))
                angle = float(np.degrees(np.arccos(cos_a)))
                if angle > self._max_angle:
                    return True, f"angle={angle:.0f}deg"

        # 通過閘門：記錄本幀移動距離
        self._dist_hist.append(dist)
        self._prev_move_dir = (dx, dy)
        return False, "ok"

    def reset(self):
        self._dist_hist.clear()
        self._prev_move_dir    = None


# ─────────────────────────────────────────────────────
# 主類別：AdaptiveKalmanPoint v3
# ─────────────────────────────────────────────────────

class AdaptiveKalmanPoint:
    """
    三級訊號清理 + 速度感知閘門 + 自適應 KF
    狀態向量：[x, y, vx, vy]
    """

    def __init__(self, cfg: AKFConfig = None):
        self.cfg = cfg or AKFConfig()
        self._kf: Optional[cv2.KalmanFilter] = None
        self.initialized    = False
        self.missing_frames = 0

        # Level 1 中位數
        self._med_x = MedianFilter1D(self.cfg.median_window)
        self._med_y = MedianFilter1D(self.cfg.median_window)

        # Level 2 EWMA
        self._ema_x = DynamicEWMA1D(self.cfg.ewma_alpha_move,
                                     self.cfg.ewma_alpha_still)
        self._ema_y = DynamicEWMA1D(self.cfg.ewma_alpha_move,
                                     self.cfg.ewma_alpha_still)

        # Level 3a 互補濾波
        self._cf_x  = DynamicComplementaryFilter1D(self.cfg.cf_cutoff_still,
                                                    self.cfg.cf_cutoff_move)
        self._cf_y  = DynamicComplementaryFilter1D(self.cfg.cf_cutoff_still,
                                                    self.cfg.cf_cutoff_move)

        # [Fix-B] 速度感知閘門
        self._gate  = VelocityAwareGate(self.cfg.outlier_gate_min,
                                         self.cfg.outlier_gate_vel,
                                         self.cfg.outlier_max_angle)

        # 新息歷史（自適應 Q）
        self._innov_buf: deque = deque(maxlen=self.cfg.innovation_window)
        self._last_smooth: Optional[Tuple[float, float]] = None

        # 量測殘差歷史（資料驅動 R，零相機假設）
        self._residual_buf: deque = deque(maxlen=self.cfg.r_residual_window)

        # 時間
        self._last_time: Optional[float] = None

        # 診斷
        self.current_q    = self.cfg.q_base
        self.current_r    = self.cfg.r_noise_base
        self.current_dt   = 1.0 / 30.0
        self.is_still     = True
        self.outlier_reason = ""

        # [效能優化] 預初始化常用的矩陣，減少每幀分配開銷
        self._kf_r_cov: np.ndarray = np.eye(2, dtype=np.float32)
        self._kf_q_cov: np.ndarray = np.eye(4, dtype=np.float32)
        self._meas_array: np.ndarray = np.zeros((2, 1), dtype=np.float32)

    # ── KF 初始化 ────────────────────────────────────

    def _init_kf(self, x: float, y: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        self._set_transition(kf, 1.0 / 30.0)
        kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        r = np.float32(self.cfg.r_noise_base)
        # 使用預初始化的矩陣以減少分配開銷
        self._kf_r_cov[0,0] = r
        self._kf_r_cov[1,1] = r
        kf.measurementNoiseCov  = self._kf_r_cov.copy()
        self._kf_q_cov[:] = 0.0
        np.fill_diagonal(self._kf_q_cov, np.float32(self.cfg.q_base))
        kf.processNoiseCov      = self._kf_q_cov.copy()
        kf.errorCovPost         = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array([[x],[y],[0.],[0.]], dtype=np.float32)
        kf.statePre  = kf.statePost.copy()
        return kf

    @staticmethod
    def _set_transition(kf: cv2.KalmanFilter, dt: float):
        dt = np.float32(max(dt, 1e-4))
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

    # ── 自適應 Q ─────────────────────────────────────

    def _compute_q(self, fx: float, fy: float) -> float:
        if self._last_smooth is None:
            return self.cfg.q_base
        dx   = fx - self._last_smooth[0]
        dy   = fy - self._last_smooth[1]
        dist = float(np.sqrt(dx*dx + dy*dy))
        self._innov_buf.append(dist)
        self.is_still = dist < self.cfg.still_threshold

        if self.is_still:
            return self.cfg.q_base * self.cfg.still_damping
        max_i  = max(self._innov_buf)
        eff    = max(dist, max_i * 0.5)
        q      = self.cfg.q_base + (eff * self.cfg.innovation_gain) ** self.cfg.speed_gain_exp
        return min(q, self.cfg.q_max)

    # ── 資料驅動 R（零相機假設）──────────────────────

    def _compute_r(self, norm_y: float = 0.5) -> float:
        """
        從量測殘差自己學每個點的真實噪音水平。
        完全不依賴相機角度或畫面位置假設。

        原理：
          殘差 = |量測值 - Kalman預測值|
          一個點若持續穩定 → 殘差小 → R 小（信任量測，少平滑）
          一個點若一直抖動 → 殘差大 → R 大（多平滑）

        用 MAD（中位數絕對偏差）估計，對突發跳點穩健：
          robust_std = median(residuals) × 1.4826

        norm_y 參數保留向後相容，僅在 r_dist_scale > 0 時才參考
        （預設 r_dist_scale=0，完全不用畫面位置）
        """
        # 殘差不足時用基準 R
        if len(self._residual_buf) < 3:
            r = self.cfg.r_noise_base
        else:
            # MAD → robust std（對異常值穩健）
            med_res = float(np.median(self._residual_buf))
            robust_std = med_res * 1.4826
            # 殘差變異映射到 R
            r = self.cfg.r_noise_base + (robust_std ** 2) * self.cfg.r_residual_gain
            r = float(np.clip(r, self.cfg.r_noise_base, self.cfg.r_noise_max))

        # 相容性：僅當使用者主動開啟畫面位置假設時才疊加
        if self.cfg.r_dist_scale > 0.0:
            factor = 1.0 + self.cfg.r_dist_scale * (1.0 - norm_y) ** 2
            r = min(r * factor, self.cfg.r_noise_max)

        self.current_r = r
        return r

    def _record_residual(self, meas_x: float, meas_y: float):
        """記錄量測值與 Kalman 預測值的殘差（供下一幀估 R）"""
        if self._kf is None:
            return
        pred_x = float(self._kf.statePre[0][0])
        pred_y = float(self._kf.statePre[1][0])
        res = float(np.sqrt((meas_x - pred_x)**2 + (meas_y - pred_y)**2))
        self._residual_buf.append(res)

    # ── 主更新 ───────────────────────────────────────

    def update(self, raw_x: float, raw_y: float,
               norm_y: float = 0.5) -> Tuple[float, float]:
        # ── 漏洞修正：NaN / Inf 守衛 ──
        # YOLOv8 在信心極低時可能輸出 NaN，會永久污染 Kalman 狀態
        if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
            if self.initialized:
                return self.predict_only()   # 當作偵測丟失
            return 0.0, 0.0

        # ── 漏洞修正：norm_y clamp 到 [0,1] ──
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        now = time.time()
        dt  = float(np.clip(now - self._last_time, 1e-4, 0.5)) \
              if self._last_time else 1.0 / 30.0
        self._last_time  = now
        self.current_dt  = dt

        # ── 第一次初始化（用原始值，三級濾波也初始化）──
        if not self.initialized:
            self._med_x.filter(raw_x, True); self._med_y.filter(raw_y, True)
            self._ema_x.filter(raw_x, True); self._ema_y.filter(raw_y, True)
            self._cf_x.filter(raw_x, dt, True); self._cf_y.filter(raw_y, dt, True)
            self._kf            = self._init_kf(raw_x, raw_y)
            self.initialized    = True
            self._last_smooth   = (raw_x, raw_y)
            self.missing_frames = 0
            return raw_x, raw_y

        # ── [Fix-B] 速度感知閘門（用原始值判斷）────────
        # 必須在三級濾波之前，否則濾波器壓掉位移
        # 導致 Kalman 速度狀態學不到，閘門永遠是靜止尺寸
        is_out, reason = self._gate.is_outlier(
            raw_x, raw_y, self._last_smooth
        )
        if is_out:
            self.outlier_reason = reason
            self._set_transition(self._kf, dt)
            pred = self._kf.predict()
            sx, sy = float(pred[0][0]), float(pred[1][0])
            self._last_smooth = (sx, sy)
            return sx, sy
        self.outlier_reason = ""

        # ── 靜止判定（用 Kalman 速度狀態，比位移差更準）
        vx = float(self._kf.statePost[2][0])
        vy = float(self._kf.statePost[3][0])
        speed = float(np.sqrt(vx*vx + vy*vy))
        # 同時參考原始位移（速度剛建立時的過渡期）
        if self._last_smooth:
            raw_dist = float(np.sqrt(
                (raw_x - self._last_smooth[0])**2 +
                (raw_y - self._last_smooth[1])**2
            ))
        else:
            raw_dist = 0.0
        is_still = (speed < self.cfg.still_threshold and
                    raw_dist < self.cfg.still_threshold * 1.5)
        self.is_still = is_still

        # ── Level 1：中位數（靜止去跳點，移動直通）──
        m_x = self._med_x.filter(raw_x, is_still)
        m_y = self._med_y.filter(raw_y, is_still)

        # ── Level 2：動態 EWMA（去高頻）─────────────
        e_x = self._ema_x.filter(m_x, is_still)
        e_y = self._ema_y.filter(m_y, is_still)

        # ── Level 3a：動態互補濾波 ───────────────────
        f_x = self._cf_x.filter(e_x, dt, is_still)
        f_y = self._cf_y.filter(e_y, dt, is_still)

        # ── 自適應 Q ──────────────────────────────────
        q = self._compute_q(f_x, f_y)
        self.current_q = q

        self._set_transition(self._kf, dt)
        # 使用預初始化矩陣以減少分配開銷
        self._kf_q_cov[:] = 0.0
        np.fill_diagonal(self._kf_q_cov, np.float32(q))
        self._kf.processNoiseCov = self._kf_q_cov.copy()
        if is_still:
            d = np.float32(self.cfg.velocity_decay * self.cfg.still_damping * 8)
            d = min(d, np.float32(self.cfg.velocity_decay))
            self._kf.statePost[2][0] = float(np.float32(self._kf.statePost[2][0]) * d)
            self._kf.statePost[3][0] = float(np.float32(self._kf.statePost[3][0]) * d)

        # ── Kalman predict ────────────────────────────
        self._kf.predict()   # 產生 statePre（預測值）

        # ── 資料驅動 R：predict 後記錄殘差，再算 R ─────
        # 殘差用「原始量測 raw」對預測值算，反映 YOLOv8 真實噪音
        # （用濾波後的 f 會看不出原始抖動，因為三級濾波已壓平）
        self._record_residual(raw_x, raw_y)
        r = self._compute_r(norm_y)
        # 使用預初始化矩陣以減少分配開銷
        self._kf_r_cov[0,0] = r
        self._kf_r_cov[1,1] = r
        self._kf.measurementNoiseCov = self._kf_r_cov.copy()

        # ── Kalman correct ────────────────────────────
        # 使用預初始化矩陣以減少分配開銷
        self._meas_array[0, 0] = np.float32(f_x)
        self._meas_array[1, 0] = np.float32(f_y)
        smoothed = self._kf.correct(self._meas_array)

        sx = float(smoothed[0][0])
        sy = float(smoothed[1][0])
        self._last_smooth   = (sx, sy)
        self.missing_frames = 0
        return sx, sy

    # ── 遮擋預測 ─────────────────────────────────────

    def predict_only(self) -> Tuple[float, float]:
        self.missing_frames += 1
        if not self.initialized:
            return 0.0, 0.0
        now = time.time()
        dt  = float(np.clip(now - self._last_time, 1e-4, 0.2)) \
              if self._last_time else 1.0/30.0
        self._last_time = now
        self._set_transition(self._kf, dt)
        d = np.float32(self.cfg.velocity_decay)
        self._kf.statePost[2][0] = float(np.float32(self._kf.statePost[2][0]) * d)
        self._kf.statePost[3][0] = float(np.float32(self._kf.statePost[3][0]) * d)
        pred = self._kf.predict()
        sx, sy = float(pred[0][0]), float(pred[1][0])
        self._last_smooth = (sx, sy)
        return sx, sy

    def reset(self):
        self._kf            = None
        self.initialized    = False
        self.missing_frames = 0
        self._innov_buf.clear()
        self._residual_buf.clear()
        self._last_smooth   = None
        self._last_time     = None
        self._med_x.reset(); self._med_y.reset()
        self._ema_x.reset(); self._ema_y.reset()
        self._cf_x.reset();  self._cf_y.reset()
        self._gate.reset()
        self.is_still       = True
        self.outlier_reason = ""

    @property
    def alive(self) -> bool:
        return self.missing_frames < self.cfg.max_missing

    @property
    def state_info(self) -> str:
        s = "still" if self.is_still else "move"
        if self.outlier_reason: s = f"gate[{self.outlier_reason}]"
        if not self.alive:      s = "lost"
        return (f"Q={self.current_q:.0f} "
                f"R={self.current_r:.1f} "
                f"dt={self.current_dt*1000:.0f}ms [{s}]")


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
    print("=== AdaptiveKalmanPoint v3 Unit Tests ===\n")
    cfg = AKFConfig()
    rng = np.random.default_rng(42)

    # ── Test 1：靜止收斂 & 抖動抑制 ─────────────────
    print("Test 1: 靜止收斂 + 抖動抑制")
    kp = AdaptiveKalmanPoint(cfg)
    positions = []
    for _ in range(60):
        nx = 100.0 + rng.normal(0, 4)   # ±4px 噪音
        ny = 200.0 + rng.normal(0, 4)
        sx, sy = kp.update(nx, ny, 0.7)
        positions.append((sx, sy))
        time.sleep(0.001)
    # 最後 20 幀的標準差（靜止穩定性）
    arr = np.array(positions[-20:])
    std_x = float(np.std(arr[:,0]))
    std_y = float(np.std(arr[:,1]))
    err   = float(np.sqrt((sx-100)**2 + (sy-200)**2))
    print(f"  最終位置=({sx:.2f},{sy:.2f})  誤差={err:.2f}px")
    print(f"  最後20幀標準差 σx={std_x:.3f}px  σy={std_y:.3f}px  (目標<1.5px)")
    print(f"  {'✅ PASS' if std_x<1.5 and std_y<1.5 and err<5 else '❌ FAIL'}\n")

    # ── Test 2：連續移動響應（每幀 20px，無卡住）────
    print("Test 2: 連續移動響應（20px/幀，10幀）")
    kp2 = AdaptiveKalmanPoint(cfg)
    for _ in range(20):
        kp2.update(100.0, 200.0, 0.7); time.sleep(0.033)
    tx = 100.0
    max_lag = 0.0
    for i in range(10):
        tx += 20.0
        sx, sy = kp2.update(tx, 200.0, 0.7); time.sleep(0.033)
        lag = abs(sx - tx)
        max_lag = max(max_lag, lag)
        print(f"  f{i+1:02d}: tgt={tx:.0f} sm={sx:.1f} lag={lag:.1f}px Q={kp2.current_q:.0f}")
    print(f"  最大 lag={max_lag:.1f}px  {'✅ PASS' if max_lag<45 else '❌ FAIL'}\n")

    # ── Test 3：[Fix-B] 快速移動不卡住 ──────────────
    print("Test 3: [Fix-B] 快速移動不卡住（位移超過舊閘門 100px）")
    kp3 = AdaptiveKalmanPoint(cfg)
    # 先建立速度狀態（每幀 30px）
    x = 100.0
    for _ in range(15):
        x += 30.0
        kp3.update(x, 200.0, 0.7); time.sleep(0.033)
    # 繼續快速移動，觀察是否被舊閘門擋住
    before = kp3._last_smooth[0]
    x += 30.0
    sx, sy = kp3.update(x, 200.0, 0.7); time.sleep(0.033)
    moved  = abs(sx - before)
    frozen = moved < 3.0   # 若移動<3px = 卡住了
    print(f"  移動前={before:.1f}  移動後={sx:.1f}  位移={moved:.1f}px")
    print(f"  outlier_reason='{kp3.outlier_reason}'")
    print(f"  {'✅ PASS (不卡住)' if not frozen else '❌ FAIL (仍卡住)'}\n")

    # ── Test 4：異常值閘門（誤偵測跳點）────────────
    print("Test 4: 異常值閘門（靜止時出現遠跳點）")
    kp4 = AdaptiveKalmanPoint(cfg)
    for _ in range(25):
        kp4.update(200.0, 300.0, 0.5); time.sleep(0.001)
    bx, by = kp4._last_smooth
    # 靜止時突然出現 200px 遠的跳點
    kp4.update(200.0+200, 300.0, 0.5)
    drift = abs(kp4._last_smooth[0] - bx)
    print(f"  靜止基準={bx:.1f}  跳點後={kp4._last_smooth[0]:.1f}  飄移={drift:.1f}px")
    print(f"  gate_reason='{kp4.outlier_reason}'")
    print(f"  {'✅ PASS' if drift < 20 else '❌ FAIL'}\n")

    # ── Test 5：資料驅動 R（零相機假設）─────────────
    print("Test 5: 資料驅動 R（穩定點 R 小，抖動點 R 大）")
    rng5 = np.random.default_rng(7)
    # 穩定點：殘差小
    kp_stable = AdaptiveKalmanPoint(cfg)
    for _ in range(25):
        kp_stable.update(400.0 + rng5.normal(0, 1.5), 300.0, 0.5)
        time.sleep(0.001)
    r_stable = kp_stable.current_r

    # 抖動點：殘差大
    kp_jitter = AdaptiveKalmanPoint(cfg)
    for _ in range(25):
        kp_jitter.update(400.0 + rng5.normal(0, 10.0), 300.0, 0.5)
        time.sleep(0.001)
    r_jitter = kp_jitter.current_r

    print(f"  穩定點 R={r_stable:.2f}（殘差小→信任量測）")
    print(f"  抖動點 R={r_jitter:.2f}（殘差大→多平滑）")
    print(f"  比值={r_jitter/r_stable:.2f}x  （完全由資料決定，無相機假設）")
    print(f"  {'✅ PASS' if r_jitter > r_stable * 1.3 else '❌ FAIL'}\n")

    # ── Test 6：遮擋預測不飄移 ───────────────────────
    print("Test 6: 遮擋預測")
    kp6 = AdaptiveKalmanPoint(cfg)
    for _ in range(30):
        kp6.update(300.0, 400.0, 0.6); time.sleep(0.001)
    bx, by = kp6._last_smooth
    for _ in range(10):
        kp6.predict_only(); time.sleep(0.033)
    drift = float(np.sqrt(
        (kp6._last_smooth[0]-bx)**2 + (kp6._last_smooth[1]-by)**2
    ))
    print(f"  基準=({bx:.1f},{by:.1f})  10幀後=({kp6._last_smooth[0]:.1f},{kp6._last_smooth[1]:.1f})")
    print(f"  飄移={drift:.2f}px  {'✅ PASS' if drift < 15 else '❌ FAIL'}\n")

    # ── Test 7：三級濾波抖動對比 ─────────────────────
    print("Test 7: 三級濾波效果對比（輸入±8px噪音）")
    raw_vals, smooth_vals = [], []
    kp7 = AdaptiveKalmanPoint(cfg)
    for i in range(80):
        noise = rng.normal(0, 8)
        raw   = 400.0 + noise
        sx, _ = kp7.update(raw, 300.0, 0.6)
        if i >= 20:
            raw_vals.append(raw)
            smooth_vals.append(sx)
        time.sleep(0.001)
    raw_std    = float(np.std(raw_vals))
    smooth_std = float(np.std(smooth_vals))
    reduction  = (1 - smooth_std / raw_std) * 100
    print(f"  輸入 σ={raw_std:.2f}px  輸出 σ={smooth_std:.2f}px")
    print(f"  抖動抑制率={reduction:.1f}%  {'✅ PASS' if reduction > 80 else '❌ FAIL'}\n")

    print("=== 所有測試完成 ===")


if __name__ == "__main__":
    _run_tests()
