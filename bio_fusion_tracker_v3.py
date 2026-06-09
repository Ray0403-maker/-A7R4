"""
MIVR-CEIQ Biomechanical Fusion Tracker
bio_fusion_tracker_v3.py (imports adaptive_kalman_v4)

核心思路：
  人體各關節點之間存在運動學關聯。
  當腳踝點信心值低或被遮擋時，
  利用手腕/手肘的可見性與步態節律輔助估算腳部位置。

三層融合機制：
  Layer 1 - 直接觀測：腳踝信心值高 → 直接使用 AKF 平滑
  Layer 2 - 上半身輔助：腳踝低信心 → 用手臂擺動相位估算腳步節律
  Layer 3 - 遮擋預測：腳踝消失 → 用臀部/膝蓋外插 + 步態週期預測

步態生物力學依據：
  - 右手擺前 ≈ 左腳在前（對側協調）
  - 手臂擺動頻率 ≈ 步頻（1:1 關係）
  - 臀部側向位移 → 重心預測 → 支撐腳估算
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import time

# 引入自適應卡爾曼
try:
    from adaptive_kalman_v4 import AdaptiveKalmanPoint, AKFConfig
except ImportError:
    raise ImportError("需要 adaptive_kalman_v4.py，請確認在同一目錄")


# ─────────────────────────────────────────────────────
# YOLOv8-Pose COCO 關節點索引
# ─────────────────────────────────────────────────────

KP = {
    "nose":       0,
    "l_eye": 1, "r_eye": 2,
    "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow":    7,  "r_elbow":    8,
    "l_wrist":    9,  "r_wrist":   10,
    "l_hip":     11,  "r_hip":     12,
    "l_knee":    13,  "r_knee":    14,
    "l_ankle":   15,  "r_ankle":   16,
}

# 手臂點（可見度通常高）
ARM_KPS  = [KP["l_wrist"], KP["r_wrist"],
            KP["l_elbow"], KP["r_elbow"],
            KP["l_shoulder"], KP["r_shoulder"]]

# 軀幹點（最穩定）
TORSO_KPS = [KP["l_hip"], KP["r_hip"],
             KP["l_shoulder"], KP["r_shoulder"]]

# 下肢點
LEG_KPS  = [KP["l_knee"], KP["r_knee"],
            KP["l_ankle"], KP["r_ankle"]]

CONF_VISIBLE = 0.4    # 視為可見的信心閾值
CONF_LOW     = 0.25   # 低信心但仍參考


# ─────────────────────────────────────────────────────
# 步態節律追蹤器
# ─────────────────────────────────────────────────────

class GaitRhythmTracker:
    """
    追蹤手腕的垂直位移節律，推算腳步週期。
    生物力學：手臂擺動與腳步呈 1:1 對側協調。
    右手腕往前（y 減小）→ 左腳應在前方（y 也減小）
    """

    def __init__(self, buf_len: int = 60):
        # 手腕 y 座標歷史（正規化，0~1）
        self._l_wrist_y: deque = deque(maxlen=buf_len)
        self._r_wrist_y: deque = deque(maxlen=buf_len)
        self._timestamps: deque = deque(maxlen=buf_len)

        # 估算的步頻（Hz）
        self.stride_freq: float = 0.0

        # 當前步態相位（0~2π）
        self.l_ankle_phase: float = 0.0
        self.r_ankle_phase: float = np.pi   # 對側，差半個週期

        # 手腕擺動幅度（像素，用來判斷是否在行走）
        self.wrist_amplitude: float = 0.0
        self.is_walking: bool = False
        self.WALK_AMPLITUDE_THRESH = 8.0    # 像素

        self._last_update = time.time()

    def update(self, kpts: np.ndarray, img_h: int):
        """kpts: shape (17, 3) [x, y, conf]"""
        now = time.time()
        self._timestamps.append(now)

        lw_conf = float(kpts[KP["l_wrist"]][2])
        rw_conf = float(kpts[KP["r_wrist"]][2])

        if lw_conf > CONF_VISIBLE:
            self._l_wrist_y.append(float(kpts[KP["l_wrist"]][1]))
        if rw_conf > CONF_VISIBLE:
            self._r_wrist_y.append(float(kpts[KP["r_wrist"]][1]))

        # 計算擺動幅度
        if len(self._r_wrist_y) >= 10:
            arr = np.array(list(self._r_wrist_y)[-20:])
            self.wrist_amplitude = float(arr.max() - arr.min())
            self.is_walking = self.wrist_amplitude > self.WALK_AMPLITUDE_THRESH

        # 估算步頻（找手腕 y 的過零點頻率）
        if len(self._r_wrist_y) >= 30:
            self._estimate_stride_freq()

        # 更新相位
        if self.stride_freq > 0.5:
            dt = now - self._last_update
            d_phase = 2 * np.pi * self.stride_freq * dt
            self.l_ankle_phase = (self.l_ankle_phase + d_phase) % (2 * np.pi)
            self.r_ankle_phase = (self.r_ankle_phase + d_phase) % (2 * np.pi)

        self._last_update = now

    def _estimate_stride_freq(self):
        """利用自相關估算週期"""
        arr = np.array(list(self._r_wrist_y))
        arr = arr - arr.mean()
        n = len(arr)
        if n < 20:
            return

        # 自相關
        corr = np.correlate(arr, arr, mode='full')[n-1:]
        corr = corr[1:n//2]   # 去掉零延遲，只看前半段

        if len(corr) < 5:
            return

        # 找第一個局部最大值（= 週期）
        peaks = []
        for i in range(1, len(corr)-1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > 0:
                peaks.append(i)

        if peaks:
            period_frames = peaks[0] + 1
            if len(self._timestamps) >= 2:
                total_time = self._timestamps[-1] - self._timestamps[0]
                fps_est = len(self._timestamps) / max(total_time, 0.01)
                period_sec = period_frames / fps_est
                if 0.3 < period_sec < 2.0:   # 合理步頻範圍
                    self.stride_freq = 1.0 / period_sec

    def get_ankle_phase_offset(self, side: str) -> float:
        """回傳指定腳的當前相位（用於步態輔助預測）"""
        return self.l_ankle_phase if side == "L" else self.r_ankle_phase

    @property
    def confidence(self) -> float:
        """步態追蹤的可信度（0~1）"""
        if not self.is_walking or self.stride_freq < 0.5:
            return 0.0
        return min(1.0, self.wrist_amplitude / 30.0)


# ─────────────────────────────────────────────────────
# 人體比例估算器
# ─────────────────────────────────────────────────────

class BodyProportionEstimator:
    """
    根據可見的軀幹關節點推算全身比例，
    用於在腳踝不可見時從髖部外插位置。

    人體比例（統計平均）：
      髖到膝：0.245 * 身高
      膝到踝：0.246 * 身高
      肩到髖：0.288 * 身高
    """

    def __init__(self):
        self._height_history: deque = deque(maxlen=30)
        self.estimated_height: float = 0.0   # 像素
        self._hip_to_ankle_ratio: float = 0.52  # (髖到踝) / 身高

    def update(self, kpts: np.ndarray):
        """用可見關節點持續更新身高估算"""
        # 嘗試用肩-踝計算
        for s_id, a_id in [(KP["l_shoulder"], KP["l_ankle"]),
                           (KP["r_shoulder"], KP["r_ankle"])]:
            sc = float(kpts[s_id][2])
            ac = float(kpts[a_id][2])
            if sc > CONF_VISIBLE and ac > CONF_VISIBLE:
                sy = float(kpts[s_id][1])
                ay = float(kpts[a_id][1])
                seg = abs(ay - sy)
                # 肩到踝 ≈ 0.74 * 身高，所以身高 = seg / 0.74
                height_est = seg / 0.74
                self._height_history.append(height_est)

        if self._height_history:
            self.estimated_height = np.median(list(self._height_history))

    def extrapolate_ankle_from_hip(self,
                                    hip_x: float, hip_y: float,
                                    knee_x: Optional[float],
                                    knee_y: Optional[float],
                                    side: str) -> Optional[Tuple[float, float]]:
        """
        當腳踝不可見時，從髖部（+ 膝蓋）外插踝部位置。
        """
        if self.estimated_height < 50:
            return None   # 身高估算不足，不外插

        if knee_x is not None and knee_y is not None:
            # 有膝蓋：沿大腿方向外插小腿
            dx = knee_x - hip_x
            dy = knee_y - hip_y
            seg_len = np.sqrt(dx*dx + dy*dy)
            if seg_len < 10:
                return None
            # 小腿長度 ≈ 大腿長度（統計均值）
            nx = dx / seg_len
            ny = dy / seg_len
            ankle_x = knee_x + nx * seg_len * 1.05
            ankle_y = knee_y + ny * seg_len * 1.05
        else:
            # 只有髖部：直接往下外插
            ankle_y = hip_y + self.estimated_height * self._hip_to_ankle_ratio
            ankle_x = hip_x   # 假設站直
        return ankle_x, ankle_y


# ─────────────────────────────────────────────────────
# 生物力學融合追蹤器（主類別）
# ─────────────────────────────────────────────────────

@dataclass
class AnkleEstimate:
    x: float
    y: float
    confidence: float      # 最終輸出的可信度
    source: str            # "direct" | "low_conf_fused" | "extrapolated" | "predicted"
    raw_x: Optional[float] = None
    raw_y: Optional[float] = None


class BioFusionTracker:
    """
    單一學生的生物力學融合追蹤器。
    整合：
      - AdaptiveKalmanPoint（腳踝直接觀測）
      - GaitRhythmTracker（步態節律輔助）
      - BodyProportionEstimator（人體比例外插）
    """

    def __init__(self, akf_cfg: AKFConfig = None):
        cfg = akf_cfg or AKFConfig()

        # 每個腳踝各自一個 AKF
        self._akf: Dict[str, AdaptiveKalmanPoint] = {
            "L": AdaptiveKalmanPoint(cfg),
            "R": AdaptiveKalmanPoint(cfg),
        }

        # 輔助估算器
        self.gait   = GaitRhythmTracker()
        self.body   = BodyProportionEstimator()

        # 上一幀的有效腳踝位置（用於短暫遮擋插值）
        self._last_valid: Dict[str, Tuple[float, float]] = {}

        # 腳踝消失計數
        self._missing: Dict[str, int] = {"L": 0, "R": 0}
        self.MAX_MISSING_DIRECT = 3    # 超過此幀數啟用輔助估算

        # 影像尺寸（距離自適應用）
        self._img_h: int = 720
        self._img_w: int = 1280

    def process(self, kpts: np.ndarray,
                img_h: int, img_w: int) -> Dict[str, AnkleEstimate]:
        """
        主處理函式。
        kpts: shape (17, 3) [x, y, conf]（YOLOv8-Pose 輸出）
        回傳：{"L": AnkleEstimate, "R": AnkleEstimate}
        """
        # 記錄影像高度（供距離自適應 norm_y 計算）
        self._img_h = img_h
        self._img_w = img_w
        # 更新輔助估算器
        self.gait.update(kpts, img_h)
        self.body.update(kpts)

        results = {}
        for side, ank_id in [("L", KP["l_ankle"]), ("R", KP["r_ankle"])]:
            ank_conf = float(kpts[ank_id][2])
            raw_x    = float(kpts[ank_id][0])
            raw_y    = float(kpts[ank_id][1])

            est = self._estimate_ankle(
                side, ank_id, ank_conf, raw_x, raw_y, kpts
            )
            results[side] = est

        return results

    def _estimate_ankle(self, side: str, ank_id: int,
                        ank_conf: float, raw_x: float, raw_y: float,
                        kpts: np.ndarray) -> AnkleEstimate:

        # ── Layer 1：直接高信心觀測 ──────────────────
        if ank_conf >= CONF_VISIBLE:
            norm_y = raw_y / max(self._img_h, 1)   # [Bug-4] 距離自適應
            sx, sy = self._akf[side].update(raw_x, raw_y, norm_y)
            self._missing[side] = 0
            self._last_valid[side] = (sx, sy)
            return AnkleEstimate(
                x=sx, y=sy,
                confidence=min(ank_conf, 1.0),
                source="direct",
                raw_x=raw_x, raw_y=raw_y
            )

        # ── Layer 2：低信心 → 融合上半身輔助 ─────────
        if ank_conf >= CONF_LOW:
            # 取得上半身輔助信號
            arm_vote_x, arm_vote_y, arm_weight = \
                self._get_arm_vote(side, kpts)

            # 加權融合
            # 修正：手腕投票權重大幅調低（0.5 → 0.15）
            # 原因：手腕對側協調係數是統計平均值，套到個人會把
            #       準確的腳踝拉偏。腳踝即使信心中等，自身位置
            #       仍比「手腕推估」可靠，融合只做輕微修正。
            w_direct = ank_conf
            w_arm    = arm_weight * (1.0 - ank_conf) * 0.15

            total_w = w_direct + w_arm
            if total_w > 0:
                fused_x = (raw_x * w_direct + arm_vote_x * w_arm) / total_w
                fused_y = (raw_y * w_direct + arm_vote_y * w_arm) / total_w
            else:
                fused_x, fused_y = raw_x, raw_y

            norm_y = fused_y / max(self._img_h, 1)
            sx, sy = self._akf[side].update(fused_x, fused_y, norm_y)
            self._missing[side] = 0
            self._last_valid[side] = (sx, sy)
            return AnkleEstimate(
                x=sx, y=sy,
                confidence=ank_conf * 0.7,
                source="low_conf_fused",
                raw_x=raw_x, raw_y=raw_y
            )

        # ── Layer 3：完全不可見 → 外插或預測 ─────────
        self._missing[side] += 1

        # 嘗試從髖/膝外插
        hip_id  = KP["l_hip"]  if side == "L" else KP["r_hip"]
        knee_id = KP["l_knee"] if side == "L" else KP["r_knee"]

        hip_conf  = float(kpts[hip_id][2])
        knee_conf = float(kpts[knee_id][2])

        knee_xy = None
        if knee_conf > CONF_VISIBLE:
            knee_xy = (float(kpts[knee_id][0]), float(kpts[knee_id][1]))

        if hip_conf > CONF_VISIBLE:
            extrap = self.body.extrapolate_ankle_from_hip(
                float(kpts[hip_id][0]), float(kpts[hip_id][1]),
                knee_xy[0] if knee_xy else None,
                knee_xy[1] if knee_xy else None,
                side
            )
            if extrap:
                ex, ey = extrap
                norm_y = ey / max(self._img_h, 1)
                sx, sy = self._akf[side].update(ex, ey, norm_y)
                self._last_valid[side] = (sx, sy)
                return AnkleEstimate(
                    x=sx, y=sy,
                    confidence=0.35,
                    source="extrapolated"
                )

        # AKF 純預測（速度衰減）
        if self._akf[side].initialized:
            px, py = self._akf[side].predict_only()
            return AnkleEstimate(
                x=px, y=py,
                confidence=max(0.1, 0.4 - self._missing[side] * 0.05),
                source="predicted"
            )

        # 完全無法估算
        return AnkleEstimate(x=0, y=0, confidence=0.0, source="lost")

    def _get_arm_vote(self, side: str,
                      kpts: np.ndarray) -> Tuple[float, float, float]:
        """
        利用對側手腕位置估算腳踝位置（生物力學對側協調）。
        右手腕的 x 位移方向 ≈ 左腳的位移方向（對側）
        返回：(vote_x, vote_y, weight)
        """
        # 對側手腕
        opp_wrist_id = KP["r_wrist"] if side == "L" else KP["l_wrist"]
        opp_conf = float(kpts[opp_wrist_id][2])

        # 同側手腕
        same_wrist_id = KP["l_wrist"] if side == "L" else KP["r_wrist"]
        same_conf = float(kpts[same_wrist_id][2])

        # 同側髖（位置參考）
        hip_id = KP["l_hip"] if side == "L" else KP["r_hip"]
        hip_conf = float(kpts[hip_id][2])

        if hip_conf < CONF_LOW:
            return 0.0, 0.0, 0.0   # 無髖部參考，不投票

        hip_x = float(kpts[hip_id][0])
        hip_y = float(kpts[hip_id][1])

        # 用髖部 + 估算的腿長預測踝部
        if self.body.estimated_height > 50:
            pred_y = hip_y + self.body.estimated_height * 0.52
            pred_x = hip_x

            # 對側手腕的 x 偏移 → 預測腳的 x 偏移（對側協調）
            if opp_conf > CONF_VISIBLE:
                opp_x = float(kpts[opp_wrist_id][0])
                center_x = (float(kpts[KP["l_hip"]][0]) +
                             float(kpts[KP["r_hip"]][0])) / 2
                opp_offset = opp_x - center_x
                # 腳步幅度約為手臂擺幅的 0.6 倍
                pred_x = hip_x + opp_offset * 0.3

            weight = hip_conf * 0.6
            return pred_x, pred_y, weight

        return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────
# 多人管理器
# ─────────────────────────────────────────────────────

class MultiBioFusionTracker:
    """管理多個學生的 BioFusionTracker 實例"""

    def __init__(self, akf_cfg: AKFConfig = None, max_tracked: int = 32):
        self.cfg = akf_cfg
        self._trackers: Dict[int, BioFusionTracker] = {}
        # [Bug-5 修正] 記錄每個 tracker 最後活躍幀，做 LRU 清理
        self._last_seen: Dict[int, int] = {}
        self._frame_counter: int = 0
        self._max_tracked = max_tracked   # 同時追蹤上限（教室人數通常 < 32）
        self._evict_after = 90            # 90 幀（約 3 秒）未出現則清除

    def process_person(self, person_id: int,
                       kpts: np.ndarray,
                       img_h: int, img_w: int) -> Dict[str, AnkleEstimate]:
        self._frame_counter += 1
        if person_id not in self._trackers:
            self._trackers[person_id] = BioFusionTracker(self.cfg)
        self._last_seen[person_id] = self._frame_counter
        result = self._trackers[person_id].process(kpts, img_h, img_w)
        # 硬上限每次檢查（開銷極小），時間性 stale 每 30 幀檢查
        if len(self._trackers) > self._max_tracked:
            self._evict_to_cap()
        if self._frame_counter % 30 == 0:
            self._evict_stale()
        return result

    def _evict_to_cap(self):
        """[Bug-5] 超過硬上限時，立即移除最久未見者"""
        while len(self._trackers) > self._max_tracked:
            oldest = min(self._last_seen.items(), key=lambda kv: kv[1])[0]
            self._trackers.pop(oldest, None)
            self._last_seen.pop(oldest, None)

    def _evict_stale(self):
        """[Bug-5] 清除長時間未出現的 tracker，防止記憶體無限增長"""
        # 1. 移除超過 _evict_after 幀未出現的
        stale = [pid for pid, last in self._last_seen.items()
                 if self._frame_counter - last > self._evict_after]
        for pid in stale:
            self._trackers.pop(pid, None)
            self._last_seen.pop(pid, None)
        # 2. 若仍超過上限，移除最久未見的
        if len(self._trackers) > self._max_tracked:
            ordered = sorted(self._last_seen.items(), key=lambda kv: kv[1])
            n_remove = len(self._trackers) - self._max_tracked
            for pid, _ in ordered[:n_remove]:
                self._trackers.pop(pid, None)
                self._last_seen.pop(pid, None)

    def get_gait_info(self, person_id: int) -> Optional[GaitRhythmTracker]:
        t = self._trackers.get(person_id)
        return t.gait if t else None

    def reset_person(self, person_id: int):
        if person_id in self._trackers:
            del self._trackers[person_id]
            self._last_seen.pop(person_id, None)

    def reset_all(self):
        self._trackers.clear()
        self._last_seen.clear()

    @property
    def active_persons(self) -> int:
        return len(self._trackers)


# ─────────────────────────────────────────────────────
# 視覺化工具
# ─────────────────────────────────────────────────────

SOURCE_COLORS = {
    "direct":         (0,  220, 110),   # 綠：直接偵測
    "low_conf_fused": (0,  180, 255),   # 藍：低信心融合
    "extrapolated":   (0,  220, 255),   # 青：骨架外插
    "predicted":      (80, 120, 255),   # 紫：純預測
    "lost":           (60,  60,  60),   # 灰：已失去
}

SOURCE_LABELS = {
    "direct":         "D",
    "low_conf_fused": "F",
    "extrapolated":   "E",
    "predicted":      "P",
    "lost":           "X",
}


def draw_ankle_estimate(disp: np.ndarray,
                        est: AnkleEstimate,
                        person_id: int,
                        side: str,
                        sx: float, sy: float,
                        debug: bool = False,
                        kpts: Optional[np.ndarray] = None,
                        img_w: int = 1280,
                        img_h: int = 720):
    """在 display 影像上繪製腳踝估算結果"""
    if est.confidence < 0.05:
        return

    col = SOURCE_COLORS.get(est.source, (100, 100, 100))

    dax = int(est.x * sx)
    day = int(est.y * sy)

    # 外圈大小反映信心
    radius = max(6, int(11 * est.confidence))
    cv2.circle(disp, (dax, day), radius, col, -1)
    cv2.circle(disp, (dax, day), radius + 4, col, 2)

    # 來源標籤
    label = f"P{person_id}{side}[{SOURCE_LABELS[est.source]}]"
    if debug:
        label += f" c={est.confidence:.2f}"
    cv2.putText(disp, label,
                (dax + 14, day - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # Debug：顯示原始點（灰）
    if debug and est.raw_x is not None:
        rdx = int(est.raw_x * sx)
        rdy = int(est.raw_y * sy)
        cv2.circle(disp, (rdx, rdy), 4, (70, 70, 70), -1)
        cv2.line(disp, (rdx, rdy), (dax, day), (50, 50, 50), 1)

    # Debug：顯示手腕點（橙）+ 連線
    if debug and kpts is not None:
        wrist_id = KP["l_wrist"] if side == "L" else KP["r_wrist"]
        wc = float(kpts[wrist_id][2])
        if wc > CONF_LOW:
            wx = int(float(kpts[wrist_id][0]) * sx)
            wy = int(float(kpts[wrist_id][1]) * sy)
            cv2.circle(disp, (wx, wy), 6, (0, 165, 255), 2)
            # 虛線連接手腕與腳踝（輔助關係視覺化）
            _draw_dashed_line(disp, (wx, wy), (dax, day), (0, 100, 180))


def draw_gait_indicator(disp: np.ndarray,
                        gait: GaitRhythmTracker,
                        person_id: int,
                        x: int, y: int):
    """在畫面角落顯示步態節律指示器"""
    if not gait.is_walking or gait.confidence < 0.1:
        cv2.putText(disp, f"P{person_id} gait:still",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
        return

    # 步態相位圓弧
    cv2.circle(disp, (x + 10, y - 5), 8, (60, 60, 60), 1)
    angle = int(np.degrees(gait.l_ankle_phase))
    cv2.ellipse(disp, (x + 10, y - 5), (8, 8), 0, -90, -90 + angle,
                (0, 220, 110), 2)

    info = (f"P{person_id} "
            f"{gait.stride_freq:.1f}Hz "
            f"amp:{gait.wrist_amplitude:.0f}px "
            f"c:{gait.confidence:.2f}")
    cv2.putText(disp, info, (x + 24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 180, 100), 1)


def _draw_dashed_line(img, p1, p2, color, dash_len=8):
    d = np.array(p2) - np.array(p1)
    length = np.linalg.norm(d)
    if length < 1: return
    d = d / length
    drawn = 0
    draw = True
    while drawn < length:
        end = min(drawn + dash_len, length)
        if draw:
            s = (int(p1[0] + d[0]*drawn), int(p1[1] + d[1]*drawn))
            e = (int(p1[0] + d[0]*end),   int(p1[1] + d[1]*end))
            cv2.line(img, s, e, color, 1)
        drawn = end
        draw  = not draw


# ─────────────────────────────────────────────────────
# 簡易單元測試
# ─────────────────────────────────────────────────────

def _run_tests():
    print("=== BioFusionTracker Unit Tests ===\n")
    cfg = AKFConfig()

    def make_kpts(
        l_ank=None, r_ank=None,
        l_knee=None, r_knee=None,
        l_hip=None,  r_hip=None,
        l_sho=None,  r_sho=None,
        l_wri=None,  r_wri=None,
    ):
        """建立測試用 kpts (17, 3)"""
        kp = np.zeros((17, 3), dtype=np.float32)
        def set_pt(idx, pt):
            if pt: kp[idx] = [pt[0], pt[1], pt[2]]
        set_pt(KP["l_ankle"],   l_ank  or (0,0,0))
        set_pt(KP["r_ankle"],   r_ank  or (0,0,0))
        set_pt(KP["l_knee"],    l_knee or (0,0,0))
        set_pt(KP["r_knee"],    r_knee or (0,0,0))
        set_pt(KP["l_hip"],     l_hip  or (0,0,0))
        set_pt(KP["r_hip"],     r_hip  or (0,0,0))
        set_pt(KP["l_shoulder"],l_sho  or (0,0,0))
        set_pt(KP["r_shoulder"],r_sho  or (0,0,0))
        set_pt(KP["l_wrist"],   l_wri  or (0,0,0))
        set_pt(KP["r_wrist"],   r_wri  or (0,0,0))
        return kp

    tracker = BioFusionTracker(cfg)

    # Test 1: 直接偵測
    print("Test 1: 直接高信心偵測")
    kp = make_kpts(l_ank=(200,400,0.9), r_ank=(240,400,0.9))
    for _ in range(20):
        res = tracker.process(kp, 720, 1280)
    assert res["L"].source == "direct"
    print(f"  L: ({res['L'].x:.1f},{res['L'].y:.1f}) source={res['L'].source} ✅\n")

    # Test 2: 腳踝消失，用髖膝外插
    print("Test 2: 腳踝消失 → 髖膝外插")
    kp2 = make_kpts(
        l_hip=(200,250,0.9), r_hip=(240,250,0.9),
        l_knee=(200,350,0.9), r_knee=(240,350,0.9),
        l_sho=(200,100,0.9), r_sho=(240,100,0.9),
        # 腳踝消失
        l_ank=(200,400,0.0), r_ank=(240,400,0.0),
    )
    # 先讓 body estimator 學到身高
    for _ in range(10):
        tracker2 = BioFusionTracker(cfg)
        kp_with_ank = make_kpts(
            l_ank=(200,400,0.9), r_ank=(240,400,0.9),
            l_hip=(200,250,0.9), r_hip=(240,250,0.9),
            l_sho=(200,100,0.9), r_sho=(240,100,0.9),
        )
        res2 = tracker2.process(kp_with_ank, 720, 1280)
    res2 = tracker2.process(kp2, 720, 1280)
    print(f"  L: ({res2['L'].x:.1f},{res2['L'].y:.1f}) source={res2['L'].source}")
    print(f"  {'✅ PASS' if res2['L'].source in ('extrapolated','predicted','direct') else '⚠️  check'}\n")

    # Test 3: 步態節律追蹤
    print("Test 3: 步態節律偵測（手腕 y 正弦擺動 1.5Hz）")
    gt = GaitRhythmTracker()
    import time as _time
    t_start = _time.time() - 3.0   # 模擬已過去 3 秒
    for i in range(90):            # 90 幀 @ 30fps = 3 秒
        t_now = t_start + i / 30.0
        wrist_y = 400 + 30 * np.sin(2 * np.pi * 1.5 * (i / 30.0))
        kp3 = make_kpts(
            r_wri=(300.0, wrist_y, 0.9),
            l_wri=(300.0, 400 - 30 * np.sin(2 * np.pi * 1.5 * (i / 30.0)), 0.9),
        )
        # 手動注入時間戳
        gt._timestamps.append(t_now)
        gt._r_wrist_y.append(wrist_y)
        gt._l_wrist_y.append(kp3[KP["l_wrist"]][1])
        if i >= 29:
            gt._estimate_stride_freq()
            arr = np.array(list(gt._r_wrist_y)[-20:])
            gt.wrist_amplitude = float(arr.max() - arr.min())
            gt.is_walking = gt.wrist_amplitude > gt.WALK_AMPLITUDE_THRESH
    print(f"  步頻估算：{gt.stride_freq:.2f}Hz（目標 1.5Hz）")
    print(f"  手腕振幅：{gt.wrist_amplitude:.1f}px  是否行走：{gt.is_walking}")
    ok3 = abs(gt.stride_freq - 1.5) < 0.5 and gt.is_walking
    print(f"  {'✅ PASS' if ok3 else '⚠️  marginal'}\n")

    print("=== 測試完成 ===")


if __name__ == "__main__":
    _run_tests()
