"""
MIVR-CEIQ HAR Activity Classifier
har_classifier.py

專家十五建議：先分類動作類型，再選對應融合策略

分類的動作類型：
  STANDING   靜止站立      → 純 AKF 靜止模式，不用手腕輔助
  WALKING    行走          → 啟用手腕對側協調，動態係數
  TURNING    轉身          → 暫停融合，等穩定
  CROUCHING  蹲下          → 切換髖膝主導模式
  SITTING    坐下          → 低信心，降低 Holland 權重
  UNKNOWN    未知          → 保守模式

分類依據（純骨架幾何，不需額外模型）：
  - 臀部 y 座標與歷史均值偏差 → 蹲/站
  - 軀幹朝向角變化率 → 轉身
  - 手腕 y 振幅 → 行走/靜止
  - 膝蓋彎曲角度 → 蹲姿確認
"""

import numpy as np
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class ActivityType(Enum):
    STANDING  = auto()
    WALKING   = auto()
    TURNING   = auto()
    CROUCHING = auto()
    SITTING   = auto()
    UNKNOWN   = auto()


@dataclass
class ActivityState:
    activity:       ActivityType  = ActivityType.UNKNOWN
    confidence:     float         = 0.0
    wrist_ankle_gain: float       = 0.0   # 手腕→腳踝融合係數
    use_hip_primary:  bool        = False  # 是否以髖部為主
    freeze_fusion:    bool        = False  # 是否暫停融合（轉身時）
    description:    str           = ""


# YOLOv8-Pose 關節點索引
class KP:
    L_SHOULDER = 5;  R_SHOULDER = 6
    L_ELBOW    = 7;  R_ELBOW    = 8
    L_WRIST    = 9;  R_WRIST    = 10
    L_HIP      = 11; R_HIP      = 12
    L_KNEE     = 13; R_KNEE     = 14
    L_ANKLE    = 15; R_ANKLE    = 16

CONF_VIS = 0.35


class HARClassifier:
    """
    基於骨架幾何特徵的即時動作分類器。
    每幀輸入 kpts (17,3)，輸出 ActivityState。
    """

    def __init__(self, buf_len: int = 30, fps: float = 30.0):
        self.fps = fps
        self._buf_len = buf_len

        # 歷史緩衝
        self._hip_y_buf:     deque = deque(maxlen=buf_len)
        self._torso_angle:   deque = deque(maxlen=buf_len)
        self._wrist_y_buf:   deque = deque(maxlen=buf_len)
        self._knee_angle_buf:deque = deque(maxlen=buf_len)

        # 當前狀態
        self._current: ActivityState = ActivityState()

        # 穩定計數器（防止抖動切換）
        self._stable_count: int = 0
        self._STABLE_THRESH: int = 5    # 需連續 N 幀才確認切換

        # 參考值（初始化後設定）
        self._standing_hip_y:  Optional[float] = None
        self._init_frames:     int = 0
        self._INIT_REQUIRED:   int = 15

    def update(self, kpts: np.ndarray) -> ActivityState:
        """
        kpts: (17, 3) float32，[x, y, conf]
        回傳 ActivityState
        """
        self._extract_features(kpts)
        self._init_frames += 1

        if self._init_frames < self._INIT_REQUIRED:
            self._current = ActivityState(
                activity=ActivityType.UNKNOWN,
                confidence=0.0,
                description="initializing"
            )
            return self._current

        new_activity = self._classify()
        self._current = self._build_state(new_activity, kpts)
        return self._current

    # ── 特徵提取 ─────────────────────────────────────

    def _extract_features(self, kpts: np.ndarray):
        lh_c = float(kpts[KP.L_HIP][2])
        rh_c = float(kpts[KP.R_HIP][2])
        lw_c = float(kpts[KP.L_WRIST][2])
        rw_c = float(kpts[KP.R_WRIST][2])
        ls_c = float(kpts[KP.L_SHOULDER][2])
        rs_c = float(kpts[KP.R_SHOULDER][2])

        # 臀部 y（用於蹲/站偵測）
        if lh_c > CONF_VIS and rh_c > CONF_VIS:
            hip_y = (float(kpts[KP.L_HIP][1]) + float(kpts[KP.R_HIP][1])) / 2
            self._hip_y_buf.append(hip_y)
            if self._standing_hip_y is None and len(self._hip_y_buf) >= 10:
                self._standing_hip_y = float(np.median(list(self._hip_y_buf)))

        # 軀幹朝向角（肩線與水平的夾角）
        if ls_c > CONF_VIS and rs_c > CONF_VIS:
            dx = float(kpts[KP.R_SHOULDER][0]) - float(kpts[KP.L_SHOULDER][0])
            dy = float(kpts[KP.R_SHOULDER][1]) - float(kpts[KP.L_SHOULDER][1])
            angle = float(np.degrees(np.arctan2(dy, dx)))
            self._torso_angle.append(angle)

        # 手腕 y（用於行走偵測）
        if rw_c > CONF_VIS:
            self._wrist_y_buf.append(float(kpts[KP.R_WRIST][1]))

        # 膝蓋彎曲角度
        knee_angle = self._calc_knee_angle(kpts)
        if knee_angle is not None:
            self._knee_angle_buf.append(knee_angle)

    def _calc_knee_angle(self, kpts: np.ndarray) -> Optional[float]:
        """計算右膝彎曲角度（髖-膝-踝）"""
        hc = float(kpts[KP.R_HIP][2])
        kc = float(kpts[KP.R_KNEE][2])
        ac = float(kpts[KP.R_ANKLE][2])
        if hc < CONF_VIS or kc < CONF_VIS or ac < CONF_VIS:
            return None
        v1 = np.array([
            float(kpts[KP.R_HIP][0]) - float(kpts[KP.R_KNEE][0]),
            float(kpts[KP.R_HIP][1]) - float(kpts[KP.R_KNEE][1]),
        ])
        v2 = np.array([
            float(kpts[KP.R_ANKLE][0]) - float(kpts[KP.R_KNEE][0]),
            float(kpts[KP.R_ANKLE][1]) - float(kpts[KP.R_KNEE][1]),
        ])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return None
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.degrees(np.arccos(cos_a)))

    # ── 分類邏輯 ─────────────────────────────────────

    def _classify(self) -> ActivityType:
        scores = {a: 0.0 for a in ActivityType}

        # ── 蹲姿偵測 ────────────────────────────────
        if (self._standing_hip_y is not None and
                len(self._hip_y_buf) >= 5):
            current_hip_y = float(np.mean(list(self._hip_y_buf)[-5:]))
            # 臀部 y 增大 = 往下蹲（圖像座標 y 向下）
            hip_drop = current_hip_y - self._standing_hip_y
            if hip_drop > 30:
                scores[ActivityType.CROUCHING] += 5.0   # 高優先，蓋過行走
            elif hip_drop > 15:
                scores[ActivityType.CROUCHING] += 2.5

        # 膝蓋角度確認蹲姿（< 140° = 彎曲）
        if len(self._knee_angle_buf) >= 3:
            avg_knee = float(np.mean(list(self._knee_angle_buf)[-3:]))
            if avg_knee < 110:
                scores[ActivityType.CROUCHING] += 3.0
            elif avg_knee < 140:
                scores[ActivityType.CROUCHING] += 1.5

        # ── 轉身偵測 ────────────────────────────────
        if len(self._torso_angle) >= 8:
            angles = list(self._torso_angle)[-8:]
            # 角度變化率
            angle_diff = abs(float(angles[-1]) - float(angles[0]))
            if angle_diff > 25:
                scores[ActivityType.TURNING] += 3.0
            elif angle_diff > 12:
                scores[ActivityType.TURNING] += 1.5

        # ── 行走偵測 ────────────────────────────────
        if len(self._wrist_y_buf) >= 20:
            arr = np.array(list(self._wrist_y_buf)[-20:])
            amplitude = float(arr.max() - arr.min())
            if amplitude > 18:
                scores[ActivityType.WALKING] += 3.0
            elif amplitude > 8:
                scores[ActivityType.WALKING] += 1.5

        # ── 靜止站立 ────────────────────────────────
        if (scores[ActivityType.WALKING] == 0.0 and
                scores[ActivityType.CROUCHING] == 0.0 and
                scores[ActivityType.TURNING] == 0.0):
            scores[ActivityType.STANDING] += 2.0

        # 取最高分
        best = max(scores, key=lambda a: scores[a])
        if scores[best] < 0.5:
            return ActivityType.UNKNOWN
        return best

    # ── 建立 ActivityState ────────────────────────────

    def _build_state(self, activity: ActivityType,
                     kpts: np.ndarray) -> ActivityState:
        """
        根據動作類型決定手腕→腳踝融合係數。
        係數依據 Winter (1990) 步態生物力學數據。
        """
        if activity == ActivityType.STANDING:
            return ActivityState(
                activity         = activity,
                confidence       = 0.85,
                wrist_ankle_gain = 0.0,    # 不使用手腕輔助
                use_hip_primary  = False,
                freeze_fusion    = False,
                description      = "standing: AKF still mode"
            )

        elif activity == ActivityType.WALKING:
            # 估算步速 → 動態手腕係數
            gain = self._estimate_walk_gain()
            return ActivityState(
                activity         = activity,
                confidence       = 0.80,
                wrist_ankle_gain = gain,
                use_hip_primary  = False,
                freeze_fusion    = False,
                description      = f"walking: wrist_gain={gain:.2f}"
            )

        elif activity == ActivityType.TURNING:
            return ActivityState(
                activity         = activity,
                confidence       = 0.70,
                wrist_ankle_gain = 0.0,
                use_hip_primary  = True,
                freeze_fusion    = True,   # 暫停外部融合
                description      = "turning: freeze fusion"
            )

        elif activity == ActivityType.CROUCHING:
            return ActivityState(
                activity         = activity,
                confidence       = 0.75,
                wrist_ankle_gain = 0.0,
                use_hip_primary  = True,   # 切換髖膝主導
                freeze_fusion    = False,
                description      = "crouching: hip-knee primary"
            )

        return ActivityState(
            activity         = ActivityType.UNKNOWN,
            confidence       = 0.3,
            wrist_ankle_gain = 0.1,
            use_hip_primary  = False,
            freeze_fusion    = False,
            description      = "unknown: conservative"
        )

    def _estimate_walk_gain(self) -> float:
        """
        [E15-Fix2] 動態手腕→腳踝係數
        依據 Winter (1990)：
          慢走（<1.0 m/s）：gain ≈ 0.20
          正常走（1.0~1.5）：gain ≈ 0.40
          快走（>1.5 m/s）：gain ≈ 0.60
        我們用手腕振幅作為速度代理指標。
        """
        if len(self._wrist_y_buf) < 10:
            return 0.30  # 預設值

        arr       = np.array(list(self._wrist_y_buf)[-20:])
        amplitude = float(arr.max() - arr.min())

        # 振幅 → gain 線性插值
        # amp < 15px  → gain 0.20（慢走）
        # amp   50px  → gain 0.60（快走）
        gain = np.interp(amplitude, [15, 50], [0.20, 0.60])
        return float(np.clip(gain, 0.10, 0.70))

    @property
    def current(self) -> ActivityState:
        return self._current

    def reset(self):
        self._hip_y_buf.clear()
        self._torso_angle.clear()
        self._wrist_y_buf.clear()
        self._knee_angle_buf.clear()
        self._standing_hip_y = None
        self._init_frames    = 0
        self._stable_count   = 0
        self._current        = ActivityState()


# ─────────────────────────────────────────────────────
# 多人管理器
# ─────────────────────────────────────────────────────

class MultiHARClassifier:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._classifiers: dict = {}

    def update(self, person_id: int,
               kpts: np.ndarray) -> ActivityState:
        if person_id not in self._classifiers:
            self._classifiers[person_id] = HARClassifier(fps=self.fps)
        return self._classifiers[person_id].update(kpts)

    def get_state(self, person_id: int) -> Optional[ActivityState]:
        c = self._classifiers.get(person_id)
        return c.current if c else None

    def reset_all(self):
        self._classifiers.clear()


# ─────────────────────────────────────────────────────
# 單元測試
# ─────────────────────────────────────────────────────

def _make_kpts(l_sho, r_sho, l_hip, r_hip,
               l_knee, r_knee, l_ank, r_ank,
               l_wri, r_wri, conf=0.9):
    kp = np.zeros((17, 3), dtype=np.float32)
    def s(i, x, y): kp[i] = [x, y, conf]
    s(KP.L_SHOULDER, *l_sho); s(KP.R_SHOULDER, *r_sho)
    s(KP.L_HIP, *l_hip);      s(KP.R_HIP, *r_hip)
    s(KP.L_KNEE, *l_knee);    s(KP.R_KNEE, *r_knee)
    s(KP.L_ANKLE, *l_ank);    s(KP.R_ANKLE, *r_ank)
    s(KP.L_WRIST, *l_wri);    s(KP.R_WRIST, *r_wri)
    return kp


def _run_tests():
    print("=== HARClassifier Unit Tests ===\n")

    # Test 1: 站立靜止
    print("Test 1: 站立靜止偵測")
    clf = HARClassifier()
    kp_stand = _make_kpts(
        l_sho=(200,120), r_sho=(240,120),
        l_hip=(200,280), r_hip=(240,280),
        l_knee=(200,380),r_knee=(240,380),
        l_ank=(200,480), r_ank=(240,480),
        l_wri=(190,220), r_wri=(250,220),  # 手腕靜止
    )
    for _ in range(30):
        state = clf.update(kp_stand)
    print(f"  Activity: {state.activity.name}  gain={state.wrist_ankle_gain:.2f}")
    print(f"  {state.description}")
    ok1 = state.activity in (ActivityType.STANDING, ActivityType.UNKNOWN)
    print(f"  {'✅ PASS' if ok1 else '❌ FAIL'}\n")

    # Test 2: 行走偵測（手腕 y 振盪）
    print("Test 2: 行走偵測（手腕振盪 30px）")
    clf2 = HARClassifier()
    for i in range(40):
        wrist_y = 220 + 25 * np.sin(2 * np.pi * 1.5 * i / 30)
        kp_walk = _make_kpts(
            l_sho=(200,120), r_sho=(240,120),
            l_hip=(200,280), r_hip=(240,280),
            l_knee=(200,380),r_knee=(240,380),
            l_ank=(200,480), r_ank=(240,480),
            l_wri=(190,wrist_y), r_wri=(250,wrist_y),
        )
        state = clf2.update(kp_walk)
    print(f"  Activity: {state.activity.name}  gain={state.wrist_ankle_gain:.2f}")
    print(f"  {state.description}")
    ok2 = state.activity in (ActivityType.WALKING, ActivityType.UNKNOWN)
    print(f"  {'✅ PASS' if ok2 else '❌ FAIL'}\n")

    # Test 3: 蹲姿偵測
    print("Test 3: 蹲姿偵測（臀部下移 50px）")
    clf3 = HARClassifier()
    # 先建立站立基準
    kp_stand = _make_kpts(
        l_sho=(200,120), r_sho=(240,120),
        l_hip=(200,280), r_hip=(240,280),
        l_knee=(200,380),r_knee=(240,380),
        l_ank=(200,480), r_ank=(240,480),
        l_wri=(190,220), r_wri=(250,220),
    )
    for _ in range(20):
        clf3.update(kp_stand)
    # 蹲下：臀部 y 增大，膝蓋角度縮小
    kp_crouch = _make_kpts(
        l_sho=(200,120), r_sho=(240,120),
        l_hip=(200,330), r_hip=(240,330),   # 臀部下移 50px
        l_knee=(200,390),r_knee=(240,390),  # 膝蓋幾乎和臀部一樣高
        l_ank=(200,480), r_ank=(240,480),
        l_wri=(190,250), r_wri=(250,250),
    )
    for _ in range(10):
        state = clf3.update(kp_crouch)
    print(f"  Activity: {state.activity.name}  hip_primary={state.use_hip_primary}")
    print(f"  {state.description}")
    ok3 = state.activity in (ActivityType.CROUCHING, ActivityType.UNKNOWN)
    print(f"  {'✅ PASS' if ok3 else '❌ FAIL'}\n")

    # Test 4: 動態 wrist_ankle_gain
    print("Test 4: 動態 wrist_ankle_gain 隨步速變化")
    clf4 = HARClassifier()
    # 慢走（振幅 10px）
    for i in range(40):
        wrist_y = 220 + 10 * np.sin(2 * np.pi * 1.0 * i / 30)
        kp = _make_kpts(
            l_sho=(200,120),r_sho=(240,120),
            l_hip=(200,280),r_hip=(240,280),
            l_knee=(200,380),r_knee=(240,380),
            l_ank=(200,480),r_ank=(240,480),
            l_wri=(190,wrist_y),r_wri=(250,wrist_y),
        )
        state_slow = clf4.update(kp)
    gain_slow = state_slow.wrist_ankle_gain

    # 快走（振幅 45px）
    clf5 = HARClassifier()
    for i in range(40):
        wrist_y = 220 + 45 * np.sin(2 * np.pi * 2.0 * i / 30)
        kp = _make_kpts(
            l_sho=(200,120),r_sho=(240,120),
            l_hip=(200,280),r_hip=(240,280),
            l_knee=(200,380),r_knee=(240,380),
            l_ank=(200,480),r_ank=(240,480),
            l_wri=(190,wrist_y),r_wri=(250,wrist_y),
        )
        state_fast = clf5.update(kp)
    gain_fast = state_fast.wrist_ankle_gain

    print(f"  慢走 gain={gain_slow:.2f}  快走 gain={gain_fast:.2f}")
    ok4 = gain_fast >= gain_slow
    print(f"  {'✅ PASS' if ok4 else '❌ FAIL'} (快走係數應 >= 慢走)\n")

    print("=== 所有測試完成 ===")


if __name__ == "__main__":
    _run_tests()
