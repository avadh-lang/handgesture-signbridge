"""Thin wrapper around MediaPipe Hands (legacy solutions API).

Keeps all the MediaPipe-specific bits in one place so the rest of the code
only ever deals with plain landmark lists.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("GLOG_minloglevel", "2")       # hush MediaPipe / TFLite logs
os.environ.setdefault("GLOG_logtostderr", "0")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class DetectedHand:
    landmarks: list          # 21 normalised landmarks (have .x .y .z)
    handedness: str          # "Left" / "Right"
    score: float


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        static: bool = False,
    ) -> None:
        self._hands = mp_hands.Hands(
            static_image_mode=static,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr) -> list[DetectedHand]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        hands: list[DetectedHand] = []
        if not result.multi_hand_landmarks:
            return hands

        handedness = result.multi_handedness or []
        for idx, lm in enumerate(result.multi_hand_landmarks):
            label, score = "Right", 1.0
            if idx < len(handedness):
                cls = handedness[idx].classification[0]
                label, score = cls.label, cls.score
            hands.append(DetectedHand(list(lm.landmark), label, score))
        return hands

    def draw(self, frame_bgr, detected: list[DetectedHand]):
        """Overlay the hand skeleton for every detected hand."""
        if not detected:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        for hand in detected:
            proto = _to_landmark_list(hand.landmarks)
            mp_draw.draw_landmarks(
                frame_bgr,
                proto,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self) -> None:
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _to_landmark_list(landmarks):
    from mediapipe.framework.formats import landmark_pb2

    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=getattr(lm, "z", 0.0))
    return proto
