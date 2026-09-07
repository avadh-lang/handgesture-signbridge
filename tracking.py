"""One tracker interface, two back-ends.

* ``holistic`` (default) - MediaPipe Holistic: both hands *and* a body pose, so
  signs can be located relative to the torso (chin vs chest vs shoulder) the way
  real sign language works.
* ``hands`` - the lighter MediaPipe Hands from ``hand_tracker.py``; no body
  reference, locations fall back to image-frame coordinates.

Both yield a ``skeleton.Pose`` per frame and know how to draw themselves.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

import cv2
import mediapipe as mp
import numpy as np

from hand_tracker import HandTracker
from skeleton import BodyRef, Pose, as_xy

_mp_holistic = mp.solutions.holistic
_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles
_POSE_CONNECTIONS = _mp_holistic.POSE_CONNECTIONS
_HAND_CONNECTIONS = _mp_holistic.HAND_CONNECTIONS


def _lm_list(landmarks):
    from mediapipe.framework.formats import landmark_pb2
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=getattr(lm, "z", 0.0),
                           visibility=getattr(lm, "visibility", 0.0))
    return proto


class HolisticTracker:
    kind = "holistic"

    def __init__(self, complexity: int = 1, detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5, static: bool = False,
                 swap_hands: bool = False):
        self._h = _mp_holistic.Holistic(
            static_image_mode=static,
            model_complexity=complexity,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.swap_hands = swap_hands
        self._last = None

    def process(self, frame_bgr) -> Pose:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._h.process(rgb)
        self._last = res

        left = as_xy(res.left_hand_landmarks.landmark) if res.left_hand_landmarks else None
        right = as_xy(res.right_hand_landmarks.landmark) if res.right_hand_landmarks else None
        if self.swap_hands:
            left, right = right, left
        body = BodyRef.from_pose_landmarks(
            res.pose_landmarks.landmark if res.pose_landmarks else None
        )
        return Pose(left=left, right=right, body=body)

    def draw(self, frame_bgr, pose: Pose):
        res = self._last
        if res is None:
            return frame_bgr
        if res.pose_landmarks:
            _mp_draw.draw_landmarks(
                frame_bgr, res.pose_landmarks, _POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_draw.DrawingSpec((180, 180, 180), 1, 1),
                connection_drawing_spec=_mp_draw.DrawingSpec((90, 90, 90), 1),
            )
        for hlm in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hlm:
                _mp_draw.draw_landmarks(
                    frame_bgr, hlm, _HAND_CONNECTIONS,
                    _mp_styles.get_default_hand_landmarks_style(),
                    _mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def close(self):
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HandsTracker:
    kind = "hands"

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.5, static: bool = False,
                 swap_hands: bool = False):
        self._t = HandTracker(max_hands=max_hands,
                              detection_confidence=detection_confidence,
                              tracking_confidence=tracking_confidence, static=static)
        self.swap_hands = swap_hands
        self._last: list = []

    def process(self, frame_bgr) -> Pose:
        hands = self._t.process(frame_bgr)
        self._last = hands
        left = right = None
        for h in hands:
            label = h.handedness
            if self.swap_hands:
                label = "Right" if label == "Left" else "Left"
            if label == "Left":
                left = as_xy(h.landmarks)
            else:
                right = as_xy(h.landmarks)
        return Pose(left=left, right=right, body=None)

    def draw(self, frame_bgr, pose: Pose):
        self._t.draw(frame_bgr, self._last)
        return frame_bgr

    def close(self):
        self._t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_tracker(kind: str = "holistic", static: bool = False,
                 swap_hands: bool = False):
    if kind == "hands":
        return HandsTracker(static=static, swap_hands=swap_hands)
    return HolisticTracker(static=static, swap_hands=swap_hands)


def dominant_hand(pose: Pose) -> tuple[np.ndarray | None, str]:
    """The hand to fingerspell with: the only one present, else the higher one."""
    if pose.right is not None and pose.left is None:
        return pose.right, "Right"
    if pose.left is not None and pose.right is None:
        return pose.left, "Left"
    if pose.left is not None and pose.right is not None:
        if pose.right[:, 1].mean() <= pose.left[:, 1].mean():
            return pose.right, "Right"
        return pose.left, "Left"
    return None, ""
