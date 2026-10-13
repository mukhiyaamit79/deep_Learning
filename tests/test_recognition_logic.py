from services.face_recognition import FrameAnalysis, FrameStatus, Prediction, RecognitionStabilizer, RecognitionThresholds
from utils.image_utils import FaceBox, box_iou, expand_box

BOX = FaceBox(100, 100, 120, 120)


def frame(label, t, box=BOX, status=FrameStatus.CANDIDATE, score=0.97):
    pred = Prediction(0, label, score, 0.9, ((label, score),)) if label else None
    return FrameAnalysis(status, (box,) if label else (), pred, "", t)


def test_needs_consecutive_frames():
    stab = RecognitionStabilizer(required_frames=3, max_gap_s=1.0)
    assert not stab.update(frame("Amit", 0.0)).is_stable
    assert not stab.update(frame("Amit", 0.1)).is_stable
    snap = stab.update(frame("Amit", 0.2))
    assert snap.is_stable and snap.label == "Amit" and snap.event_id == 1


def test_unknown_frame_or_other_student_restarts():
    stab = RecognitionStabilizer(required_frames=3, max_gap_s=1.0)
    stab.update(frame("Amit", 0.0))
    stab.update(frame(None, 0.1, status=FrameStatus.NO_FACE))
    assert stab.update(frame("Amit", 0.2)).consecutive == 1
    assert stab.update(frame("Udhav", 0.3)).consecutive == 1


def test_jumping_face_box_restarts():
    stab = RecognitionStabilizer(required_frames=3, max_gap_s=1.0)
    stab.update(frame("Amit", 0.0))
    assert stab.update(frame("Amit", 0.1, box=FaceBox(400, 300, 120, 120))).consecutive == 1


def test_long_pause_restarts_and_new_event_after_reset():
    stab = RecognitionStabilizer(required_frames=1, max_gap_s=0.5)
    stab.update(frame("Amit", 0.0))
    second = stab.update(frame("Amit", 2.0))
    assert second.consecutive == 1
    stab.reset()
    assert stab.update(frame("Amit", 2.1)).event_id == second.event_id + 1


def test_threshold_rules():
    t = RecognitionThresholds(confidence_threshold=0.9, min_margin=0.0)
    assert Prediction(0, "A", 0.95, 0.9, ()).passes(t)
    assert not Prediction(0, "A", 0.85, 0.8, ()).passes(t)


def test_box_helpers():
    assert box_iou(BOX, BOX) == 1.0
    assert box_iou(BOX, FaceBox(500, 500, 10, 10)) == 0.0
    grown = expand_box(FaceBox(0, 0, 50, 50), 0.5, (60, 60, 3))
    assert (grown.x, grown.y, grown.w, grown.h) == (0, 0, 60, 60)
