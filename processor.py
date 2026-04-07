"""
╔══════════════════════════════════════════════════════════════╗
║  SilentAssist — Video Preprocessing Pipeline                ║
║  ──────────────────────────────────────────────────────────  ║
║  Uses MediaPipe FaceLandmarker (Tasks API ≥ 0.10.x) to      ║
║  extract a normalised lip ROI from every frame and packs    ║
║  them into a PyTorch tensor shaped (B, C, T, H, W) for a   ║
║  3D-CNN backbone.                                            ║
╚══════════════════════════════════════════════════════════════╝

Landmark indices used (MediaPipe Face Mesh 478-point topology):
  • Outer lip ring  → indices 61, 146, 91, 181, 84, 17, 314, 405,
                       321, 375, 291, 308, 324, 318, 402, 317,
                       14, 87, 178, 88, 95
  • These span the full visible lip contour — ideal for cropping.
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional

# ── MediaPipe Tasks API setup ────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the downloaded model asset
_MODEL_ASSET_PATH = str(Path(__file__).parent / "face_landmarker.task")

# Lip contour landmark indices (outer ring, MediaPipe 478-pt mesh)
LIP_LANDMARK_IDS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
]

# Target spatial size for the lip ROI crop (matches LipNet convention)
ROI_HEIGHT = 64
ROI_WIDTH  = 128

# Max frames to process (pad / truncate to this length)
MAX_FRAMES = 75


def _create_landmarker() -> FaceLandmarker:
    """
    Create a FaceLandmarker instance using the Tasks API.
    Uses IMAGE running mode for frame-by-frame processing.
    """
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_ASSET_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def _save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temporary path so OpenCV
    can open it with cv2.VideoCapture (which needs a real file path).
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def _extract_lip_roi(
    frame_bgr: np.ndarray,
    landmarker: FaceLandmarker,
    padding_factor: float = 0.35,
) -> Optional[np.ndarray]:
    """
    Detect the face in *frame_bgr* and return a cropped, resized
    grayscale image of the lip region.

    Args:
        frame_bgr:      Raw BGR frame from OpenCV.
        landmarker:     MediaPipe FaceLandmarker instance.
        padding_factor: Extra padding (proportion) around the tight
                        bounding box of the lip landmarks so that
                        surrounding chin / nose context is preserved.

    Returns:
        np.ndarray of shape (ROI_HEIGHT, ROI_WIDTH) in uint8,
        or None if no face was detected.
    """
    h, w, _ = frame_bgr.shape

    # Convert BGR → RGB and create MediaPipe Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    landmarks = result.face_landmarks[0]

    # Gather lip pixel coordinates
    xs = [landmarks[i].x * w for i in LIP_LANDMARK_IDS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARK_IDS]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding
    pad_x = int((x_max - x_min) * padding_factor)
    pad_y = int((y_max - y_min) * padding_factor)

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    # Crop & convert to grayscale
    lip_crop = frame_bgr[y_min:y_max, x_min:x_max]
    if lip_crop.size == 0:
        return None

    gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))
    return resized


def preprocess_video(
    video_source,
    max_frames: int = MAX_FRAMES,
) -> Tuple[torch.Tensor, list]:
    """
    Full preprocessing pipeline: video → lip-ROI tensor.

    Args:
        video_source: Either a file-path string or a Streamlit
                      UploadedFile object.
        max_frames:   Number of temporal frames. Shorter clips are
                      zero-padded; longer ones are uniformly sampled.

    Returns:
        tensor:       Float32 tensor of shape (1, 1, T, H, W)
                      normalised to [0, 1].
        roi_frames:   List of raw uint8 ROI crops (for UI preview).
    """
    # ── Resolve the path ─────────────────────────────────────
    if isinstance(video_source, str):
        video_path = video_source
    else:
        video_path = _save_uploaded_file(video_source)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # ── Create the landmarker ────────────────────────────────
    landmarker = _create_landmarker()

    # ── Read all frames & extract lip ROIs ───────────────────
    all_rois: list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = _extract_lip_roi(frame, landmarker)
        if roi is not None:
            all_rois.append(roi)
    cap.release()

    # Close the landmarker to release resources
    landmarker.close()

    # Clean up temp file if we created one
    if not isinstance(video_source, str):
        os.unlink(video_path)

    if len(all_rois) == 0:
        raise ValueError(
            "No lip region detected in any frame. "
            "Ensure the video contains a clearly visible face."
        )

    # ── Temporal normalisation ───────────────────────────────
    #    Uniform sampling when clip is too long;
    #    zero-padding when clip is too short.
    n = len(all_rois)
    if n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        all_rois = [all_rois[i] for i in indices]
    elif n < max_frames:
        pad_count = max_frames - n
        all_rois += [np.zeros_like(all_rois[0])] * pad_count

    # ── Stack into a tensor (B=1, C=1, T, H, W) ─────────────
    frames_np = np.stack(all_rois, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(frames_np).unsqueeze(0).unsqueeze(0)
    # Shape:  (1, 1, max_frames, ROI_HEIGHT, ROI_WIDTH)

    return tensor, all_rois


# ══════════════════════════════════════════════════════════════
#  Live Camera — Single-Frame Utilities
# ══════════════════════════════════════════════════════════════
def extract_lip_roi_from_frame(
    frame_bgr: np.ndarray,
    landmarker: FaceLandmarker,
    padding_factor: float = 0.35,
    draw_overlay: bool = True,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Process a single BGR frame for live-camera mode.

    Returns:
        roi:            Grayscale lip crop (ROI_HEIGHT × ROI_WIDTH) or None.
        annotated_frame: The original frame with lip landmarks / ROI box
                         drawn on it (for live preview).
    """
    annotated = frame_bgr.copy()
    h, w, _ = frame_bgr.shape

    # Convert BGR → RGB and create MediaPipe Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        # No face → draw a "no face" indicator
        cv2.putText(
            annotated, "No face detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return None, annotated

    landmarks = result.face_landmarks[0]

    # Gather lip pixel coordinates
    xs = [landmarks[i].x * w for i in LIP_LANDMARK_IDS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARK_IDS]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Draw lip contour points on the annotated frame
    if draw_overlay:
        lip_pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
             for i in LIP_LANDMARK_IDS],
            dtype=np.int32,
        )
        cv2.polylines(annotated, [lip_pts], isClosed=True,
                      color=(88, 166, 255), thickness=2)

    # Add padding
    pad_x = int((x_max - x_min) * padding_factor)
    pad_y = int((y_max - y_min) * padding_factor)

    x_min_p = max(0, x_min - pad_x)
    x_max_p = min(w, x_max + pad_x)
    y_min_p = max(0, y_min - pad_y)
    y_max_p = min(h, y_max + pad_y)

    # Draw ROI bounding box
    if draw_overlay:
        cv2.rectangle(annotated, (x_min_p, y_min_p), (x_max_p, y_max_p),
                      (63, 185, 80), 2)
        cv2.putText(
            annotated, "LIP ROI", (x_min_p, y_min_p - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (63, 185, 80), 1,
        )

    # Crop & convert to grayscale
    lip_crop = frame_bgr[y_min_p:y_max_p, x_min_p:x_max_p]
    if lip_crop.size == 0:
        return None, annotated

    gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))
    return resized, annotated


def build_tensor_from_buffer(
    roi_buffer: list,
    max_frames: int = MAX_FRAMES,
) -> torch.Tensor:
    """
    Convert a list/deque of ROI frames into a model-ready tensor.

    Args:
        roi_buffer:  List of uint8 grayscale ROI frames.
        max_frames:  Target temporal length.

    Returns:
        Float32 tensor of shape (1, 1, T, H, W) normalised to [0, 1].
    """
    rois = list(roi_buffer)

    if len(rois) == 0:
        # Return zeros if buffer is empty
        return torch.zeros(1, 1, max_frames, ROI_HEIGHT, ROI_WIDTH)

    n = len(rois)
    if n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        rois = [rois[i] for i in indices]
    elif n < max_frames:
        pad_count = max_frames - n
        rois += [np.zeros_like(rois[0])] * pad_count

    frames_np = np.stack(rois, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(frames_np).unsqueeze(0).unsqueeze(0)
    return tensor


def get_landmarker() -> FaceLandmarker:
    """Public accessor for creating a FaceLandmarker (for live mode)."""
    return _create_landmarker()

