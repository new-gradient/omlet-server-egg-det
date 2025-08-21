from typing import Any, List, Optional, Tuple
import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required for image preprocessing. Please install 'Pillow'.")


class DetectionResult:
    """Simple container to align with app.py expectations.

    Attributes:
        boxes (np.ndarray): Array of shape (N, 4) in xyxy format in original image scale.
        scores (np.ndarray): Array of shape (N,) with confidence scores [0,1].
        classes (np.ndarray): Array of shape (N,) with class indices (optional, default 0 for 'egg').
    """

    def __init__(self, boxes: np.ndarray, scores: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None):
        self.boxes = boxes.astype(np.float32) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
        self.scores = scores.astype(np.float32) if scores is not None else np.ones((len(self.boxes),), dtype=np.float32)
        self.classes = classes.astype(np.int64) if classes is not None else np.zeros((len(self.boxes),), dtype=np.int64)


def _preprocess_image(image_path: str, size: int = 640) -> Tuple[np.ndarray, float, float, int, int]:
    """Load image, resize to size x size, normalize to [0,1], CHW float32.

    Returns:
        input_tensor: np.ndarray with shape (1, 3, size, size)
        sx: scale factor for width (orig_w / size)
        sy: scale factor for height (orig_h / size)
        orig_w, orig_h: original image dimensions
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # HWC, [0,1]
    # Optional: Uncomment if your model expects ImageNet normalization
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # arr = (arr - mean) / std

    chw = np.transpose(arr, (0, 1, 2))  # HWC -> HWC (no-op for readability)
    chw = np.transpose(chw, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(chw, axis=0)  # NCHW

    sx = float(orig_w) / float(size)
    sy = float(orig_h) / float(size)

    return input_tensor.astype(np.float32), sx, sy, orig_w, orig_h


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    # Remove leading batch dim if present
    if x.ndim >= 2 and x.shape[0] == 1:
        return np.squeeze(x, axis=0)
    return x


def _select_arrays(outputs: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Attempt to infer boxes, scores, classes from model outputs.
    Returns (boxes[N,4], scores[N], classes[N]). Any may be None if not found.
    """
    boxes = None
    scores = None
    classes = None

    # Squeeze potential batch dims
    squeezed = [_squeeze_batch(o) for o in outputs]

    # First try to find a (N,6) or (N,5) tensor: [x1,y1,x2,y2,score,(cls)]
    for out in squeezed:
        if out.ndim == 2 and out.shape[-1] in (5, 6):
            b = out[:, :4]
            s = out[:, 4]
            c = out[:, 5].astype(np.int64) if out.shape[-1] == 6 else None
            boxes = b if boxes is None else boxes
            scores = s if scores is None else scores
            classes = c if (classes is None and c is not None) else classes
            # Do not break; continue to allow other outputs to complement missing pieces

    # If boxes not found, look for (N,4)
    if boxes is None:
        candidates = [o for o in squeezed if o.ndim == 2 and o.shape[-1] == 4]
        if candidates:
            boxes = candidates[0]

    # If scores not found, look for a 1D float array with matching N
    if boxes is not None and scores is None:
        n = boxes.shape[0]
        cand_scores = [o for o in squeezed if o.ndim == 1 and o.shape[0] == n and np.issubdtype(o.dtype, np.floating)]
        if cand_scores:
            scores = cand_scores[0]

    # If classes not found, look for a 1D int array with matching N
    if boxes is not None and classes is None:
        n = boxes.shape[0]
        cand_classes = [o for o in squeezed if o.ndim == 1 and o.shape[0] == n and np.issubdtype(o.dtype, np.integer)]
        if cand_classes:
            classes = cand_classes[0].astype(np.int64)

    return boxes, scores, classes


def run_detection(image_path: str, conf_threshold: float, session: Any) -> Tuple[Optional[DetectionResult], List[str]]:
    """Run RT-DETR ONNX detection on an image and return results for egg counting.

    Args:
        image_path: Path to image file (local path).
        conf_threshold: Confidence threshold to filter detections.
        session: onnxruntime.InferenceSession (already created by app).

    Returns:
        (DetectionResult or None, labels)
        labels: class names list; for egg counting we return ["egg"].
    """
    # Preprocess
    inp, sx, sy, _, _ = _preprocess_image(image_path, size=640)

    # Prepare inputs
    input_name = session.get_inputs()[0].name
    inputs = {input_name: inp}

    # Some RT-DETR exports also require an 'orig_target_sizes' or similar; if present, try to feed it
    # by checking any additional float/int input shapes.
    if len(session.get_inputs()) > 1:
        for extra in session.get_inputs()[1:]:
            # Heuristics: if expects (N, 2) ints, feed original sizes mapped from 640 resize
            shape = extra.shape
            name = extra.name
            if isinstance(shape, list) and len(shape) == 2 and shape[1] == 2:
                # Provide original size as (H, W) in int64
                # Since we resized image to 640x640, provide the resized size
                hw = np.array([[640, 640]], dtype=np.int64)
                inputs[name] = hw
            elif len(shape) == 1 and (shape[0] == 2 or shape[0] is None):
                # Rare case: single-dim vector with size info
                inputs[name] = np.array([640, 640], dtype=np.int64)
            else:
                # If unknown, attempt a reasonable default based on expected dtype
                dt = np.float32 if extra.type == 'tensor(float)' else np.int64
                # Try a scalar 1 if allowable; otherwise skip
                try:
                    inputs[name] = np.array(1, dtype=dt)
                except Exception:
                    pass

    # Inference
    outputs = session.run(None, inputs)

    # Parse outputs
    boxes, scores, classes = _select_arrays(outputs)

    if boxes is None:
        # No valid detections found
        return None, ["egg"]

    # Scale boxes back to original size from 640 -> orig
    boxes = boxes.copy().astype(np.float32)
    # boxes are expected as xyxy in resized space
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    # Confidence filtering
    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    keep = scores >= float(conf_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    if classes is not None and classes.shape[0] == keep.shape[0]:
        classes = classes[keep]
    else:
        classes = None

    if boxes.shape[0] == 0:
        return None, ["egg"]

    result = DetectionResult(boxes=boxes, scores=scores, classes=classes)

    # For egg counting we expose a single class list
    labels = ["egg"]
    return result, labels
