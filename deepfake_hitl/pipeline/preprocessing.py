"""Preprocessing module (Section 3.3, Algorithm 3.6 steps 2-3).

1. MTCNN detects faces and 5 facial landmarks; the largest face is kept.
2. The image is rotated so the eyes are horizontal (alignment).
3. A square crop around the face (plus a small margin) is taken.
4. The crop is resized to 380x380 (EfficientNet-B4 native size).
5. The crop is converted to a tensor and normalised with ImageNet mean/std.

If no face is found, NoFaceDetectedError is raised; the system never guesses.
The aligned face crop is also returned as a PIL image so the review UI and
report can show the face only (Appendix D redaction protocol).
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

import config

NO_FACE_MESSAGE = "No face detected. Case cannot be analyzed."


class NoFaceDetectedError(ValueError):
    def __init__(self, message=NO_FACE_MESSAGE):
        super().__init__(message)


@dataclass
class PreprocessResult:
    tensor: "object"          # torch.FloatTensor [3, 380, 380], ImageNet-normalised
    face: Image.Image         # aligned RGB face crop, 380x380
    box: tuple                # (x1, y1, x2, y2) of the detected face in the original image
    probability: float        # MTCNN detection confidence


class FacePreprocessor:
    """Wraps MTCNN. ``detector`` can be injected (tests use a stub).

    A detector is any callable ``detector(pil_image) -> (boxes, probs, landmarks)``
    with the same shapes as ``facenet_pytorch.MTCNN.detect(img, landmarks=True)``.
    """

    def __init__(self, detector=None, device="cpu", image_size=config.IMAGE_SIZE,
                 margin=config.FACE_MARGIN, min_probability=config.MIN_FACE_PROBABILITY):
        self._detector = detector
        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.min_probability = min_probability

    # -- detection ---------------------------------------------------------
    def _detect(self, img):
        if self._detector is None:
            from facenet_pytorch import MTCNN
            mtcnn = MTCNN(keep_all=True, device=self.device)
            self._detector = lambda im: mtcnn.detect(im, landmarks=True)
        return self._detector(img)

    def detect_largest_face(self, img):
        boxes, probs, landmarks = self._detect(img)
        if boxes is None or len(boxes) == 0:
            raise NoFaceDetectedError()
        candidates = [
            (i, (b[2] - b[0]) * (b[3] - b[1]))
            for i, b in enumerate(boxes)
            if probs[i] is not None and probs[i] >= self.min_probability
        ]
        if not candidates:
            raise NoFaceDetectedError()
        i = max(candidates, key=lambda c: c[1])[0]
        lm = None if landmarks is None else np.asarray(landmarks[i], dtype=float)
        return np.asarray(boxes[i], dtype=float), float(probs[i]), lm

    # -- alignment + crop ------------------------------------------------------
    def align_and_crop(self, img, box, landmarks):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if landmarks is not None and len(landmarks) >= 2:
            # MTCNN landmark order: left eye, right eye, nose, mouth-left, mouth-right
            (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
            angle = math.degrees(math.atan2(ry - ly, rx - lx))
            # rotate around the face centre so the eyes become horizontal
            img = img.rotate(angle, resample=Image.BILINEAR, center=(cx, cy))
        side = max(x2 - x1, y2 - y1) * (1.0 + 2 * self.margin)
        half = side / 2.0
        crop_box = (int(round(cx - half)), int(round(cy - half)),
                    int(round(cx + half)), int(round(cy + half)))
        # PIL pads out-of-bounds regions with black, keeping the crop square.
        face = img.crop(crop_box)
        return face.resize((self.image_size, self.image_size), Image.BICUBIC)

    # -- full pipeline -----------------------------------------------------------
    def process_image(self, img):
        img = ImageOps.exif_transpose(img).convert("RGB")
        box, prob, landmarks = self.detect_largest_face(img)
        face = self.align_and_crop(img, box, landmarks)
        return PreprocessResult(to_tensor(face), face, tuple(float(v) for v in box), prob)

    def process_path(self, path):
        with Image.open(path) as img:
            img.load()
            return self.process_image(img)


def to_tensor(face):
    """PIL RGB 380x380 -> normalised float tensor [3, H, W]."""
    import torch
    arr = np.asarray(face.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(config.IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        config.IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())
