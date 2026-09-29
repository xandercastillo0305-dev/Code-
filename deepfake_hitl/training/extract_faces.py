"""Extract aligned face crops from dataset frames with MTCNN (Section 3.5).

Works on folders of video frames (JPG/PNG) from FaceForensics++, Celeb-DF or
DFDC. Frame extraction from the videos themselves is done beforehand
(e.g. `ffmpeg -i video.mp4 -vf fps=1 frames/video_%04d.png`).

    python -m training.extract_faces --src raw/ffpp/original --dst data/train/real
    python -m training.extract_faces --src raw/ffpp/Deepfakes --dst data/train/fake

Uses the SAME preprocessing (MTCNN, alignment, 380x380) as the live system,
so training data matches what the model sees at inference. Frames with no
face are skipped and counted. No datasets are downloaded by this project.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model import default_device  # noqa: E402
from pipeline.preprocessing import FacePreprocessor, NoFaceDetectedError  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png")


def extract(src, dst, device=None, every_nth=1, preprocessor=None):
    pre = preprocessor or FacePreprocessor(device=device or default_device())
    os.makedirs(dst, exist_ok=True)
    saved = skipped = 0
    frames = []
    for dirpath, _, files in os.walk(src):
        frames += [os.path.join(dirpath, f) for f in sorted(files) if f.lower().endswith(EXTS)]
    for i, path in enumerate(frames):
        if i % every_nth:
            continue
        rel = os.path.relpath(path, src)
        out = os.path.join(dst, os.path.splitext(rel.replace(os.sep, "__"))[0] + ".png")
        try:
            pre.process_path(path).face.save(out)
            saved += 1
        except NoFaceDetectedError:
            skipped += 1
    print(f"{src}: saved {saved} face crops to {dst}, skipped {skipped} frames without a face")
    return saved, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--every-nth", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    extract(a.src, a.dst, a.device, a.every_nth)
