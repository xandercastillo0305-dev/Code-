"""Evidence file handling: upload validation, SHA-256 chain of custody,
redaction (blur / pixelation) and secure deletion (Appendix D).

Layout (outside static/, never served directly):

    data/uploads/CASE-0001/suspect.jpg          original upload (hash recorded)
    data/uploads/CASE-0001/reference.png        original upload
    data/uploads/CASE-0001/suspect_face.png     aligned 380x380 face crop
    data/uploads/CASE-0001/reference_face.png
"""
import hashlib
import io
import os
import shutil

from PIL import Image, ImageFilter, ImageOps

import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
ROLES_IMAGES = ("suspect", "reference")


class UploadError(ValueError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image_bytes(filename, data, max_mb=config.MAX_UPLOAD_MB):
    """JPG/PNG only, size-limited, and actually decodable by Pillow. Returns the extension."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UploadError(f"{filename or 'File'}: only JPG and PNG images are accepted.")
    if not data:
        raise UploadError(f"{filename}: file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"{filename}: file is larger than {max_mb} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Exception:
        raise UploadError(f"{filename}: not a valid image file.")
    if fmt not in config.ALLOWED_PIL_FORMATS:
        raise UploadError(f"{filename}: content is {fmt}, only JPEG and PNG are accepted.")
    return ".jpg" if ext == ".jpeg" else ext


def case_dir(case_id, upload_dir=None):
    return os.path.join(upload_dir or config.UPLOAD_DIR, case_id)


def face_path(case_id, which, upload_dir=None):
    assert which in ROLES_IMAGES
    return os.path.join(case_dir(case_id, upload_dir), f"{which}_face.png")


def redact(img, mode="blur", strength=None):
    """Blur (Gaussian) or pixelate an image for display."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if mode == "pixelate":
        block = strength or 24
        w, h = img.size
        small = img.resize((max(1, w // block), max(1, h // block)), Image.BILINEAR)
        return small.resize((w, h), Image.NEAREST)
    radius = strength or max(6, img.size[0] // 30)
    return img.filter(ImageFilter.GaussianBlur(radius))


def image_png_bytes(img, max_side=None):
    if max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def secure_delete_file(path, passes=1):
    """Overwrite a file with random bytes before unlinking it.

    Note: on SSDs / copy-on-write or journaling filesystems an overwrite does
    not guarantee physical erasure; full-disk encryption is recommended for
    deployment (documented in README).
    """
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        for _ in range(passes):
            f.seek(0)
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())
    os.remove(path)
    return True


def secure_delete_tree(directory):
    removed = 0
    if not os.path.isdir(directory):
        return 0
    for root, _, files in os.walk(directory):
        for name in files:
            removed += secure_delete_file(os.path.join(root, name))
    shutil.rmtree(directory, ignore_errors=True)
    return removed
