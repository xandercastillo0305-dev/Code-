"""Create safe, non-explicit demo images (see demo/README.md)."""
import os

import numpy as np
from PIL import Image, ImageEnhance
from skimage import data
from skimage.transform import swirl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = Image.fromarray(data.astronaut())
    ref.save(os.path.join(OUT, "reference.png"))

    ref.resize((448, 448), Image.BICUBIC).save(os.path.join(OUT, "suspect_authentic.jpg"), quality=80)

    # Synthetic "manipulation": swirl the inner face region and shift its tone.
    arr = np.asarray(ref).astype(np.float64) / 255.0
    y0, y1, x0, x1 = 70, 170, 180, 270                      # inner face region of the portrait
    region = swirl(arr[y0:y1, x0:x1], strength=2.5, radius=55)
    arr[y0:y1, x0:x1] = region
    fake = Image.fromarray((arr * 255).astype(np.uint8))
    box = (x0, y0, x1, y1)
    fake.paste(ImageEnhance.Color(fake.crop(box)).enhance(1.6), box)
    fake.save(os.path.join(OUT, "suspect_manipulated.png"))

    Image.fromarray(data.coffee()).save(os.path.join(OUT, "no_face.png"))
    print(f"Demo samples written to {OUT}")


if __name__ == "__main__":
    main()
