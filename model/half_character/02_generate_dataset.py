import json
import os
import random
import cv2
import freetype
import numpy as np
import uharfbuzz as hb
from tqdm import tqdm

BASE_DIR = os.path.dirname(__file__)
FONT_DIR = os.path.join(BASE_DIR, "fonts")
CLASS_FILE = os.path.join(BASE_DIR, "classes.json")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_N = 160
TEST_N = 40
IMG_SIZE = (64, 64)


def get_glyph_ids(text, data):
    # Initialize HarfBuzz to apply OpenType shaping rules
    face = hb.Face(data)
    font = hb.Font(face)
    buf = hb.Buffer()

    # Append ZWJ after the character to encourage half-form shaping
    buf.add_str(text if text.endswith("\u200d") else text + "\u200d")
    buf.guess_segment_properties()

    # Apply Indic shaping features to produce the half-character form
    hb.shape(font, buf, {"half": True, "akhn": True, "rphf": True, "blwf": True})

    # Exclude common control/space glyph IDs
    return [g.codepoint for g in buf.glyph_infos if g.codepoint > 3]


def render_glyph(gid, face):
    # Render the selected glyph using FreeType
    try:
        face.load_glyph(gid, freetype.FT_LOAD_RENDER)
    except Exception:
        return None

    bm = face.glyph.bitmap
    if not bm.width or not bm.rows:
        return None

    # Convert the glyph bitmap into a NumPy array
    arr = np.array(bm.buffer, dtype=np.uint8).reshape(bm.rows, bm.width)

    # Find the actual ink area and remove empty margins
    pts = np.argwhere(arr > 20)
    if not len(pts):
        return None

    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    crop = 255 - arr[y1 : y2 + 1, x1 : x2 + 1]

    # Resize proportionally while leaving a 6-pixel margin on each side
    h, w = crop.shape
    scale = min((IMG_SIZE[0] - 12) / h, (IMG_SIZE[1] - 12) / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # Center the glyph on a white 64x64 canvas
    img = np.full(IMG_SIZE, 255, dtype=np.uint8)
    y, x = (IMG_SIZE[0] - nh) // 2, (IMG_SIZE[1] - nw) // 2
    img[y : y + nh, x : x + nw] = crop
    return img


def augment(img):
    img = img.copy()

    # Randomly vary stroke thickness to simulate different writing styles
    k = random.choice([0, 1, 2])
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img = random.choice([cv2.erode, cv2.dilate])(img, kernel)

    # Apply slight rotation to introduce natural handwriting variation
    center = (IMG_SIZE[1] // 2, IMG_SIZE[0] // 2)
    mat = cv2.getRotationMatrix2D(center, random.uniform(-8, 8), 1)
    img = cv2.warpAffine(img, mat, IMG_SIZE, borderValue=255)

    # Apply horizontal shear to vary the character's slant
    shear = random.uniform(-0.12, 0.12)
    mat = np.float32([[1, shear, 0], [0, 1, 0]])
    img = cv2.warpAffine(img, mat, IMG_SIZE, borderValue=255)

    # Remove a few top rows to simulate shirorekha removal
    if random.random() > 0.35:
        img[: random.randint(1, 3), :] = 255

    # Convert to a clean binary image
    return cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]


def generate_dataset():
    with open(CLASS_FILE, encoding="utf-8") as f:
        classes = json.load(f)
    fonts = [
        os.path.join(FONT_DIR, f)
        for f in os.listdir(FONT_DIR)
        if f.lower().endswith((".ttf", ".otf", ".ttc"))
    ]

    roots = {
        "train": os.path.join(DATA_DIR, "train"),
        "test": os.path.join(DATA_DIR, "test"),
    }

    print(f"Generating dataset for {len(classes)} classes...")

    for cls in tqdm(classes, desc="Classes"):
        name, code = cls["name"], cls["code"]
        glyphs = []

        # Extract the half-character glyph from each available font
        for path in fonts:
            try:
                with open(path, "rb") as f:
                    ids = get_glyph_ids(code, f.read())

                if ids:
                    face = freetype.Face(path)
                    face.set_char_size(48 * 64)
                    img = render_glyph(ids[0], face)

                    if img is not None:
                        glyphs.append(img)
            except Exception:
                # Skip fonts that cannot shape or render this character
                continue

        if not glyphs:
            print(f"[WARN] No glyph found: {name}")
            continue

        # Generate independent training and testing samples
        for split, count in (("train", TRAIN_N), ("test", TEST_N)):
            out_dir = os.path.join(roots[split], name)
            os.makedirs(out_dir, exist_ok=True)

            for i in range(count):
                # Randomize the base font and its appearance
                img = augment(random.choice(glyphs))
                cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img)

    print(f"Dataset ready: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
