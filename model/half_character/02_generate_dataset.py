import json
import os
import shutil
import cv2
import freetype
import numpy as np
import uharfbuzz as hb
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")
CLASS_FILE = os.path.join(BASE_DIR, "classes.json")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_N = 200
TEST_N = 50
IMG_SIZE = (64, 64)
RANDOM_SEED = 42
FONT_SIZE = 128
MARGIN = 6

RESERVED_TEST_FONTS = {
    "NotoSerif-Regular.ttf",
    "NotoSerif-Bold.ttf",
    "Rajdhani-Regular.ttf",
    "Rajdhani-Bold.ttf",
}

HANDWRITING_FONTS = {"kalam", "dekko", "amita", "tillana", "laila", "baloo"}


def validate_font(font_path, cls):
    # Verify the font shapes a true half-form rather than falling back
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        font = hb.Font(hb.Face(data))
        font.scale = (FONT_SIZE, FONT_SIZE)
    except Exception:
        return False

    code = cls["code"]
    text = code if code.endswith("\u200d") else code + "\u200d"
    full_char = cls["char"]

    # 1. Shape half-character sequence
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"half": True, "akhn": True, "rphf": True, "blwf": True})
    half_gids = [g.codepoint for g in buf.glyph_infos if g.codepoint > 3]
    if not half_gids or 0 in half_gids:
        return False

    # 2. Shape full consonant to detect missing half-form fallback
    buf = hb.Buffer()
    buf.add_str(full_char)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    full_gids = [g.codepoint for g in buf.glyph_infos if g.codepoint > 3]
    if half_gids == full_gids:
        return False

    # 3. Reject full consonant + virama stack
    buf = hb.Buffer()
    buf.add_str("\u094d")
    buf.guess_segment_properties()
    hb.shape(font, buf)
    vir_gids = [g.codepoint for g in buf.glyph_infos if g.codepoint > 3]
    if any(g in full_gids for g in half_gids) and any(g in vir_gids for g in half_gids):
        return False

    return True


def render_glyph(font_path, text):
    # Render shaped text to high-res canvas and return tight ink bounds
    try:
        ft_face = freetype.Face(font_path)
        ft_face.set_char_size(FONT_SIZE * 64)

        with open(font_path, "rb") as f:
            data = f.read()
        hb_font = hb.Font(hb.Face(data))
        hb_font.scale = (FONT_SIZE, FONT_SIZE)

        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf, {"half": True, "akhn": True, "rphf": True, "blwf": True})

        dim = FONT_SIZE * 4
        canvas = np.zeros((dim, dim), dtype=np.float32)
        cur_x, cur_y = FONT_SIZE, FONT_SIZE * 2
        has_ink = False

        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint <= 3:
                cur_x += pos.x_advance
                cur_y += pos.y_advance
                continue

            try:
                ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            except Exception:
                continue

            bm = ft_face.glyph.bitmap
            w, h = bm.width, bm.rows
            if w > 0 and h > 0:
                left = cur_x + pos.x_offset + ft_face.glyph.bitmap_left
                top = cur_y - pos.y_offset - ft_face.glyph.bitmap_top
                arr = np.array(bm.buffer, dtype=np.uint8).reshape(h, w)
                x0, y0 = int(round(left)), int(round(top))
                x1, y1 = x0 + w, y0 + h

                if 0 <= x0 and 0 <= y0 and x1 <= dim and y1 <= dim:
                    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], arr)
                    has_ink = True

            cur_x += pos.x_advance
            cur_y += pos.y_advance

        if not has_ink:
            return None

        pts = np.argwhere(canvas > 20)
        if not len(pts):
            return None

        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        return canvas[y1 : y2 + 1, x1 : x2 + 1]
    except Exception:
        return None


def fit_to_canvas(crop, pad=MARGIN):
    h, w = crop.shape
    scale = min((IMG_SIZE[0] - 2 * max(2, pad)) / h, (IMG_SIZE[1] - 2 * max(2, pad)) / w)
    nw, nh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(IMG_SIZE, dtype=np.float32)
    y, x = (IMG_SIZE[0] - nh) // 2, (IMG_SIZE[1] - nw) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas, nw, nh


def augment(crop, rng, is_train=True):
    img = crop.copy()

    # 1. Dynamic stroke thickness & continuous pen pressure variation
    if is_train:
        op = rng.choice(["none", "dilate", "erode", "pressure"], p=[0.25, 0.25, 0.25, 0.25])
        if op == "dilate":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rng.choice([2, 3, 4]),) * 2)
            img = cv2.dilate(img, k, iterations=1)
        elif op == "erode":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rng.choice([2, 3]),) * 2)
            eroded = cv2.erode(img, k, iterations=1)
            if np.count_nonzero(eroded > 20) > 0.40 * np.count_nonzero(img > 20):
                img = eroded
        elif op == "pressure":
            mask = (img > 20).astype(np.uint8)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            h, w = img.shape
            pressure = cv2.GaussianBlur(
                (rng.random((h, w), dtype=np.float32) * 2 - 1), (15, 15), 5.0
            ) * rng.uniform(0.6, 1.4)
            img = np.clip((dist + pressure) * 60, 0, 255).astype(np.float32)

    # 2. Proportional placement on canvas
    h, w = img.shape
    pad = MARGIN + (rng.integers(-2, 3) if is_train else 0)
    scale_jitter = rng.uniform(0.85, 1.10) if is_train else rng.uniform(0.92, 1.05)
    sx = rng.integers(-4, 5) if is_train else rng.integers(-2, 3)
    sy = rng.integers(-4, 5) if is_train else rng.integers(-2, 3)
    angle = rng.uniform(-12.0, 12.0) if is_train else rng.uniform(-6.0, 6.0)
    shear = rng.uniform(-0.15, 0.15) if is_train else rng.uniform(-0.06, 0.06)

    scale = min((IMG_SIZE[0] - 2 * max(2, pad)) / h, (IMG_SIZE[1] - 2 * max(2, pad)) / w) * scale_jitter
    nw, nh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(IMG_SIZE, dtype=np.float32)
    cx = np.clip((IMG_SIZE[1] - nw) // 2 + sx, 1, IMG_SIZE[1] - nw - 1)
    cy = np.clip((IMG_SIZE[0] - nh) // 2 + sy, 1, IMG_SIZE[0] - nh - 1)
    canvas[cy : cy + nh, cx : cx + nw] = resized

    # 3. Soften sharp digital corners into rounded pen nib terminals
    if is_train:
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, close_k)

    # 4. Rotation and Slant (Shear)
    center = (IMG_SIZE[1] / 2.0, IMG_SIZE[0] / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    canvas = cv2.warpAffine(canvas, mat, IMG_SIZE, flags=cv2.INTER_LINEAR, borderValue=0)

    if shear != 0.0:
        s_mat = np.float32([[1, shear, -shear * center[1]], [0, 1, 0]])
        canvas = cv2.warpAffine(canvas, s_mat, IMG_SIZE, flags=cv2.INTER_LINEAR, borderValue=0)

    # 5. Organic Shirorekha (top bar) tilt or partial gap
    if is_train and rng.random() > 0.35:
        pts = np.argwhere(canvas > 20)
        if len(pts):
            top_y = pts.min(axis=0)[0]
            bar_h = min(8, max(2, pts.max(axis=0)[0] - top_y))
            bar_cut = canvas[top_y : top_y + bar_h, :].copy()
            if rng.random() > 0.40:
                bar_mat = cv2.getRotationMatrix2D((IMG_SIZE[1] / 2.0, bar_h / 2.0), rng.uniform(-3.5, 3.5), 1.0)
                canvas[top_y : top_y + bar_h, :] = cv2.warpAffine(bar_cut, bar_mat, (IMG_SIZE[1], bar_h), borderValue=0)
            else:
                canvas[top_y : top_y + rng.integers(1, 4), :] = 0

    # 6. Natural pen lifts / broken ink streaks
    if is_train and rng.random() > 0.45:
        rx, ry = rng.integers(10, IMG_SIZE[1] - 10), rng.integers(10, IMG_SIZE[0] - 10)
        canvas[ry : ry + rng.integers(2, 5), rx : rx + rng.integers(2, 6)] = 0

    # 7. Multi-scale Elastic Warping: Macro arm curvature + Micro pen tremor
    if is_train and rng.random() > 0.30:
        grid_y, grid_x = np.indices(IMG_SIZE, dtype=np.float32)
        macro_x = cv2.GaussianBlur((rng.random(IMG_SIZE) * 2 - 1).astype(np.float32), (17, 17), 5.5) * rng.uniform(2.5, 4.5)
        macro_y = cv2.GaussianBlur((rng.random(IMG_SIZE) * 2 - 1).astype(np.float32), (17, 17), 5.5) * rng.uniform(2.5, 4.5)
        micro_x = cv2.GaussianBlur((rng.random(IMG_SIZE) * 2 - 1).astype(np.float32), (5, 5), 1.2) * rng.uniform(0.5, 1.2)
        micro_y = cv2.GaussianBlur((rng.random(IMG_SIZE) * 2 - 1).astype(np.float32), (5, 5), 1.2) * rng.uniform(0.5, 1.2)
        canvas = cv2.remap(canvas, grid_x + macro_x + micro_x, grid_y + macro_y + micro_y, interpolation=cv2.INTER_LINEAR, borderValue=0)

    # 8. Grayscale ink absorption and paper grain
    gamma = rng.uniform(0.8, 1.3) if is_train else 1.0
    norm_ink = np.clip(canvas / 255.0, 0.0, 1.0) ** gamma

    if is_train and rng.random() > 0.40:
        norm_ink = cv2.GaussianBlur(norm_ink, (3, 3), sigmaX=rng.uniform(0.3, 0.6))

    paper = rng.uniform(248, 255, size=IMG_SIZE).astype(np.float32)
    min_ink = rng.integers(15, 55) if is_train else 25
    out = np.clip(paper - norm_ink * (paper - min_ink), 0, 255).astype(np.uint8)

    # Occasional stray ink specks
    if is_train and rng.random() > 0.50:
        for _ in range(rng.integers(1, 4)):
            out[rng.integers(2, IMG_SIZE[0] - 2), rng.integers(2, IMG_SIZE[1] - 2)] = rng.integers(30, 90)

    return out


def is_valid_image(img):
    if img is None or img.shape != IMG_SIZE or img.dtype != np.uint8:
        return False
    ink_count = np.count_nonzero(img < 200)
    if ink_count < 40 or ink_count > 3200:
        return False
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    return not np.any(border < 200)


def generate_dataset():
    with open(CLASS_FILE, encoding="utf-8") as f:
        classes = json.load(f)

    fonts = sorted([f for f in os.listdir(FONT_DIR) if f.lower().endswith((".ttf", ".otf", ".ttc"))])
    roots = {
        "train": os.path.join(DATA_DIR, "train"),
        "test": os.path.join(DATA_DIR, "test"),
    }
    for r in roots.values():
        os.makedirs(r, exist_ok=True)

    rng_train = np.random.default_rng(RANDOM_SEED)
    rng_test = np.random.default_rng(RANDOM_SEED + 99999)

    print(f"Generating dataset for {len(classes)} classes...")

    for cls in tqdm(classes, desc="Classes"):
        name, code = cls["name"], cls["code"]
        text = code if code.endswith("\u200d") else code + "\u200d"

        train_crops, test_crops = [], []
        for fn in fonts:
            fpath = os.path.join(FONT_DIR, fn)
            if not validate_font(fpath, cls):
                continue
            crop = render_glyph(fpath, text)
            if crop is None:
                continue
            (test_crops if fn in RESERVED_TEST_FONTS else train_crops).append((fn, crop))

        if not train_crops:
            print(f"[WARN] No training fonts found for: {name}")
            continue
        if not test_crops:
            test_crops.append(train_crops.pop())

        for split, count in (("train", TRAIN_N), ("test", TEST_N)):
            out_dir = os.path.join(roots[split], name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            crops_pool = train_crops if split == "train" else test_crops
            rng = rng_train if split == "train" else rng_test
            is_train = split == "train"

            hw_crops = [c for c in crops_pool if any(k in c[0].lower() for k in HANDWRITING_FONTS)]
            std_crops = [c for c in crops_pool if not any(k in c[0].lower() for k in HANDWRITING_FONTS)]

            for i in range(count):
                if is_train and hw_crops and rng.random() < 0.70:
                    _, base_crop = hw_crops[rng.integers(0, len(hw_crops))]
                elif std_crops:
                    _, base_crop = std_crops[rng.integers(0, len(std_crops))]
                else:
                    _, base_crop = crops_pool[i % len(crops_pool)]

                img = None
                for _ in range(25):
                    candidate = augment(base_crop, rng, is_train=is_train)
                    if is_valid_image(candidate):
                        img = candidate
                        break

                if img is None:
                    canvas, _, _ = fit_to_canvas(base_crop, MARGIN)
                    img = 255 - np.clip(canvas, 0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img)

    # Save visual inspection sheet of 12 classes
    preview_rows = []
    for c in list(classes)[:12]:
        c_train_dir = os.path.join(roots["train"], c["name"])
        if os.path.exists(c_train_dir):
            fns = sorted(os.listdir(c_train_dir))[:8]
            imgs = [cv2.imread(os.path.join(c_train_dir, f), cv2.IMREAD_GRAYSCALE) for f in fns if f.endswith(".png")]
            if imgs:
                preview_rows.append(np.hstack(imgs))

    if preview_rows:
        preview_path = os.path.join(DATA_DIR, "preview_handwritten.png")
        cv2.imwrite(preview_path, np.vstack(preview_rows))
        print(f"Visual preview saved: {preview_path}")

    print(f"Dataset ready: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
