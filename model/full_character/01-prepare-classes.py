import os
import cv2
import numpy as np
from tqdm import tqdm


def process_image(img_path, output_path):
    original_img = cv2.imread(img_path)
    if original_img is None:
        return

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Binarize with Otsu
    _, fixed_bin = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    white_pixels = np.sum(fixed_bin == 255)
    black_pixels = np.sum(fixed_bin == 0)

    if white_pixels > black_pixels:
        img_sh = fixed_bin
    else:
        img_sh = cv2.bitwise_not(fixed_bin)

    binary_sh = (img_sh == 0).astype("uint8")

    horizontal_profile = np.sum(binary_sh, axis=1)
    max_val = np.max(horizontal_profile)
    if max_val == 0:
        cv2.imwrite(output_path, img_sh)
        return

    headline_threshold = 0.7 * max_val
    headline_rows = np.where(horizontal_profile >= headline_threshold)[0]

    valid_bands = []
    if len(headline_rows) > 0:
        bands = []
        current_band = [headline_rows[0]]
        for j in range(1, len(headline_rows)):
            if headline_rows[j] == headline_rows[j - 1] + 1:
                current_band.append(headline_rows[j])
            else:
                bands.append(current_band)
                current_band = [headline_rows[j]]
        bands.append(current_band)

        w = img_sh.shape[1]
        for band in bands:
            coverages = []
            for r in band:
                row_black_pixels = np.sum(img_sh[r] == 0)
                coverages.append(row_black_pixels / w)
            if np.mean(coverages) > 0.4:
                valid_bands.append(band)

    shirorekha_removed = img_sh.copy()
    for band in valid_bands:
        mid_row = int(np.mean(band))
        thickness = 1
        for t in range(-thickness, thickness + 1):
            r = mid_row + t
            if 0 <= r < shirorekha_removed.shape[0]:
                shirorekha_removed[r, :] = 255

    reconnected = shirorekha_removed.copy()
    if len(valid_bands) > 0:
        cut_rows = [int(np.mean(band)) for band in valid_bands]
        zone_top = max(0, min(cut_rows) - 3)
        zone_bottom = min(img_sh.shape[0], max(cut_rows) + 3)
        max_gap = 5

        binary_rec = (shirorekha_removed == 0).astype("uint8")
        vertical_profile = np.sum(binary_rec, axis=0)
        if np.max(vertical_profile) > 0:
            threshold = 0.75 * np.max(vertical_profile)
            candidate_cols = np.where(vertical_profile >= threshold)[0]

            for col in candidate_cols:
                r = zone_top
                while r < zone_bottom:
                    if reconnected[r, col] == 0:
                        r2 = r + 1
                        gap = 0
                        while r2 < zone_bottom and reconnected[r2, col] == 255:
                            gap += 1
                            if gap > max_gap:
                                break
                            r2 += 1
                        if r2 < zone_bottom and 0 < gap <= max_gap:
                            for fill_r in range(r + 1, r2):
                                reconnected[fill_r, col] = 0
                            r = r2
                        else:
                            r += 1
                    else:
                        r += 1

    cv2.imwrite(output_path, reconnected)


def main():
    base_path = os.path.join(os.path.dirname(__file__), "dataset")
    sets = [("training", "train"), ("testing", "test")]

    for input_name, output_name in sets:
        input_base_dir = os.path.join(base_path, input_name)
        output_base_dir = os.path.join(base_path, output_name)

        if not os.path.exists(input_base_dir):
            print(f"Skipping {input_name}: Directory not found.")
            continue

        os.makedirs(output_base_dir, exist_ok=True)
        classes = sorted(
            [
                d
                for d in os.listdir(input_base_dir)
                if os.path.isdir(os.path.join(input_base_dir, d))
            ]
        )

        print(f"Processing {len(classes)} classes from {input_name}...")
        for class_name in tqdm(classes, desc=f"Processing {input_name}"):
            input_class_dir = os.path.join(input_base_dir, class_name)
            output_class_dir = os.path.join(output_base_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            images = [
                f
                for f in os.listdir(input_class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            for img_name in images:
                process_image(
                    os.path.join(input_class_dir, img_name),
                    os.path.join(output_class_dir, img_name),
                )


if __name__ == "__main__":
    main()
