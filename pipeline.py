import cv2
import os
import numpy as np


def extract_dataset(split="train"):
    parquet_path = f"dataset/data/{split}.parquet"
    output_dir = f"dataset/images/{split}/images"
    metadata_path = f"dataset/images/{split}/metadata.json"

    if not os.path.exists(parquet_path):
        print(f"Warning: Dataset file {parquet_path} not found.")
        return

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Dataset for '{split}' appears to be already extracted.")
        return

    import pandas as pd
    import json

    print(f"Extracting '{split}' dataset from {parquet_path}...")
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    df = pd.read_parquet(parquet_path)
    if "image.bytes" in df.columns:
        df = df.rename(
            columns={"image.bytes": "image_bytes", "image.path": "image_path"}
        )

    try:
        from tqdm import tqdm

        iterable = tqdm(df.itertuples(index=False), total=len(df))
    except ImportError:
        iterable = df.itertuples(index=False)

    for row in iterable:
        img_bytes = row.image_bytes
        text = row.label

        name, ext = os.path.splitext(row.filename)
        try:
            filename = f"{int(name):04d}{ext}"
        except ValueError:
            filename = f"{name}{ext}"

        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(img_bytes)

        metadata.append({"filename": filename, "text": text})

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(metadata)} images to {output_dir}.\n")


def run_pipeline():
    extract_dataset(split="train")

    INPUT_DIR = "dataset/images/train/images"
    OUTPUT_DIR = "output/segmented-characters"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(image_files)} images to process.")
    image_files = image_files[0:5]

    for i, image_filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(image_files)}: {image_filename}")
        input_path = os.path.join(INPUT_DIR, image_filename)
        base_name = os.path.splitext(image_filename)[0]

        # --- 1. PREPROCESSING ---
        original_img = cv2.imread(input_path)
        if original_img is None:
            print(f"Failed to read {input_path}")
            continue

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        adaptive_img = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
        )
        padded = cv2.copyMakeBorder(
            adaptive_img,
            top=4,
            bottom=4,
            left=4,
            right=4,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )

        # --- 2. SHIROREKHA DETECTION AND REMOVAL ---
        img_sh = padded
        binary_sh = (img_sh == 0).astype("uint8")

        horizontal_profile = np.sum(binary_sh, axis=1)
        max_val = np.max(horizontal_profile)
        if max_val == 0:
            continue

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
                    black_pixels = np.sum(img_sh[r] == 0)
                    coverages.append(black_pixels / w)
                if np.mean(coverages) > 0.6:
                    valid_bands.append(band)

        shirorekha_removed = img_sh.copy()
        for band in valid_bands:
            mid_row = int(np.mean(band))
            thickness = 1
            for t in range(-thickness, thickness + 1):
                r = mid_row + t
                if 0 <= r < shirorekha_removed.shape[0]:
                    shirorekha_removed[r, :] = 255

        # Vertical profile reconnection
        reconnected = shirorekha_removed.copy()
        if len(valid_bands) > 0:
            cut_rows = [int(np.mean(band)) for band in valid_bands]
            zone_top = max(0, min(cut_rows) - 3)
            zone_bottom = min(img_sh.shape[0], max(cut_rows) + 3)
            max_gap = 5

            binary_rec = (shirorekha_removed == 0).astype("uint8")
            vertical_profile = np.sum(binary_rec, axis=0)
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

        # --- 3. SEGMENTATION ---
        final_prep_img = reconnected
        binary = (final_prep_img == 0).astype("uint8")

        num_labels_init, _, stats_init, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels_init <= 1:
            continue

        heights = stats_init[1:, cv2.CC_STAT_HEIGHT]
        centers_y = stats_init[1:, cv2.CC_STAT_TOP] + (heights / 2)

        median_height = np.median(heights)
        median_center = np.median(centers_y)

        middle_top = int(median_center - 0.55 * median_height)
        middle_bottom = int(median_center + 0.55 * median_height)

        split_binary = binary.copy()
        search_margin = int(median_height * 0.20)

        for p in range(1, num_labels_init):
            x, y, w_comp, h = stats_init[p, :4]
            bottom_y = y + h
            if y < middle_bottom and bottom_y > (middle_bottom + search_margin):
                roi = split_binary[y:bottom_y, x : x + w_comp]
                local_middle = middle_bottom - y
                search_start = max(0, local_middle - search_margin)
                search_end = min(roi.shape[0], local_middle + search_margin)

                if search_start < search_end:
                    row_sums = np.sum(roi[search_start:search_end, :], axis=1)
                    if len(row_sums) > 0:
                        local_split_y = search_start + np.argmin(row_sums)
                        global_split_y = y + local_split_y
                        split_binary[
                            max(0, global_split_y - 1) : global_split_y, x : x + w_comp
                        ] = 0

        # Fill horizontal gaps in lower zone
        zone_top_fill = middle_bottom
        zone_bottom_fill = split_binary.shape[0]
        image_width = split_binary.shape[1]
        max_gap_fill = 4

        for r in range(zone_top_fill, zone_bottom_fill):
            c = 0
            while c < image_width:
                if split_binary[r, c] == 1:
                    c2 = c + 1
                    gap = 0
                    while c2 < image_width and split_binary[r, c2] == 0:
                        gap += 1
                        if gap > max_gap_fill:
                            break
                        c2 += 1
                    if c2 < image_width and 0 < gap <= max_gap_fill:
                        split_binary[r, c + 1 : c2] = 1
                        c = c2
                    else:
                        c += 1
                else:
                    c += 1

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            split_binary, connectivity=8
        )

        categorized_components = {"upper": [], "middle": [], "lower": []}

        for p in range(1, num_labels):
            x, y, w_comp, h = stats[p, :4]
            # Ignore extremely small noise
            if w_comp * h < 5:
                continue

            cy = y + (h / 2)
            if cy < middle_top:
                zone = "upper"
            elif cy > middle_bottom:
                zone = "lower"
            else:
                zone = "middle"

            crop = final_prep_img[
                max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w_comp + 1
            ]
            padded_crop = cv2.copyMakeBorder(
                crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
            )

            categorized_components[zone].append(
                {"x": x, "y": y, "w": w_comp, "h": h, "crop": padded_crop}
            )

        # --- 4. SAVING ---
        bbox_img = cv2.cvtColor(split_binary * 255, cv2.COLOR_GRAY2BGR)

        image_out_dir = os.path.join(OUTPUT_DIR, base_name)
        for zone in ["upper", "middle", "lower"]:
            os.makedirs(os.path.join(image_out_dir, zone), exist_ok=True)

        for zone in categorized_components:
            categorized_components[zone].sort(key=lambda item: item["x"])

            if zone == "upper":
                color = (255, 0, 0)
            elif zone == "middle":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            for idx, comp in enumerate(categorized_components[zone]):
                cx, cy, cw, ch = comp["x"], comp["y"], comp["w"], comp["h"]
                cropped = comp["crop"]
                cv2.rectangle(bbox_img, (cx, cy), (cx + cw, cy + ch), color, 1)

                fname = f"{idx:02d}_x{cx}.png"
                cv2.imwrite(os.path.join(image_out_dir, zone, fname), cropped)

        bbox_save_path = os.path.join(image_out_dir, "bboxes.png")
        cv2.imwrite(bbox_save_path, bbox_img)

    print("Process successfully completed!")


if __name__ == "__main__":
    run_pipeline()
