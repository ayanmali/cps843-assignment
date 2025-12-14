from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_all_data  # noqa: E402
from src.preprocessing import (
    adjust_skew_hough,
    correct_slant,
    load_img,
    preprocess,
    threshold,
)  # noqa: E402
from src.segmentation import (
    segment_characters_projection,
    visualize_segmentation,
)  # noqa: E402


def remove_fragments(char_img: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        char_img, connectivity=8
    )
    if num_labels <= 1:
        return char_img
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.zeros_like(char_img)
    clean[labels == largest_label] = 255
    return clean


def extract_segments_from_image(
    source_image: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    segments: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    img_h, img_w = source_image.shape[:2]

    for x, y, w, h in bounding_boxes:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)

        crop = source_image[y0:y1, x0:x1].copy()
        crop = remove_fragments(crop)
        segments.append((crop, (x0, y0, x1 - x0, y1 - y0)))

    return segments


def run_one(img_path: str) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    img = load_img(img_path)
    processed = preprocess(img)
    processed = adjust_skew_hough(processed)
    processed = correct_slant(processed)
    th = threshold(processed)

    segs = segment_characters_projection(th)
    bboxes = [bbox for _, bbox in segs]
    return extract_segments_from_image(processed, bboxes)


def main() -> None:
    n_samples = 200
    n_save_good = 6
    n_save_bad = 6

    out_dir = REPO_ROOT / "outputs" / "batch_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_data().head(n_samples).copy()

    rows = []
    good_saved = 0
    bad_saved = 0

    for _, row in df.iterrows():
        img_rel = str(row["image_path"])
        img_path = str(REPO_ROOT / img_rel)
        gt = str(row["GroundTruth"])

        img = load_img(img_path)
        if img is None:
            rows.append(
                {
                    "ImgName": row["ImgName"],
                    "GroundTruth": gt,
                    "image_path": img_rel,
                    "status": "load_fail",
                    "gt_len": len(gt),
                    "num_segments": -1,
                    "match": 0,
                }
            )
            continue

        processed = preprocess(img)
        processed = adjust_skew_hough(processed)
        processed = correct_slant(processed)
        th = threshold(processed)

        segs = segment_characters_projection(th)
        bboxes = [bbox for _, bbox in segs]
        chars = extract_segments_from_image(processed, bboxes)

        num_segments = len(chars)
        gt_len = len(gt)
        match = int(num_segments == gt_len)

        rows.append(
            {
                "ImgName": row["ImgName"],
                "GroundTruth": gt,
                "image_path": img_rel,
                "status": "ok",
                "gt_len": gt_len,
                "num_segments": num_segments,
                "match": match,
            }
        )

        if match and good_saved < n_save_good:
            viz_path = (
                out_dir
                / f"good_{good_saved:02d}_{str(row['ImgName']).replace('/', '_')}.png"
            )
            visualize_segmentation(processed, chars, str(viz_path))
            good_saved += 1

        if (not match) and bad_saved < n_save_bad:
            viz_path = (
                out_dir
                / f"bad_{bad_saved:02d}_{str(row['ImgName']).replace('/', '_')}.png"
            )
            visualize_segmentation(processed, chars, str(viz_path))
            bad_saved += 1

    results = pd.DataFrame(rows)
    results_path = out_dir / "batch_results.csv"
    results.to_csv(results_path, index=False)

    ok = results[results["status"] == "ok"]
    if len(ok) > 0:
        success_rate = float(ok["match"].mean())
        print(f"Processed: {len(results)} (ok={len(ok)})")
        print(f"Exact-count match rate: {success_rate:.3f}")
    else:
        print("Processed: 0 (ok=0)")

    print(f"Results: {results_path}")
    print(f"Examples: {out_dir}")


if __name__ == "__main__":
    main()
