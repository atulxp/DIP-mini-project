"""Segmentation and region-based experiments for the traffic-sign pipeline."""

from collections import deque
from pathlib import Path
import csv
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition import OUTPUT_DIR, list_dataset_images, load_image, save_image
from src.filtering import apply_median_filter

matplotlib.use("Agg")

SEGMENTATION_DIR = OUTPUT_DIR / "segmentation"
REGION_GROWING_DIR = OUTPUT_DIR / "region_growing"
REGION_SPLIT_MERGE_DIR = OUTPUT_DIR / "region_splitting_merging"


def ensure_segmentation_directories():
    for folder in (SEGMENTATION_DIR, REGION_GROWING_DIR, REGION_SPLIT_MERGE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def to_gray(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def segmentation_input(image):
    """Reuse Task 4's best salt-and-pepper filter before segmentation."""
    return apply_median_filter(image, 3) if image is not None else None


def _normalized_uint8(values):
    values = np.abs(values).astype(np.float32)
    maximum = float(values.max()) if values.size else 0.0
    return np.zeros(values.shape, dtype=np.uint8) if maximum == 0 else np.clip(values * 255.0 / maximum, 0, 255).astype(np.uint8)


def roberts_operator(gray):
    gray = to_gray(gray)
    gx = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, 0], [0, -1]], dtype=np.float32))
    gy = cv2.filter2D(gray, cv2.CV_32F, np.array([[0, 1], [-1, 0]], dtype=np.float32))
    return _normalized_uint8(cv2.magnitude(gx, gy))


def prewitt_operator(gray):
    gray = to_gray(gray)
    gx = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32))
    gy = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32))
    return _normalized_uint8(cv2.magnitude(gx, gy))


def sobel_operator(gray):
    gray = to_gray(gray)
    return _normalized_uint8(cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))


def laplacian_operator(gray):
    return _normalized_uint8(cv2.Laplacian(to_gray(gray), cv2.CV_32F))


def global_threshold(gray, threshold=128):
    return cv2.threshold(to_gray(gray), int(np.clip(threshold, 0, 255)), 255, cv2.THRESH_BINARY)[1]


def otsu_threshold(gray):
    threshold, result = cv2.threshold(to_gray(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold), result


def adaptive_threshold(gray, block_size=31, constant=5):
    block_size = max(3, int(block_size) | 1)
    return cv2.adaptiveThreshold(to_gray(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)


def edge_pixel_count(edge_image, relative_threshold=0.25):
    return int(np.count_nonzero(edge_image >= int(255 * relative_threshold)))


def binary_components(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    return max(0, count - 1), int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def _save_task5_figure(name, original, filtered, gray, edges, thresholds):
    panels = [("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), ("Task 4 output", filtered), ("Grayscale", gray)]
    panels.extend(edges.items())
    panels.extend(thresholds.items())
    columns = 5
    fig, axes = plt.subplots(int(np.ceil(len(panels) / columns)), columns, figsize=(18, 7))
    axes = np.atleast_1d(axes).ravel()
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    fig.suptitle(f"Task 5 segmentation evidence: {name}")
    fig.tight_layout()
    path = SEGMENTATION_DIR / f"{Path(name).stem}_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run_task5_experiments(image_paths=None, global_value=128, adaptive_block=31, adaptive_constant=5):
    ensure_segmentation_directories()
    selected = list(image_paths or list_dataset_images()[:3])[:3]
    rows, scores = [], {"Global Threshold": [], "Otsu": [], "Adaptive Threshold": []}
    for image_path in selected:
        original = load_image(str(image_path))
        if original is None:
            continue
        filtered = segmentation_input(original)
        gray = to_gray(filtered)
        edges = {"Roberts": roberts_operator(gray), "Prewitt": prewitt_operator(gray), "Sobel": sobel_operator(gray), "Laplacian": laplacian_operator(gray)}
        otsu_value, otsu = otsu_threshold(gray)
        thresholds = {"Global Threshold": global_threshold(gray, global_value), f"Otsu ({otsu_value})": otsu, "Adaptive Threshold": adaptive_threshold(gray, adaptive_block, adaptive_constant)}
        _save_task5_figure(image_path.name, original, filtered, gray, edges, thresholds)
        save_image(filtered, SEGMENTATION_DIR / f"{Path(image_path).stem}_task4_median3x3.png", "Task 4 filtered input")
        for method, result in (("Global Threshold", thresholds["Global Threshold"]), ("Otsu", otsu), ("Adaptive Threshold", thresholds["Adaptive Threshold"])):
            components, largest = binary_components(result)
            ratio = float(np.mean(result > 0))
            scores[method].append(abs(ratio - 0.30) + min(components, 1000) / max(1, gray.size) * 4)
            rows.append({"image": image_path.name, "dimensions": f"{gray.shape[1]}x{gray.shape[0]}", "min_intensity": int(gray.min()), "max_intensity": int(gray.max()), "method": method, "threshold": otsu_value if method == "Otsu" else global_value if method == "Global Threshold" else f"block={adaptive_block},C={adaptive_constant}", "foreground_ratio": f"{ratio:.4f}", "components": components, "largest_component_pixels": largest})
        for method, result in edges.items():
            rows.append({"image": image_path.name, "dimensions": f"{gray.shape[1]}x{gray.shape[0]}", "min_intensity": int(gray.min()), "max_intensity": int(gray.max()), "method": method, "threshold": "", "foreground_ratio": "", "components": "", "largest_component_pixels": edge_pixel_count(result)})
    best_method = min(scores, key=lambda method: float(np.mean(scores[method]))) if scores else "Otsu"
    csv_path = SEGMENTATION_DIR / "task5_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    report_path = SEGMENTATION_DIR / "task5_summary.md"
    report_lines = [
        "# Task 5 - Traffic-sign segmentation", "",
        "## Aim", "Separate traffic-sign structure from its background using edge detection and thresholding.", "",
        "## Pipeline input", "Each experiment starts with the selected dataset image, applies the existing Task 4 Median 3x3 filtered output, and then converts that result to grayscale. It does not load an unrelated image.", "",
        f"## Best Segmentation Method\n**{best_method}**", "",
        "The choice is based on the measured foreground ratio and connected-component fragmentation in the selected images. The score favors a compact foreground near 30% and penalizes fragmentation; it is a transparent heuristic, not a fabricated accuracy value.", "",
        "## Measured comparison", "| Image | Dimensions | Min | Max | Method | Threshold | Foreground ratio | Components | Edge/largest-component pixels |", "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(f"| {row['image']} | {row['dimensions']} | {row['min_intensity']} | {row['max_intensity']} | {row['method']} | {row['threshold']} | {row['foreground_ratio']} | {row['components']} | {row['largest_component_pixels']} |")
    report_lines.extend(["", "## Interpretation", "Roberts, Prewitt, Sobel, and Laplacian are saved in each per-image comparison figure. Edge pixel counts describe detected response strength above 25% of the normalized response and are not an accuracy score. Threshold rows report actual foreground ratios and connected components, so broken edges, false edges, missing regions, and extra regions can be discussed from the figures alongside this table.", ""])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"best_method": best_method, "rows": rows, "csv": csv_path, "report": report_path}


def neighbourhoods(connectivity):
    return ((-1, 0), (1, 0), (0, -1), (0, 1)) if int(connectivity) == 4 else ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def region_grow(gray, seed, threshold, connectivity=8):
    gray = to_gray(gray)
    height, width = gray.shape
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Seed point is outside the image")
    accepted = np.zeros((height, width), dtype=bool)
    queued = np.zeros((height, width), dtype=bool)
    queue, region_mean, region_size = deque([(x, y)]), float(gray[y, x]), 0
    queued[y, x] = True
    while queue:
        current_x, current_y = queue.popleft()
        value = float(gray[current_y, current_x])
        if abs(value - region_mean) > float(threshold):
            continue
        accepted[current_y, current_x], region_size = True, region_size + 1
        region_mean += (value - region_mean) / region_size
        for offset_x, offset_y in neighbourhoods(connectivity):
            nx, ny = current_x + offset_x, current_y + offset_y
            if 0 <= nx < width and 0 <= ny < height and not queued[ny, nx]:
                queued[ny, nx] = True
                if abs(float(gray[ny, nx]) - region_mean) <= float(threshold):
                    queue.append((nx, ny))
    return accepted.astype(np.uint8) * 255, region_mean


def run_region_growing_experiments(image_path=None, seed=None, thresholds=(10, 25, 45), connectivity_values=(4, 8)):
    ensure_segmentation_directories()
    image_path = image_path or list_dataset_images()[0]
    original = load_image(str(image_path))
    if original is None:
        raise ValueError("Could not load the region-growing image")
    filtered = segmentation_input(original)
    gray = to_gray(filtered)
    height, width = gray.shape
    seed = seed or (width // 2, height // 2)
    rows, panels = [], [("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), ("Grayscale / Task 4 output", gray)]
    for connectivity in connectivity_values:
        for threshold in thresholds:
            started = time.perf_counter()
            mask, mean = region_grow(gray, seed, threshold, connectivity)
            elapsed = (time.perf_counter() - started) * 1000
            panels.append((f"{connectivity}-connected, T={threshold}, pixels={int(np.count_nonzero(mask))}", mask))
            rows.append({"image": image_path.name, "seed_x": seed[0], "seed_y": seed[1], "connectivity": connectivity, "threshold": threshold, "pixels": int(np.count_nonzero(mask)), "processing_ms": f"{elapsed:.3f}", "final_mean": f"{mean:.2f}"})
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, (title, image) in zip(np.atleast_1d(axes).ravel(), panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.suptitle(f"Task 6 region growing: seed={seed}")
    fig.tight_layout()
    figure_path = REGION_GROWING_DIR / f"{Path(image_path).stem}_experiments.png"
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    csv_path = REGION_GROWING_DIR / "region_growing_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    bounded_rows = [row for row in rows if int(row["pixels"]) <= int(gray.size * 0.10)]
    best = max(bounded_rows or rows, key=lambda row: int(row["pixels"]))
    summary_path = REGION_GROWING_DIR / "task6_summary.md"
    summary_lines = [
        "# Task 6 - Region growing", "", f"Image: `{image_path.name}`", f"Dimensions: `{gray.shape[1]}x{gray.shape[0]}`; grayscale min/max: `{int(gray.min())}/{int(gray.max())}`", f"Seed strategy: automatic centre seed `{seed}`", "Criterion: a candidate pixel is accepted when its intensity differs from the current region mean by no more than the selected threshold.", "", "## Measured comparison", "| Connectivity | Threshold | Pixels | Processing time (ms) | Final mean |", "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(f"| {row['connectivity']} | {row['threshold']} | {row['pixels']} | {row['processing_ms']} | {row['final_mean']} |")
    summary_lines.extend(["", f"Best bounded configuration: **{best['connectivity']}-connected, threshold {best['threshold']}**. It is the largest measured region not exceeding 10% of the image area, a leakage guard for this traffic-sign experiment. The figure and CSV retain every threshold/connectivity result.", ""])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return {"figure": figure_path, "csv": csv_path, "summary": summary_path, "rows": rows, "best": best, "seed": seed}


def split_region(gray, homogeneity_threshold=12, min_size=16):
    gray = to_gray(gray)
    height, width = gray.shape
    regions = []
    def visit(x, y, region_width, region_height):
        block = gray[y:y + region_height, x:x + region_width]
        if block.size == 0:
            return
        if float(block.std()) <= homogeneity_threshold or min(region_width, region_height) <= min_size:
            regions.append((x, y, region_width, region_height, float(block.mean()), float(block.std())))
            return
        half_width, half_height = region_width // 2, region_height // 2
        if half_width == 0 or half_height == 0:
            regions.append((x, y, region_width, region_height, float(block.mean()), float(block.std())))
            return
        for next_x, next_y, next_width, next_height in ((x, y, half_width, half_height), (x + half_width, y, region_width - half_width, half_height), (x, y + half_height, half_width, region_height - half_height), (x + half_width, y + half_height, region_width - half_width, region_height - half_height)):
            visit(next_x, next_y, next_width, next_height)
    visit(0, 0, width, height)
    return regions


def _adjacent(first, second):
    x1, y1, w1, h1 = first[:4]
    x2, y2, w2, h2 = second[:4]
    horizontal = (x1 + w1 == x2 or x2 + w2 == x1) and max(y1, y2) < min(y1 + h1, y2 + h2)
    vertical = (y1 + h1 == y2 or y2 + h2 == y1) and max(x1, x2) < min(x1 + w1, x2 + w2)
    return horizontal or vertical


def merge_regions(regions, similarity_threshold=12):
    merged = [list(region) for region in regions]
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            if changed:
                break
            for second_index in range(first_index + 1, len(merged)):
                first, second = merged[first_index], merged[second_index]
                if _adjacent(first, second) and abs(first[4] - second[4]) <= similarity_threshold:
                    x, y = min(first[0], second[0]), min(first[1], second[1])
                    right, bottom = max(first[0] + first[2], second[0] + second[2]), max(first[1] + first[3], second[1] + second[3])
                    area = first[2] * first[3] + second[2] * second[3]
                    mean = (first[4] * first[2] * first[3] + second[4] * second[2] * second[3]) / area
                    merged[first_index] = [x, y, right - x, bottom - y, mean, 0.0]
                    merged.pop(second_index)
                    changed = True
                    break
    return [tuple(region) for region in merged]


def region_visualization(gray, regions, title, output_path):
    canvas = cv2.cvtColor(to_gray(gray), cv2.COLOR_GRAY2RGB)
    for x, y, width, height, _, _ in regions:
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (255, 40, 40), 1)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.imshow(canvas)
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_split_merge(image_path=None, homogeneity_threshold=12, min_size=16, similarity_threshold=12):
    ensure_segmentation_directories()
    image_path = image_path or list_dataset_images()[0]
    original = load_image(str(image_path))
    if original is None:
        raise ValueError("Could not load the split-and-merge image")
    gray = to_gray(segmentation_input(original))
    started = time.perf_counter()
    split = split_region(gray, homogeneity_threshold, min_size)
    split_time = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    merged = merge_regions(split, similarity_threshold)
    merge_time = (time.perf_counter() - started) * 1000
    split_path = REGION_SPLIT_MERGE_DIR / f"{Path(image_path).stem}_split_regions.png"
    merged_path = REGION_SPLIT_MERGE_DIR / f"{Path(image_path).stem}_merged_regions.png"
    region_visualization(gray, split, f"Split regions ({len(split)})", split_path)
    region_visualization(gray, merged, f"Merged regions ({len(merged)})", merged_path)
    report_path = REGION_SPLIT_MERGE_DIR / "task7_summary.md"
    report_path.write_text(f"# Task 7 - Region splitting and merging\n\nImage: `{Path(image_path).name}`\n\n- Homogeneity threshold (standard deviation): {homogeneity_threshold}\n- Minimum region size: {min_size}\n- Similarity threshold (mean intensity): {similarity_threshold}\n- Regions after splitting: {len(split)}\n- Regions after merging: {len(merged)}\n- Split time: {split_time:.3f} ms\n- Merge time: {merge_time:.3f} ms\n\nThe figures show measured region boundaries for comparison with Tasks 5 and 6.\n", encoding="utf-8")
    return {"split": split, "merged": merged, "split_path": split_path, "merged_path": merged_path, "report": report_path}