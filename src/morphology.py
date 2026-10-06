"""Task 8 binary morphology for the traffic-sign processing pipeline."""

from pathlib import Path
import csv
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition import OUTPUT_DIR, load_image, save_image
from src.segmentation import adaptive_threshold, global_threshold, otsu_threshold, segmentation_input, to_gray

matplotlib.use("Agg")

MORPHOLOGY_DIR = OUTPUT_DIR / "morphology"


def ensure_morphology_directories():
    MORPHOLOGY_DIR.mkdir(parents=True, exist_ok=True)


def binary_image(image, threshold_method="otsu", threshold=128, adaptive_block=31, adaptive_constant=5):
    """Return a 0/255 mask and the threshold description used to create it."""
    gray = to_gray(image)
    if gray is None or gray.size == 0:
        raise ValueError("A non-empty segmentation image is required")
    if threshold_method == "otsu":
        used_threshold, result = otsu_threshold(gray)
        description = f"Otsu ({used_threshold})"
    elif threshold_method == "global":
        used_threshold = int(np.clip(threshold, 0, 255))
        result = global_threshold(gray, used_threshold)
        description = f"Global ({used_threshold})"
    elif threshold_method == "adaptive":
        result = adaptive_threshold(gray, adaptive_block, adaptive_constant)
        description = f"Adaptive (block={max(3, int(adaptive_block) | 1)}, C={adaptive_constant})"
    else:
        raise ValueError("threshold_method must be otsu, global, or adaptive")
    return np.where(result > 0, 255, 0).astype(np.uint8), description


def structuring_element(name):
    """Create a supported binary structuring element with its display label."""
    key = name.strip().lower().replace(" ", "")
    if key in {"3x3square", "square3", "3x3"}:
        element, label = np.ones((3, 3), dtype=np.uint8), "3x3 Square"
    elif key in {"5x5square", "square5", "5x5"}:
        element, label = np.ones((5, 5), dtype=np.uint8), "5x5 Square"
    elif key in {"cross", "cross3", "3x3cross"}:
        element, label = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), "Cross"
    elif key in {"disk", "disk5", "circular"}:
        element, label = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8), "Disk"
    else:
        raise ValueError("Unknown structuring element")
    return element, label


def _validate_inputs(binary, element):
    binary, element = np.asarray(binary), np.asarray(element)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError("Binary image must be a non-empty 2-D array")
    if element.ndim != 2 or element.size == 0 or not np.any(element):
        raise ValueError("Structuring element must have foreground pixels")
    if element.shape[0] % 2 == 0 or element.shape[1] % 2 == 0:
        raise ValueError("Structuring element dimensions must be odd")
    return binary > 0, element > 0


def erode(binary, element):
    mask, kernel = _validate_inputs(binary, element)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=False)
    result = np.ones(mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(kernel)):
        result &= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
    return result.astype(np.uint8) * 255


def dilate(binary, element):
    mask, kernel = _validate_inputs(binary, element)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=False)
    result = np.zeros(mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(kernel)):
        result |= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
    return result.astype(np.uint8) * 255


def opening(binary, element):
    return dilate(erode(binary, element), element)


def closing(binary, element):
    return erode(dilate(binary, element), element)


def foreground_pixels(binary):
    return int(np.count_nonzero(np.asarray(binary) > 0))


def _operation(binary, element, operation):
    started = time.perf_counter()
    if operation == "Erosion":
        result = erode(binary, element)
    elif operation == "Dilation":
        result = dilate(binary, element)
    elif operation == "Opening":
        result = opening(binary, element)
    elif operation == "Closing":
        result = closing(binary, element)
    else:
        raise ValueError("Unsupported morphology operation")
    return result, (time.perf_counter() - started) * 1000


def _save_figure(path, panels, title):
    columns = min(4, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (label, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray")
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_task8(image_path, threshold_method="otsu", threshold=128, adaptive_block=31, adaptive_constant=5, selected_element="3x3 square", selected_operation="Erosion"):
    """Run Task 8 from the selected traffic-sign image and save measured evidence."""
    ensure_morphology_directories()
    original = load_image(str(image_path))
    if original is None:
        raise ValueError("Could not load the selected traffic-sign image")
    filtered = segmentation_input(original)
    gray = to_gray(filtered)
    binary, threshold_description = binary_image(gray, threshold_method, threshold, adaptive_block, adaptive_constant)
    base_name = Path(image_path).stem
    save_image(binary, MORPHOLOGY_DIR / f"{base_name}_binary.png", "Task 8 binary image")
    rows, panels = [], [("Original traffic sign", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), ("Final segmentation input", gray), (f"Binary ({threshold_description})", binary)]
    names = [("3x3 square", "3x3 Square"), ("5x5 square", "5x5 Square"), ("cross", "Cross"), ("disk", "Disk")]
    selected_label = structuring_element(selected_element)[1]
    before = foreground_pixels(binary)
    for element_name, element_label in names:
        element, _ = structuring_element(element_name)
        for operation in ("Erosion", "Dilation"):
            result, elapsed = _operation(binary, element, operation)
            count = foreground_pixels(result)
            rows.append({"image": Path(image_path).name, "dimensions": f"{binary.shape[1]}x{binary.shape[0]}", "threshold": threshold_description, "structuring_element": element_label, "operation": operation, "foreground_before": before, "foreground_after": count, "pixel_change": count - before, "processing_ms": f"{elapsed:.3f}"})
            save_image(result, MORPHOLOGY_DIR / f"{base_name}_{element_name}_{operation.lower()}.png", f"{element_label} {operation}")
            if element_label == selected_label and operation == selected_operation:
                panels.append((f"{element_label} {operation}", result))
    selected_element_array, _ = structuring_element(selected_element)
    selected_result, selected_time = _operation(binary, selected_element_array, selected_operation)
    if selected_operation not in {"Erosion", "Dilation"}:
        panels.append((f"{selected_label} {selected_operation}", selected_result))
    square3 = structuring_element("3x3 square")[0]
    panels.extend([("Opening (3x3 square)", opening(binary, square3)), ("Closing (3x3 square)", closing(binary, square3))])
    figure_path = MORPHOLOGY_DIR / f"{base_name}_task8_overview.png"
    _save_figure(figure_path, panels, f"Task 8 Binary Morphology: {base_name}")
    csv_path = MORPHOLOGY_DIR / "task8_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report_path = MORPHOLOGY_DIR / "task8_summary.md"
    report_lines = ["# Task 8 - Binary Morphological Processing", "", "## Aim", "Apply measured binary erosion and dilation to the final segmentation stage of the traffic-sign pipeline.", "", "## Input and binary conversion", f"- Image: `{Path(image_path).name}`", f"- Dimensions: `{binary.shape[1]}x{binary.shape[0]}`", f"- Segmentation source: Task 4 Median 3x3 filtered image followed by `{threshold_description}`", f"- Grayscale minimum/maximum: `{int(gray.min())}/{int(gray.max())}`", f"- Binary foreground/background pixels: `{before}/{binary.size - before}`", "", "## Structuring elements", "The experiment uses 3x3 square, 5x5 square, cross, and disk masks. Each output is generated by the corresponding active mask.", "", "## Erosion and dilation comparison", "| Structuring element | Operation | Foreground before | Foreground after | Pixel change | Time (ms) |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        report_lines.append(f"| {row['structuring_element']} | {row['operation']} | {row['foreground_before']} | {row['foreground_after']} | {row['pixel_change']} | {row['processing_ms']} |")
    report_lines.extend(["", "## Interpretation", "Erosion removes foreground pixels where the full selected neighbourhood is not present; dilation adds foreground pixels where any selected neighbourhood position overlaps the object. The measured pixel changes and saved figures are the basis for discussing boundary, noise, and connectivity effects on this traffic-sign image.", "", "## Applications", "Opening reuses erosion then dilation and is useful for evaluating small-noise removal. Closing reuses dilation then erosion and is useful for evaluating gaps and holes. Their usefulness for this traffic sign should be judged from the saved overview and per-element outputs.", "", f"Selected demonstration: **{selected_label} {selected_operation}**, measured in `{selected_time:.3f} ms`.", f"Overview figure: `{figure_path}`", f"Metrics CSV: `{csv_path}`", ""])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"binary": binary, "threshold": threshold_description, "rows": rows, "figure": figure_path, "csv": csv_path, "report": report_path, "selected": selected_result}