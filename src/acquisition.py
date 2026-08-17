from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "traffic_signs"
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_output_folders():
    folders = [
        OUTPUT_DIR / "acquisition",
        OUTPUT_DIR / "color_spaces",
        OUTPUT_DIR / "sampling",
        OUTPUT_DIR / "quantization",
        OUTPUT_DIR / "formats",
        OUTPUT_DIR / "enhancement",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def list_dataset_images(dataset_dir=DATASET_DIR):
    if not dataset_dir.exists():
        return []
    valid_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_extensions],
        key=lambda p: p.name.lower(),
    )


def create_placeholder_sample(output_path=None):
    if output_path is None:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATASET_DIR / "sample_traffic_sign.png"

    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:] = (30, 30, 80)
    cv2.circle(image, (150, 150), 90, (0, 0, 255), -1)
    cv2.circle(image, (150, 150), 55, (0, 255, 255), -1)
    cv2.rectangle(image, (80, 50), (220, 250), (0, 255, 0), 8)

    cv2.imwrite(str(output_path), image)
    return output_path


def get_sample_image_path():
    dataset_images = list_dataset_images()
    if dataset_images:
        return dataset_images[0]
    return create_placeholder_sample()


def load_image(path):
    image_path = Path(path)
    if not image_path.exists():
        print(f"Image not found: {path}")
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read image: {path}")
        return None
    return image


def display_image(image, title="Image"):
    if image is None:
        return
    if len(image.shape) == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
    else:
        plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_image(image, path, title="Image"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(image.shape) == 3:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)
    print(f"Saved {title} to: {path}")


def get_image_details(image):
    if image is None:
        return {}
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    size_in_pixels = height * width
    dtype = image.dtype
    details = {
        "shape": (height, width, channels) if channels > 1 else (height, width),
        "height": height,
        "width": width,
        "channels": channels,
        "pixels": size_in_pixels,
        "data_type": str(dtype),
    }
    return details


def choose_coordinate(shape):
    height, width = shape[:2]
    while True:
        try:
            x = int(input(f"Enter x coordinate (0 to {width - 1}): ").strip())
            y = int(input(f"Enter y coordinate (0 to {height - 1}): ").strip())
            if 0 <= x < width and 0 <= y < height:
                return (x, y)
            print("Coordinate is out of range. Please try again.")
        except ValueError:
            print("Invalid coordinate. Please enter numeric values.")


def show_color_space_conversion(image):
    if image is None:
        return
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original/BGR")
    axes[0].axis("off")

    axes[1].imshow(grayscale, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(rgb)
    axes[2].set_title("RGB")
    axes[2].axis("off")

    axes[3].imshow(hsv)
    axes[3].set_title("HSV")
    axes[3].axis("off")

    output_path = OUTPUT_DIR / "color_spaces" / "color_space_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved color-space comparison: {output_path}")
    plt.show()


def sampling_comparison(image):
    if image is None:
        return
    scales = [1.0, 0.5, 0.25]
    output_dir = OUTPUT_DIR / "sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    resized_images = []
    for scale in scales:
        height = max(1, int(image.shape[0] * scale))
        width = max(1, int(image.shape[1] * scale))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        resized_images.append((scale, resized))
        filename = f"sample_{int(scale * 100)}.png"
        save_image(resized, output_dir / filename, f"{int(scale * 100)}% sampling")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (scale, resized) in zip(axes, resized_images):
        ax.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{int(scale * 100)}%")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_comparison.png")
    plt.close(fig)
    plt.show()


def gray_level_quantization(image):
    if image is None:
        return
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quantization_levels = [256, 128, 64, 32]
    output_dir = OUTPUT_DIR / "quantization"
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized_images = []
    for levels in quantization_levels:
        if levels == 256:
            quantized = grayscale.copy()
        else:
            levels = max(2, levels)
            step = 256 / levels
            quantized = np.floor(grayscale / step) * step
            quantized = quantized.astype(np.uint8)
        quantized_images.append((levels, quantized))
        filename = f"quantization_{levels}.png"
        save_image(quantized, output_dir / filename, f"Gray-level quantization {levels}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (levels, quantized) in zip(axes, quantized_images):
        ax.imshow(quantized, cmap="gray")
        ax.set_title(f"{levels} levels")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "quantization_comparison.png")
    plt.close(fig)
    plt.show()


def compare_file_formats(image):
    if image is None:
        return
    output_dir = OUTPUT_DIR / "formats"
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [
        ("bmp", "bmp"),
        ("png", "png"),
        ("jpeg", "jpg"),
    ]
    file_info = []
    for name, extension in formats:
        path = output_dir / f"image_original.{extension}"
        cv2.imwrite(str(path), image)
        size = path.stat().st_size
        file_info.append((name.upper(), str(path), size))

    print("\nFile format comparison:")
    for format_name, path, size in file_info:
        print(f"- {format_name}: {path} | size: {size} bytes")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    text = "\n".join(f"{name}: {size} bytes" for name, _, size in file_info)
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")
    ax.set_title("BMP, PNG, JPEG Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "format_comparison_summary.png")
    plt.close(fig)


def save_image_comparison(title, image, output_path):
    save_image(image, output_path, title)
