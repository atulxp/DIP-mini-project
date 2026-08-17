from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "enhancement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_grayscale(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def generate_histogram(image, title="Histogram", output_path=None):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].plot(hist, color="black")
    ax[1].set_title(title)
    ax[1].set_xlim([0, 255])
    ax[1].set_xlabel("Intensity")
    ax[1].set_ylabel("Count")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)
    return hist


def save_original_processed_comparison(original, processed, original_title, processed_title, output_path):
    if original is None or processed is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_array = original
    processed_array = processed

    if len(original_array.shape) == 3:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_BGR2RGB)
    if len(processed_array.shape) == 3:
        processed_array = cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_array if len(original_array.shape) == 3 else original_array, cmap="gray" if len(original_array.shape) == 2 else None)
    axes[0].set_title(original_title)
    axes[0].axis("off")

    axes[1].imshow(processed_array if len(processed_array.shape) == 3 else processed_array, cmap="gray" if len(processed_array.shape) == 2 else None)
    axes[1].set_title(processed_title)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_equalization_comparison(image, output_path):
    gray = ensure_grayscale(image)
    if gray is None:
        return
    equalized = cv2.equalizeHist(gray)
    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].plot(hist_original, color="black")
    axes[0, 1].set_title("Original Histogram")
    axes[0, 1].set_xlim([0, 255])
    axes[0, 1].set_xlabel("Intensity")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].imshow(equalized, cmap="gray")
    axes[1, 0].set_title("Equalized Image")
    axes[1, 0].axis("off")

    axes[1, 1].plot(hist_equalized, color="black")
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def apply_image_negative(image):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    return cv2.bitwise_not(gray)


def apply_log_transformation(image, c=1.0):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    gray_float = gray.astype(np.float32)
    log_image = c * np.log1p(gray_float)
    log_image = (log_image / np.log1p(255.0)) * 255.0
    log_image = np.clip(log_image, 0, 255)
    return np.uint8(log_image)


def apply_gamma_transformation(image, gamma=0.5):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    gray_float = gray.astype(np.float32) / 255.0
    gamma_image = np.power(gray_float, gamma) * 255.0
    gamma_image = np.clip(gamma_image, 0, 255)
    return np.uint8(gamma_image)


def apply_contrast_stretching(image, low_in=0, high_in=255):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    low_in = max(0, min(255, low_in))
    high_in = max(0, min(255, high_in))
    if high_in <= low_in:
        raise ValueError("high_in must be greater than low_in")

    gray_float = gray.astype(np.float32)
    stretched = (gray_float - low_in) * (255.0 / (high_in - low_in))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)


def equalize_histogram(image):
    gray = ensure_grayscale(image)
    if gray is None:
        return None
    return cv2.equalizeHist(gray)


def image_addition(image1, image2):
    img1 = image1.astype(np.int16)
    img2 = image2.astype(np.int16)
    result = img1 + img2
    return np.clip(result, 0, 255).astype(np.uint8)


def image_subtraction(image1, image2):
    img1 = image1.astype(np.int16)
    img2 = image2.astype(np.int16)
    result = img1 - img2
    return np.clip(result, 0, 255).astype(np.uint8)


def image_averaging(image1, image2):
    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)
    result = (img1 + img2) / 2.0
    return np.clip(result, 0, 255).astype(np.uint8)


def plot_enhancement_comparison(image_paths):
    image_paths = [str(path) for path in image_paths]
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    techniques = {
        "Original": lambda img: img,
        "Negative": lambda img: apply_image_negative(img),
        "Log": lambda img: apply_log_transformation(img, c=1.0),
        "Gamma": lambda img: apply_gamma_transformation(img, gamma=0.5),
        "Contrast Stretching": lambda img: apply_contrast_stretching(img, 20, 220),
    }

    fig, axes = plt.subplots(len(image_paths), len(techniques) + 1, figsize=(18, 6 * len(image_paths)))
    for row_index, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
        if gray is None:
            continue
        axes[row_index, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row_index, 0].set_title("Original")
        axes[row_index, 0].axis("off")

        for col_index, (name, func) in enumerate(list(techniques.items())[1:], start=1):
            processed = func(gray)
            axes[row_index, col_index].imshow(processed, cmap="gray")
            axes[row_index, col_index].set_title(name)
            axes[row_index, col_index].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "enhancement_comparison.png")
    plt.close(fig)
    print(f"Saved comparison output: {output_dir / 'enhancement_comparison.png'}")


def save_enhancement_output(image, output_name):
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), image)
    return output_path
