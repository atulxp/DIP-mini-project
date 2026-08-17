from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition import DATASET_DIR, OUTPUT_DIR, list_dataset_images, load_image, save_image

matplotlib.use("Agg")

FILTER_OUTPUT_DIR = OUTPUT_DIR / "filtering"
NOISE_DIR = FILTER_OUTPUT_DIR / "noise"
SMOOTHING_DIR = FILTER_OUTPUT_DIR / "smoothing"
SHARPENING_DIR = FILTER_OUTPUT_DIR / "sharpening"
COMPARISON_DIR = FILTER_OUTPUT_DIR / "comparison"
PIPELINE_DIR = FILTER_OUTPUT_DIR / "pipeline"


def ensure_filter_directories():
    for folder in [NOISE_DIR, SMOOTHING_DIR, SHARPENING_DIR, COMPARISON_DIR, PIPELINE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def select_test_images(count=3):
    images = list_dataset_images()
    if len(images) < count:
        return images
    return images[:count]


def add_salt_and_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    if image is None:
        return None
    noisy = image.copy().astype(np.float32)
    h, w = noisy.shape[:2]
    total_pixels = h * w
    salt_count = int(total_pixels * amount * salt_vs_pepper)
    pepper_count = int(total_pixels * amount * (1.0 - salt_vs_pepper))

    rng = np.random.default_rng()
    y_coords = rng.integers(0, h, size=salt_count)
    x_coords = rng.integers(0, w, size=salt_count)
    noisy[y_coords, x_coords] = 255

    y_coords = rng.integers(0, h, size=pepper_count)
    x_coords = rng.integers(0, w, size=pepper_count)
    noisy[y_coords, x_coords] = 0

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_noise(image, sigma=25):
    if image is None:
        return None
    noisy = image.astype(np.float32)
    noise = np.random.normal(0, sigma, noisy.shape).astype(np.float32)
    noisy = noisy + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_noise_examples(image_paths=None):
    ensure_filter_directories()
    if image_paths is None:
        image_paths = select_test_images(3)

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        image = load_image(str(image_path))
        if image is None:
            continue

        salt_pepper = add_salt_and_pepper_noise(image, amount=0.02)
        gaussian = add_gaussian_noise(image, sigma=25)

        salt_name = f"traffic_sign_{index:02d}_salt_pepper.png"
        gaussian_name = f"traffic_sign_{index:02d}_gaussian.png"

        save_image(image, NOISE_DIR / f"traffic_sign_{index:02d}_original.png", "Original image")
        save_image(salt_pepper, NOISE_DIR / salt_name, "Salt-and-pepper noise")
        save_image(gaussian, NOISE_DIR / gaussian_name, "Gaussian noise")

        results.append({
            "original": image,
            "salt_pepper": salt_pepper,
            "gaussian": gaussian,
            "name": image_path.name,
        })

    return results


def apply_mean_filter(image, kernel_size=3):
    if image is None:
        return None
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_filter(image, kernel_size=3, sigma=1.0):
    if image is None:
        return None
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def apply_median_filter(image, kernel_size=3):
    if image is None:
        return None
    return cv2.medianBlur(image, kernel_size)


def apply_laplacian_sharpen(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpened = gray.astype(np.float32) - lap
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened.astype(np.uint8)


def apply_gradient_sharpen(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    sharpened = gray.astype(np.float32) + gradient
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened.astype(np.uint8)


def apply_high_boost(image, alpha=2.5):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    high_boost = gray.astype(np.float32) + alpha * (gray.astype(np.float32) - blurred.astype(np.float32))
    high_boost = np.clip(high_boost, 0, 255)
    return high_boost.astype(np.uint8)


def compute_mse_psnr(original, filtered):
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    return mse, psnr


def save_smoothing_comparison(original, noisy, output_path):
    if original is None or noisy is None:
        return

    results = {
        "Mean 3x3": apply_mean_filter(noisy, 3),
        "Mean 5x5": apply_mean_filter(noisy, 5),
        "Gaussian 3x3": apply_gaussian_filter(noisy, 3, 1.0),
        "Gaussian 5x5": apply_gaussian_filter(noisy, 5, 1.5),
        "Median 3x3": apply_median_filter(noisy, 3),
        "Median 5x5": apply_median_filter(noisy, 5),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panel_names = ["Original", "Noisy", "Mean 3x3", "Mean 5x5", "Gaussian 3x3", "Gaussian 5x5", "Median 3x3", "Median 5x5"]
    panel_images = [
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Mean 3x3"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Mean 5x5"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Gaussian 3x3"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Gaussian 5x5"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Median 3x3"], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(results["Median 5x5"], cv2.COLOR_BGR2RGB),
    ]

    for ax, title, image in zip(axes.flat, panel_names, panel_images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sharpening_comparison(original, output_path):
    if original is None:
        return

    lap = apply_laplacian_sharpen(original)
    grad = apply_gradient_sharpen(original)
    high_boost = apply_high_boost(original)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        lap,
        grad,
        high_boost,
    ]
    titles = ["Original", "Laplacian", "Gradient Sharpening", "High-Boost"]

    for ax, image, title in zip(axes, images, titles):
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compare_filters_on_dataset(image_paths):
    ensure_filter_directories()
    summary_lines = []
    selected = image_paths or select_test_images(3)

    for idx, image_path in enumerate(selected, start=1):
        original = load_image(str(image_path))
        if original is None:
            continue

        noisy_types = {
            "salt_pepper": add_salt_and_pepper_noise(original, amount=0.02),
            "gaussian": add_gaussian_noise(original, sigma=20),
        }

        for noise_name, noisy in noisy_types.items():
            output_name = f"{Path(image_path).stem}_{noise_name}_comparison.png"
            compare_path = COMPARISON_DIR / output_name
            save_smoothing_comparison(original, noisy, compare_path)

            metrics = {
                "Mean 3x3": compute_mse_psnr(original, apply_mean_filter(noisy, 3)),
                "Mean 5x5": compute_mse_psnr(original, apply_mean_filter(noisy, 5)),
                "Gaussian 3x3": compute_mse_psnr(original, apply_gaussian_filter(noisy, 3, 1.0)),
                "Gaussian 5x5": compute_mse_psnr(original, apply_gaussian_filter(noisy, 5, 1.5)),
                "Median 3x3": compute_mse_psnr(original, apply_median_filter(noisy, 3)),
                "Median 5x5": compute_mse_psnr(original, apply_median_filter(noisy, 5)),
            }

            best_filter = max(metrics, key=lambda k: metrics[k][1])
            summary_lines.append(f"{image_path.name} | {noise_name} | best filter: {best_filter} | PSNR: {metrics[best_filter][1]:.2f} dB")

    report_path = COMPARISON_DIR / "best_filter_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return summary_lines


def choose_best_filter_for_report(image_paths):
    selected = image_paths or select_test_images(3)
    results = {
        "salt_pepper": {},
        "gaussian": {},
    }

    for image_path in selected:
        original = load_image(str(image_path))
        if original is None:
            continue

        noisy = {
            "salt_pepper": add_salt_and_pepper_noise(original, amount=0.02),
            "gaussian": add_gaussian_noise(original, sigma=20),
        }

        for noise_name, image in noisy.items():
            filters = {
                "Mean 3x3": apply_mean_filter(image, 3),
                "Mean 5x5": apply_mean_filter(image, 5),
                "Gaussian 3x3": apply_gaussian_filter(image, 3, 1.0),
                "Gaussian 5x5": apply_gaussian_filter(image, 5, 1.5),
                "Median 3x3": apply_median_filter(image, 3),
                "Median 5x5": apply_median_filter(image, 5),
            }
            psnr_values = {}
            for name, filtered in filters.items():
                _, psnr = compute_mse_psnr(original, filtered)
                psnr_values[name] = psnr
            best_name = max(psnr_values, key=psnr_values.get)
            results[noise_name][Path(image_path).stem] = best_name

    return results


def create_pipeline_comparison(image_path):
    ensure_filter_directories()
    original = load_image(str(image_path))
    if original is None:
        return None
    noisy = add_salt_and_pepper_noise(original, amount=0.02)
    filtered = apply_median_filter(noisy, 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Filtered Output")
    axes[2].axis("off")

    fig.tight_layout()
    output_path = PIPELINE_DIR / "enhancement_filtering_pipeline.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
