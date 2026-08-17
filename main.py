import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from src.acquisition import (
    DATASET_DIR,
    OUTPUT_DIR,
    choose_coordinate,
    compare_file_formats,
    create_placeholder_sample,
    display_image,
    ensure_output_folders,
    get_image_details,
    get_sample_image_path,
    gray_level_quantization,
    list_dataset_images,
    load_image,
    save_image,
    save_image_comparison,
    show_color_space_conversion,
    sampling_comparison,
)
from src.enhancement import (
    apply_contrast_stretching,
    apply_gamma_transformation,
    apply_image_negative,
    apply_log_transformation,
    equalize_histogram,
    generate_histogram,
    image_addition,
    image_averaging,
    image_subtraction,
    plot_enhancement_comparison,
)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_selected_image():
    dataset_images = list_dataset_images()
    if dataset_images:
        print("Available images in dataset/traffic_signs:")
        for i, img in enumerate(dataset_images, start=1):
            print(f"  {i}. {img.name}")
        choice = input("Select image number (press Enter for first image): ").strip()
        if choice == "":
            return load_image(str(dataset_images[0]))
        try:
            index = int(choice) - 1
            return load_image(str(dataset_images[index]))
        except (ValueError, IndexError):
            print("Invalid choice. Loading the first image instead.")
            return load_image(str(dataset_images[0]))

    print("No dataset image was found. A sample placeholder image will be created for testing.")
    sample_path = create_placeholder_sample()
    return load_image(str(sample_path))


def run_acquisition_flow():
    print_header("Week 3 - Image Acquisition and Representation")
    image = load_selected_image()
    if image is None:
        print("Image could not be loaded.")
        return

    save_image(image, OUTPUT_DIR / "acquisition" / "original_image.png", "Original Image")
    display_image(image, "Original Image")

    details = get_image_details(image)
    print("Image Information:")
    print(details)

    coord = choose_coordinate(image.shape[:2])
    print(f"Pixel at coordinate ({coord[0]}, {coord[1]}): {image[coord[1], coord[0]]}")

    show_color_space_conversion(image)

    sampling_comparison(image)
    gray_level_quantization(image)
    compare_file_formats(image)


def run_enhancement_flow():
    print_header("Week 4 - Spatial Domain Enhancement")
    image = load_selected_image()
    if image is None:
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    print("\nSelect enhancement technique:")
    print("1. Image Negative")
    print("2. Log Transformation")
    print("3. Gamma / Power-Law Transformation")
    print("4. Contrast Stretching")
    print("5. Histogram and Equalization")
    print("6. Image Arithmetic")
    print("7. Compare enhancement techniques on multiple images")
    print("8. Return to main menu")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        enhanced = apply_image_negative(gray)
        generate_histogram(gray, "Original Histogram", OUTPUT_DIR / "enhancement" / "hist_original_negative.png")
        generate_histogram(enhanced, "Negative Histogram", OUTPUT_DIR / "enhancement" / "hist_negative.png")
        display_image(gray, "Original Gray Image")
        display_image(enhanced, "Image Negative")
        save_image(enhanced, OUTPUT_DIR / "enhancement" / "image_negative.png", "Image Negative")

    elif choice == "2":
        c = float(input("Enter log constant (default 1.0): ") or "1.0")
        enhanced = apply_log_transformation(gray, c=c)
        display_image(gray, "Original Gray Image")
        display_image(enhanced, "Log Transformation")
        save_image(enhanced, OUTPUT_DIR / "enhancement" / "log_transformation.png", "Log Transformation")
        generate_histogram(gray, "Original Histogram", OUTPUT_DIR / "enhancement" / "hist_original_log.png")
        generate_histogram(enhanced, "Log Histogram", OUTPUT_DIR / "enhancement" / "hist_log.png")

    elif choice == "3":
        gamma = float(input("Enter gamma value (default 0.5): ") or "0.5")
        enhanced = apply_gamma_transformation(gray, gamma=gamma)
        display_image(gray, "Original Gray Image")
        display_image(enhanced, f"Gamma Transformation (gamma={gamma})")
        save_image(enhanced, OUTPUT_DIR / "enhancement" / f"gamma_{gamma}.png", f"Gamma Transformation (gamma={gamma})")
        generate_histogram(gray, "Original Histogram", OUTPUT_DIR / "enhancement" / "hist_original_gamma.png")
        generate_histogram(enhanced, "Gamma Histogram", OUTPUT_DIR / "enhancement" / "hist_gamma.png")

    elif choice == "4":
        low_in = int(input("Enter lower input bound (0-255, default 0): ") or "0")
        high_in = int(input("Enter upper input bound (0-255, default 255): ") or "255")
        enhanced = apply_contrast_stretching(gray, low_in=low_in, high_in=high_in)
        display_image(gray, "Original Gray Image")
        display_image(enhanced, "Contrast Stretching")
        save_image(enhanced, OUTPUT_DIR / "enhancement" / "contrast_stretching.png", "Contrast Stretching")

    elif choice == "5":
        eq = equalize_histogram(gray)
        generate_histogram(gray, "Original Histogram", OUTPUT_DIR / "enhancement" / "hist_original_equalized.png")
        generate_histogram(eq, "Equalized Histogram", OUTPUT_DIR / "enhancement" / "hist_equalized.png")
        display_image(gray, "Original Gray Image")
        display_image(eq, "Histogram Equalized Image")
        save_image(eq, OUTPUT_DIR / "enhancement" / "histogram_equalized.png", "Histogram Equalized Image")
        print("Observation: Histogram equalization increases contrast for darker images by spreading intensities.")

    elif choice == "6":
        print("Arithmetic operations use sample images from the dataset.")
        img1 = load_selected_image()
        img2 = load_selected_image()
        img_add = image_addition(img1, img2)
        img_sub = image_subtraction(img1, img2)
        img_avg = image_averaging(img1, img2)
        save_image(img_add, OUTPUT_DIR / "enhancement" / "image_addition.png", "Image Addition")
        save_image(img_sub, OUTPUT_DIR / "enhancement" / "image_subtraction.png", "Image Subtraction")
        save_image(img_avg, OUTPUT_DIR / "enhancement" / "image_averaging.png", "Image Averaging")
        print("Image Addition: useful for combining image information or enhancing signal strength.")
        print("Image Subtraction: useful for change detection and background removal.")
        print("Image Averaging: useful for noise reduction by averaging multiple image frames.")

    elif choice == "7":
        images = list_dataset_images()
        if len(images) < 3:
            print("At least three sample images are required for comparison. A placeholder image will be used if available.")
            for _ in range(3 - len(images)):
                images.append(create_placeholder_sample())
        plot_enhancement_comparison(images[:3])

    elif choice == "8":
        return

    else:
        print("Invalid choice.")


def demo_mode():
    print_header("Project Demo Mode")
    image = create_placeholder_sample()
    loaded_image = load_image(str(image))
    if loaded_image is None:
        print("Could not load the demo image.")
        return

    print("1. Image acquisition and representation")
    details = get_image_details(loaded_image)
    print(details)

    show_color_space_conversion(loaded_image)
    sampling_comparison(loaded_image)
    gray_level_quantization(loaded_image)
    compare_file_formats(loaded_image)

    gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    enhanced_negative = apply_image_negative(gray)
    enhanced_log = apply_log_transformation(gray)
    enhanced_gamma = apply_gamma_transformation(gray, gamma=0.5)
    enhanced_contrast = apply_contrast_stretching(gray, 20, 220)

    save_image(enhanced_negative, OUTPUT_DIR / "enhancement" / "demo_negative.png", "Demo Negative")
    save_image(enhanced_log, OUTPUT_DIR / "enhancement" / "demo_log.png", "Demo Log")
    save_image(enhanced_gamma, OUTPUT_DIR / "enhancement" / "demo_gamma.png", "Demo Gamma")
    save_image(enhanced_contrast, OUTPUT_DIR / "enhancement" / "demo_contrast.png", "Demo Contrast")

    print("Demo completed successfully. Results were saved under outputs/.")


def main():
    ensure_output_folders()
    parser = argparse.ArgumentParser(description="Traffic Sign DIP Project")
    parser.add_argument("--demo", action="store_true", help="Run a quick demonstration of the required DIP workflow")
    args = parser.parse_args()

    if args.demo:
        demo_mode()
        return

    while True:
        print_header("Traffic Sign Detection - DIP Mini Project")
        print("1. Image Acquisition / Representation")
        print("2. Color-space conversion")
        print("3. Sampling")
        print("4. Quantization")
        print("5. File-format comparison")
        print("6. Enhancement techniques")
        print("7. Histogram analysis")
        print("8. Histogram equalization")
        print("9. Image arithmetic")
        print("10. Compare enhancement techniques")
        print("11. Exit")

        choice = input("Select an option: ").strip()

        if choice == "1":
            run_acquisition_flow()
        elif choice == "2":
            image = load_selected_image()
            if image is not None:
                show_color_space_conversion(image)
        elif choice == "3":
            image = load_selected_image()
            if image is not None:
                sampling_comparison(image)
        elif choice == "4":
            image = load_selected_image()
            if image is not None:
                gray_level_quantization(image)
        elif choice == "5":
            image = load_selected_image()
            if image is not None:
                compare_file_formats(image)
        elif choice == "6":
            run_enhancement_flow()
        elif choice == "7":
            image = load_selected_image()
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                generate_histogram(gray, "Histogram", OUTPUT_DIR / "enhancement" / "manual_histogram.png")
                display_image(gray, "Image for histogram")
        elif choice == "8":
            image = load_selected_image()
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                eq = equalize_histogram(gray)
                display_image(gray, "Original Image")
                display_image(eq, "Histogram Equalized Image")
                save_image(eq, OUTPUT_DIR / "enhancement" / "equalized_manual.png", "Equalized Image")
        elif choice == "9":
            image1 = load_selected_image()
            image2 = load_selected_image()
            if image1 is not None and image2 is not None:
                print("Image Addition:", image_addition(image1, image2).shape)
                print("Image Subtraction:", image_subtraction(image1, image2).shape)
                print("Image Averaging:", image_averaging(image1, image2).shape)
        elif choice == "10":
            images = list_dataset_images()
            if len(images) < 3:
                print("Creating placeholder images for comparison.")
                for _ in range(3):
                    images.append(create_placeholder_sample())
            plot_enhancement_comparison(images[:3])
        elif choice == "11":
            print("Exiting the project.")
            break
        else:
            print("Invalid option. Please try again.")

        repeat = input("\nDo you want to continue? (y/n): ").strip().lower()
        if repeat not in ("y", "yes"):
            print("Exiting the project.")
            break


if __name__ == "__main__":
    main()
