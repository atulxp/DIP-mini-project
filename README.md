# Traffic Sign Detection Using Digital Image Processing Techniques

This project is a semester-long Digital Image Processing mini project focused on a complete traffic-sign processing pipeline:

- Image acquisition and fundamentals
- Image representation
- Color-space conversion
- Sampling and quantization
- File format comparison
- Spatial-domain enhancement
- Histogram analysis and equalization
- Image arithmetic operations
- Edge detection and threshold-based segmentation
- Region growing with 4- and 8-connected neighbourhoods
- Recursive region splitting and adjacent-region merging
- Binary erosion, dilation, opening, closing, and structuring-element comparison

This is not yet the final traffic-sign recognition system. It is the required foundation for later DIP stages, which will be added incrementally over the semester.

## Current implemented modules

The current implementation includes the following required work:

1. Image acquisition from a configurable dataset path
2. Image representation and coordinate inspection
3. Conversion between RGB, grayscale, and HSV
4. Sampling at 100%, 50%, and 25%
5. Gray-level quantization at 256, 128, 64, and 32 levels
6. BMP, PNG, and JPEG comparison
7. Spatial-domain enhancement techniques:
   - Image negative
   - Log transformation
   - Gamma / power-law transformation
   - Contrast stretching
8. Histogram generation and histogram equalization
9. Image arithmetic: addition, subtraction, and averaging
10. Noise, smoothing, sharpening, and filtering comparison
11. Roberts, Prewitt, Sobel, and Laplacian edge detection
12. Global, Otsu, and adaptive thresholding
13. Region growing threshold and connectivity experiments
14. Region splitting, merging, and integrated evidence reports
15. Binary morphological processing from the existing segmentation input
16. Menu-based execution for the complete project workflow

## Dataset location

The current project uses the real traffic-sign dataset stored in:

- dataset/traffic_signs/

This directory is intended to contain the selected traffic-sign images used for the current DIP project. The application accepts any valid image file in common formats such as JPG, PNG, BMP, and TIFF, and it does not rely on a fixed class-folder or filename structure.

The legacy placeholder/test sample is ignored during dataset discovery so the actual traffic-sign set is used by default.

## Installation

1. Open a terminal in the project root.
2. Create a virtual environment (optional but recommended):

   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:

   python -m pip install -r requirements.txt

## How to run

From the project root:

python main.py

The program presents a menu for the required DIP tasks. You can also run a quick demonstration mode:

python main.py --demo

## Current functionality

The menu includes the following workflow stages:

- Image acquisition and image information
- Color-space conversion
- Sampling
- Quantization
- File-format comparison
- Enhancement techniques
- Histogram analysis
- Histogram equalization
- Image arithmetic
- Enhancement comparison
- Task 5: segmentation experiments and comparison metrics
- Task 6: region-growing experiments and connectivity comparison
- Task 7: region splitting and merging with measured region counts
- Task 8: binary morphological processing from the Task 4-filtered segmentation input

Task 5 reuses the existing Task 4 Median 3x3 filtered output before grayscale conversion. Task 8 continues from that same filtered image, creates a measured Otsu/global/adaptive binary mask, and applies real neighbourhood-based morphology. The generated evidence is stored under `outputs/segmentation`, `outputs/region_growing`, `outputs/region_splitting_merging`, and `outputs/morphology`. CSV files contain measured values from the selected traffic-sign images; Markdown summaries explain the selection criteria without fabricating accuracy values.

## Demonstration workflow

Run `python main.py`, choose option 12, and run all Task 5 experiments. Then choose option 13 for Task 6, option 14 for Task 7, and option 15 for Task 8. Select the same traffic-sign image, binary conversion, structuring element, and operation. Task 8 saves the binary input, every supported element's erosion/dilation outputs, opening/closing overview, measured CSV, and Markdown report under `outputs/morphology`.

## Folder structure

- src/ - DIP processing modules
- dataset/ - user-supplied traffic-sign images
- outputs/ - generated images and comparison results
- report/ - report support materials and screenshots

## Notes

This project is intentionally scoped to the current assignment and will be extended incrementally in future weeks without adding unrelated functionality.
