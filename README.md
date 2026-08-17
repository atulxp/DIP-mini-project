# Traffic Sign Detection Using Digital Image Processing Techniques

This project is a semester-long Digital Image Processing mini project focused on the initial stages of traffic sign processing:

- Image acquisition and fundamentals
- Image representation
- Color-space conversion
- Sampling and quantization
- File format comparison
- Spatial-domain enhancement
- Histogram analysis and equalization
- Image arithmetic operations

This is not yet the final traffic-sign recognition system. It is the required foundation for later DIP stages, which will be added incrementally over the semester.

## Current implemented modules

The current implementation includes only the following required work:

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
10. Simple menu-based execution for the required project workflow

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

The menu includes only the required steps for the current semester scope:

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

Later DIP stages such as filtering, segmentation, and classification are intentionally not included in this version.

## Folder structure

- src/ - DIP processing modules
- dataset/ - user-supplied traffic-sign images
- outputs/ - generated images and comparison results
- report/ - report support materials and screenshots

## Notes

This project is intentionally scoped to the current assignment and will be extended incrementally in future weeks without adding unrelated functionality.
