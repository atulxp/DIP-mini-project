# Task 5 - Traffic-sign segmentation

## Aim
Separate traffic-sign structure from its background using edge detection and thresholding.

## Pipeline input
Each experiment starts with the selected dataset image, applies the existing Task 4 Median 3x3 filtered output, and then converts that result to grayscale. It does not load an unrelated image.

## Best Segmentation Method
**Otsu**

The choice is based on the measured foreground ratio and connected-component fragmentation in the selected images. The score favors a compact foreground near 30% and penalizes fragmentation; it is a transparent heuristic, not a fabricated accuracy value.

## Measured comparison
| Image | Dimensions | Min | Max | Method | Threshold | Foreground ratio | Components | Edge/largest-component pixels |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Global Threshold | 128 | 0.6095 | 45 | 232061 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Otsu | 150 | 0.4920 | 44 | 110385 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Adaptive Threshold | block=31,C=5 | 0.7921 | 92 | 225911 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Roberts |  |  |  | 10053 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Prewitt |  |  |  | 14838 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Sobel |  |  |  | 14726 |
| traffic_sign_001.jpg | 640x640 | 39 | 248 | Laplacian |  |  |  | 6431 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Global Threshold | 128 | 0.5286 | 187 | 150727 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Otsu | 128 | 0.5286 | 187 | 150727 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Adaptive Threshold | block=31,C=5 | 0.7184 | 143 | 202022 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Roberts |  |  |  | 22774 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Prewitt |  |  |  | 27394 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Sobel |  |  |  | 27590 |
| traffic_sign_002.jpg | 640x640 | 15 | 255 | Laplacian |  |  |  | 4837 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Global Threshold | 128 | 0.2564 | 212 | 32025 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Otsu | 116 | 0.3188 | 231 | 63636 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Adaptive Threshold | block=31,C=5 | 0.6655 | 198 | 245546 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Roberts |  |  |  | 14377 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Prewitt |  |  |  | 29801 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Sobel |  |  |  | 29414 |
| traffic_sign_003.jpg | 640x640 | 22 | 254 | Laplacian |  |  |  | 6644 |

## Interpretation
Roberts, Prewitt, Sobel, and Laplacian are saved in each per-image comparison figure. Edge pixel counts describe detected response strength above 25% of the normalized response and are not an accuracy score. Threshold rows report actual foreground ratios and connected components, so broken edges, false edges, missing regions, and extra regions can be discussed from the figures alongside this table.
