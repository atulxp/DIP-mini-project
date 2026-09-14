# Task 6 - Region growing

Image: `traffic_sign_001.jpg`
Dimensions: `640x640`; grayscale min/max: `39/248`
Seed strategy: automatic centre seed `(320, 320)`
Criterion: a candidate pixel is accepted when its intensity differs from the current region mean by no more than the selected threshold.

## Measured comparison
| Connectivity | Threshold | Pixels | Processing time (ms) | Final mean |
|---:|---:|---:|---:|---:|
| 4 | 10 | 4277 | 3.183 | 211.85 |
| 4 | 25 | 31570 | 19.080 | 209.91 |
| 4 | 45 | 44498 | 27.349 | 198.50 |
| 8 | 10 | 4232 | 4.067 | 212.15 |
| 8 | 25 | 31556 | 26.861 | 209.92 |
| 8 | 45 | 44398 | 38.699 | 198.61 |

Best bounded configuration: **4-connected, threshold 25**. It is the largest measured region not exceeding 10% of the image area, a leakage guard for this traffic-sign experiment. The figure and CSV retain every threshold/connectivity result.
