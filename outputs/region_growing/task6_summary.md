# Task 6 - Region growing

Image: `traffic_sign_001.jpg`
Dimensions: `640x640`; grayscale min/max: `39/248`
Seed strategy: automatic centre seed `(320, 320)`
Criterion: a candidate pixel is accepted when its intensity differs from the current region mean by no more than the selected threshold.

## Measured comparison
| Connectivity | Threshold | Pixels | Processing time (ms) | Final mean |
|---:|---:|---:|---:|---:|
| 4 | 10 | 4277 | 2.749 | 211.85 |

Best bounded configuration: **4-connected, threshold 10**. It is the largest measured region not exceeding 10% of the image area, a leakage guard for this traffic-sign experiment. The figure and CSV retain every threshold/connectivity result.
