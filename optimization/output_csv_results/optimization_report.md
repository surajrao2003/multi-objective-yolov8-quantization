# YOLOv8 configuration optimization report
## Inputs and settings
- **Benchmark CSV:** `optimization/configurations.csv`
- **Utility weights:** α=0.5, β=0.3, γ=0.2
- **Constraints:** map50 ≥ 0.65, latency_ms ≤ 100.0, size_mb ≤ 50.0
- **Feasible configurations:** 51 / 144

## Best configuration summary
| selection_type   | group        | model   | precision    |   resolution | hardware   |   fps |   latency_ms |   size_mb |   map50 |   utility | feasible   |
|------------------|--------------|---------|--------------|--------------|------------|-------|--------------|-----------|---------|-----------|------------|
| best_overall     |              | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |
| best_hardware    | CPU1         | YOLOv8n | FP32         |          640 | CPU1       | 23.97 |        41.72 |     12.26 |  0.6814 |    0.4287 | True       |
| best_hardware    | CPU2         | YOLOv8n | INT8 Dynamic |          640 | CPU2       | 28.03 |        35.68 |     12.26 |  0.6799 |    0.4424 | True       |
| best_hardware    | GPU1         | YOLOv8n | FP32         |          640 | GPU1       | 39.47 |        25.34 |     12.26 |  0.6799 |    0.4891 | True       |
| best_hardware    | GPU2         | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |
| best_precision   | FP16         | YOLOv8s | FP16         |         1280 | GPU2       | 12.46 |        80.28 |     21.65 |  0.761  |    0.5096 | True       |
| best_precision   | FP32         | YOLOv8n | FP32         |          640 | GPU2       | 43.75 |        22.86 |     12.26 |  0.6799 |    0.5066 | True       |
| best_precision   | INT8 Dynamic | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |
| best_precision   | INT8 Static  | YOLOv8s | INT8 Static  |          640 | GPU2       | 37.09 |        26.96 |     11.14 |  0.684  |    0.4895 | True       |
## Top feasible configurations (by utility)
|   rank | model   | precision    |   resolution | hardware   |   fps |   latency_ms |   size_mb |   map50 |   utility | feasible   |   constraint_violation |
|--------|---------|--------------|--------------|------------|-------|--------------|-----------|---------|-----------|------------|------------------------|
|      1 | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |                      0 |
|      2 | YOLOv8s | FP16         |         1280 | GPU2       | 12.46 |        80.28 |     21.65 |  0.761  |    0.5096 | True       |                      0 |
|      3 | YOLOv8n | FP32         |          640 | GPU2       | 43.75 |        22.86 |     12.26 |  0.6799 |    0.5066 | True       |                      0 |
|      4 | YOLOv8s | INT8 Static  |          640 | GPU2       | 37.09 |        26.96 |     11.14 |  0.684  |    0.4895 | True       |                      0 |
|      5 | YOLOv8n | FP32         |          640 | GPU1       | 39.47 |        25.34 |     12.26 |  0.6799 |    0.4891 | True       |                      0 |
|      6 | YOLOv8s | INT8 Static  |         1280 | GPU2       | 14.33 |        69.77 |     11.62 |  0.7337 |    0.4875 | True       |                      0 |
|      7 | YOLOv8s | FP16         |          640 | GPU2       | 30.16 |        33.15 |     21.41 |  0.7092 |    0.4863 | True       |                      0 |
|      8 | YOLOv8n | INT8 Dynamic |          640 | GPU1       | 38.79 |        25.78 |     12.26 |  0.6799 |    0.4863 | True       |                      0 |
|      9 | YOLOv8n | INT8 Dynamic |         1280 | GPU2       | 18.1  |        55.25 |     12.74 |  0.7258 |    0.4859 | True       |                      0 |
|     10 | YOLOv8n | FP32         |         1280 | GPU2       | 17.97 |        55.64 |     12.74 |  0.7258 |    0.4854 | True       |                      0 |
|     11 | YOLOv8s | INT8 Dynamic |          640 | GPU2       | 41.37 |        24.17 |     42.79 |  0.7079 |    0.4853 | True       |                      0 |
|     12 | YOLOv8s | FP32         |          640 | GPU2       | 40.69 |        24.58 |     42.79 |  0.7079 |    0.4825 | True       |                      0 |
|     13 | YOLOv8s | INT8 Dynamic |         1280 | GPU2       | 16.52 |        60.52 |     43.27 |  0.7608 |    0.4809 | True       |                      0 |
|     14 | YOLOv8n | FP16         |          640 | GPU2       | 34.22 |        29.23 |      6.14 |  0.6798 |    0.4803 | True       |                      0 |
|     15 | YOLOv8n | FP16         |         1280 | GPU2       | 13.26 |        75.41 |      6.38 |  0.7256 |    0.4791 | True       |                      0 |
## Pareto frontier (max map50 & fps, min size_mb)
| model   | precision    |   resolution | hardware   |   fps |   latency_ms |   size_mb |   map50 |   utility | feasible   |   constraint_violation |
|---------|--------------|--------------|------------|-------|--------------|-----------|---------|-----------|------------|------------------------|
| YOLOv8s | FP16         |         1280 | GPU1       |  7.89 |       126.79 |     21.65 |  0.7612 |    0.4912 | False      |                 0.2679 |
| YOLOv8s | FP16         |         1280 | GPU2       | 12.46 |        80.28 |     21.65 |  0.761  |    0.5096 | True       |                 0      |
| YOLOv8s | INT8 Dynamic |         1280 | GPU2       | 16.52 |        60.52 |     43.27 |  0.7608 |    0.4809 | True       |                 0      |
| YOLOv8s | INT8 Static  |         1280 | CPU2       |  3.38 |       295.99 |     11.62 |  0.7346 |    0.4443 | False      |                 1.9599 |
| YOLOv8s | INT8 Static  |         1280 | GPU2       | 14.33 |        69.77 |     11.62 |  0.7337 |    0.4875 | True       |                 0      |
| YOLOv8n | INT8 Dynamic |         1280 | GPU2       | 18.1  |        55.25 |     12.74 |  0.7258 |    0.4859 | True       |                 0      |
| YOLOv8n | FP16         |         1280 | CPU2       |  6.39 |       156.41 |      6.38 |  0.7257 |    0.4511 | False      |                 0.5641 |
| YOLOv8n | FP16         |         1280 | GPU1       | 10.51 |        95.12 |      6.38 |  0.7257 |    0.4679 | True       |                 0      |
| YOLOv8n | FP16         |         1280 | GPU2       | 13.26 |        75.41 |      6.38 |  0.7256 |    0.4791 | True       |                 0      |
| YOLOv8m | FP16         |          640 | GPU2       | 30.45 |        32.84 |     49.51 |  0.7236 |    0.4557 | True       |                 0      |
| YOLOv8m | INT8 Dynamic |          640 | GPU2       | 36.84 |        27.14 |     99    |  0.7236 |    0.3789 | False      |                 0.98   |
| YOLOv8n | INT8 Static  |         1280 | GPU2       | 16.65 |        60.07 |      3.91 |  0.7101 |    0.4693 | True       |                 0      |
| YOLOv8s | FP16         |          640 | GPU2       | 30.16 |        33.15 |     21.41 |  0.7092 |    0.4863 | True       |                 0      |
| YOLOv8m | INT8 Static  |          640 | GPU2       | 30.26 |        33.04 |     25.34 |  0.7082 |    0.4769 | True       |                 0      |
| YOLOv8s | INT8 Dynamic |          640 | GPU2       | 41.37 |        24.17 |     42.79 |  0.7079 |    0.4853 | True       |                 0      |
| YOLOv8s | INT8 Static  |          640 | GPU1       | 22.36 |        44.72 |     11.14 |  0.6849 |    0.4308 | True       |                 0      |
| YOLOv8s | INT8 Static  |          640 | GPU2       | 37.09 |        26.96 |     11.14 |  0.684  |    0.4895 | True       |                 0      |
| YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |                 0      |
| YOLOv8n | FP16         |          640 | GPU2       | 34.22 |        29.23 |      6.14 |  0.6798 |    0.4803 | True       |                 0      |
| YOLOv8m | INT8 Dynamic |          320 | GPU2       | 61    |        16.39 |     98.88 |  0.6609 |    0.3617 | False      |                 0.9776 |
| YOLOv8m | FP16         |          320 | GPU2       | 50.39 |        19.84 |     49.45 |  0.6608 |    0.4209 | True       |                 0      |
| YOLOv8n | INT8 Static  |          640 | GPU2       | 39.4  |        25.38 |      3.43 |  0.6588 |    0.4681 | True       |                 0      |
| YOLOv8s | FP16         |          320 | GPU2       | 57.63 |        17.35 |     21.35 |  0.6143 |    0.423  | False      |                 0.0357 |
| YOLOv8s | INT8 Dynamic |          320 | GPU2       | 67.56 |        14.8  |     42.67 |  0.6142 |    0.4189 | False      |                 0.0358 |
| YOLOv8s | INT8 Static  |          320 | GPU1       | 42.95 |        23.28 |     11.02 |  0.5647 |    0.2925 | False      |                 0.0853 |
| YOLOv8s | INT8 Static  |          320 | GPU2       | 51.92 |        19.26 |     11.02 |  0.5645 |    0.3288 | False      |                 0.0855 |
| YOLOv8n | INT8 Dynamic |          320 | GPU1       | 68.79 |        14.54 |     12.14 |  0.5215 |    0.3156 | False      |                 0.1285 |
| YOLOv8n | FP16         |          320 | CPU1       | 61.8  |        16.18 |      6.08 |  0.5206 |    0.2981 | False      |                 0.1294 |
| YOLOv8n | FP16         |          320 | GPU2       | 63.88 |        15.66 |      6.08 |  0.5201 |    0.3057 | False      |                 0.1299 |
| YOLOv8n | INT8 Dynamic |          320 | GPU2       | 74.17 |        13.48 |     12.14 |  0.5188 |    0.3327 | False      |                 0.1312 |
| YOLOv8n | INT8 Static  |          320 | GPU2       | 57.11 |        17.51 |      3.31 |  0.5052 |    0.2561 | False      |                 0.1448 |
## Best feasible per hardware
|   rank | model   | precision    |   resolution | hardware   |   fps |   latency_ms |   size_mb |   map50 |   utility | feasible   |   constraint_violation |
|--------|---------|--------------|--------------|------------|-------|--------------|-----------|---------|-----------|------------|------------------------|
|     33 | YOLOv8n | FP32         |          640 | CPU1       | 23.97 |        41.72 |     12.26 |  0.6814 |    0.4287 | True       |                      0 |
|     27 | YOLOv8n | INT8 Dynamic |          640 | CPU2       | 28.03 |        35.68 |     12.26 |  0.6799 |    0.4424 | True       |                      0 |
|      5 | YOLOv8n | FP32         |          640 | GPU1       | 39.47 |        25.34 |     12.26 |  0.6799 |    0.4891 | True       |                      0 |
|      1 | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |                      0 |
## Best feasible per precision
|   rank | model   | precision    |   resolution | hardware   |   fps |   latency_ms |   size_mb |   map50 |   utility | feasible   |   constraint_violation |
|--------|---------|--------------|--------------|------------|-------|--------------|-----------|---------|-----------|------------|------------------------|
|      2 | YOLOv8s | FP16         |         1280 | GPU2       | 12.46 |        80.28 |     21.65 |  0.761  |    0.5096 | True       |                      0 |
|      3 | YOLOv8n | FP32         |          640 | GPU2       | 43.75 |        22.86 |     12.26 |  0.6799 |    0.5066 | True       |                      0 |
|      1 | YOLOv8n | INT8 Dynamic |          640 | GPU2       | 45.91 |        21.78 |     12.26 |  0.6799 |    0.5154 | True       |                      0 |
|      4 | YOLOv8s | INT8 Static  |          640 | GPU2       | 37.09 |        26.96 |     11.14 |  0.684  |    0.4895 | True       |                      0 |
