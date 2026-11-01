# Evaluation results

IoU thresholds 0.5..0.95; a class gets an AP in a cell only with at least 100 labelled objects (ADR 0011). Brackets: 95% interval from 1000 scene-level resamples.

† fewer than 10 scenes: the interval is not reliable.

## Data behind each cell

| cell | scenes | frames | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|---|
| all | 150 | 6,019 | 15,182 | 5,002 | 6,206 | 1,059 |
| day | 135 | 5,417 | 14,072 | 4,956 | 6,174 | 945 |
| night | 15 | 602 | 1,110 | 46 | 32 | 114 |
| boston_day | 77 | 3,090 | 10,321 | 3,730 | 2,762 | 261 |
| singapore_day | 58 | 2,327 | 3,751 | 1,226 | 3,412 | 684 |
| singapore_night | 15 | 602 | 1,110 | 46 | 32 | 114 |

## Headline

### mAP@0.5 (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 43.3 [40.6, 46.1] | 43.2 [40.2, 46.0] | -0.2 [-1.3, 1.1] |
| day | 44.0 [41.1, 46.8] | 43.7 [40.7, 46.7] | -0.3 [-1.5, 1.1] |
| night | 37.5 [32.5, 42.4] | 39.5 [33.0, 46.3] | 2.0 [-1.8, 7.3] |
| boston_day | 37.5 [34.1, 41.1] | 38.6 [34.8, 42.2] | 1.0 [-0.7, 2.4] |
| singapore_day | 49.8 [46.0, 53.4] | 49.2 [45.0, 53.5] | -0.6 [-2.1, 1.2] |
| singapore_night | 37.5 [32.5, 42.4] | 39.5 [33.0, 46.3] | 2.0 [-1.8, 7.3] |

### mAP@[0.5:0.95] (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 18.8 [17.5, 20.2] | 18.6 [17.3, 19.9] | -0.2 [-0.8, 0.3] |
| day | 19.0 [17.9, 20.4] | 18.9 [17.6, 20.1] | -0.2 [-0.8, 0.3] |
| night | 18.3 [15.6, 21.1] | 18.3 [15.2, 21.5] | -0.1 [-1.4, 1.2] |
| boston_day | 16.8 [15.2, 18.5] | 17.2 [15.6, 18.8] | 0.3 [-0.3, 0.9] |
| singapore_day | 21.3 [19.2, 23.5] | 21.0 [18.9, 23.3] | -0.3 [-1.2, 0.6] |
| singapore_night | 18.3 [15.6, 21.1] | 18.3 [15.2, 21.5] | -0.1 [-1.4, 1.2] |

### median relative distance error (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 6.6 [6.4, 6.9] | 6.7 [6.4, 7.0] | 0.1 [-0.2, 0.3] |
| day | 6.6 [6.3, 7.0] | 6.7 [6.4, 7.0] | 0.1 [-0.1, 0.3] |
| night | 7.3 [6.2, 8.7] | 6.8 [5.5, 8.2] | -0.5 [-1.7, 0.6] |
| boston_day | 6.4 [6.0, 6.8] | 6.6 [6.2, 7.0] | 0.2 [-0.1, 0.5] |
| singapore_day | 7.0 [6.6, 7.6] | 6.8 [6.4, 7.4] | -0.2 [-0.6, 0.2] |
| singapore_night | 7.3 [6.2, 8.7] | 6.8 [5.5, 8.2] | -0.5 [-1.7, 0.6] |

## Per-class AP@0.5 (%)

`-` = fewer than the minimum number of objects.

| cell | model | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|
| all | baseline | 67.7 [64.4, 70.9] | 37.3 [32.1, 42.7] | 42.1 [37.8, 46.8] | 26.3 [19.6, 33.8] |
| all | fusion | 67.4 [64.1, 70.8] | 35.7 [30.9, 40.9] | 41.5 [36.6, 46.4] | 28.2 [21.2, 34.9] |
| day | baseline | 67.4 [63.9, 70.7] | 37.3 [32.2, 42.5] | 42.4 [38.0, 47.1] | 28.9 [21.4, 36.0] |
| day | fusion | 67.0 [63.5, 70.8] | 35.8 [31.1, 40.9] | 41.7 [36.6, 46.6] | 30.2 [23.3, 37.4] |
| night | baseline | 68.5 [59.7, 76.8] | - | - | 6.5 [0.9, 12.8] |
| night | fusion | 67.9 [59.4, 76.3] | - | - | 11.1 [2.5, 24.0] |
| boston_day | baseline | 66.4 [61.5, 70.3] | 33.9 [28.6, 39.7] | 36.2 [29.4, 43.0] | 13.7 [7.1, 22.9] |
| boston_day | fusion | 66.2 [61.3, 70.3] | 32.3 [27.1, 36.9] | 35.9 [27.1, 43.4] | 19.8 [13.1, 27.1] |
| singapore_day | baseline | 70.0 [64.5, 75.3] | 47.1 [37.0, 57.0] | 47.4 [42.0, 53.0] | 34.7 [25.9, 44.5] |
| singapore_day | fusion | 69.8 [63.6, 75.3] | 46.5 [35.4, 57.8] | 46.3 [41.0, 51.4] | 34.3 [25.8, 44.0] |
| singapore_night | baseline | 68.5 [59.7, 76.8] | - | - | 6.5 [0.9, 12.8] |
| singapore_night | fusion | 67.9 [59.4, 76.3] | - | - | 11.1 [2.5, 24.0] |

## Distance error by true distance (median relative error %, matched count)

| cell | model | 0-20 m | 20-40 m | 40-60 m | 60-inf m |
|---|---|---|---|---|---|
| all | baseline | 6.2 (n=4245) | 6.5 (n=6956) | 7.1 (n=3291) | 8.4 (n=1093) |
| all | fusion | 6.4 (n=4276) | 6.6 (n=6940) | 7.1 (n=3242) | 8.4 (n=1146) |
| day | baseline | 6.2 (n=3824) | 6.4 (n=6711) | 7.1 (n=3208) | 8.3 (n=1049) |
| day | fusion | 6.5 (n=3846) | 6.5 (n=6713) | 7.0 (n=3160) | 8.3 (n=1098) |
| night | baseline | 6.0 (n=421) | 9.0 (n=245) | 9.4 (n=83) | 11.4 (n=44) |
| night | fusion | 5.8 (n=430) | 8.1 (n=227) | 8.4 (n=82) | 12.8 (n=48) |
| boston_day | baseline | 5.9 (n=2163) | 6.1 (n=4392) | 7.0 (n=2243) | 8.3 (n=802) |
| boston_day | fusion | 6.2 (n=2196) | 6.4 (n=4403) | 7.0 (n=2198) | 8.0 (n=836) |
| singapore_day | baseline | 6.6 (n=1661) | 6.9 (n=2319) | 7.6 (n=965) | 8.9 (n=247) |
| singapore_day | fusion | 6.8 (n=1650) | 6.6 (n=2310) | 7.1 (n=962) | 9.3 (n=262) |
| singapore_night | baseline | 6.0 (n=421) | 9.0 (n=245) | 9.4 (n=83) | 11.4 (n=44) |
| singapore_night | fusion | 5.8 (n=430) | 8.1 (n=227) | 8.4 (n=82) | 12.8 (n=48) |
