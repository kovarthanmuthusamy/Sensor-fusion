# Evaluation results

IoU thresholds 0.5..0.95; a class gets an AP in a cell only with at least 100 labelled objects (ADR 0011). Brackets: 95% interval from 1000 scene-level resamples.

† fewer than 10 scenes: the interval is not reliable.

**Common classes** (eligible in every cell): car, two_wheeler. Only the mAP over these is comparable between cells; a cell's own mAP averages whichever classes it can score (see the headline tables).

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

### mAP@0.5 over the common classes (%), comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 51.2 [47.2, 55.6] | 51.9 [47.9, 56.1] | 0.8 [-1.9, 3.4] |
| day | 52.2 [47.8, 56.9] | 52.3 [47.8, 56.9] | 0.1 [-2.7, 2.9] |
| night | 41.4 [35.3, 48.0] | 48.7 [39.0, 57.7] | 7.2 [2.3, 10.7] |
| boston_day | 46.2 [41.0, 51.6] | 47.0 [41.2, 53.0] | 0.8 [-3.4, 5.5] |
| singapore_day | 55.1 [49.0, 61.8] | 55.4 [49.9, 61.5] | 0.3 [-3.4, 3.7] |
| singapore_night | 41.4 [35.3, 48.0] | 48.7 [39.0, 57.7] | 7.2 [2.3, 10.7] |

### mAP@[0.5:0.95] over the common classes (%), comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 21.9 [20.0, 24.0] | 23.0 [21.2, 25.2] | 1.2 [0.1, 2.3] |
| day | 22.5 [20.5, 24.8] | 23.3 [21.3, 25.5] | 0.8 [-0.4, 1.9] |
| night | 16.4 [13.6, 19.7] | 21.3 [17.3, 25.5] | 4.9 [2.7, 7.0] |
| boston_day | 21.1 [19.0, 23.5] | 21.7 [19.4, 24.2] | 0.5 [-1.4, 2.4] |
| singapore_day | 22.2 [19.4, 25.4] | 23.5 [20.9, 26.9] | 1.3 [0.0, 2.7] |
| singapore_night | 16.4 [13.6, 19.7] | 21.3 [17.3, 25.5] | 4.9 [2.7, 7.0] |

### mAP@0.5 over each cell's own eligible classes (%), NOT comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 47.5 [44.5, 50.6] | 49.5 [46.3, 52.7] | 2.0 [0.4, 3.7] |
| day | 48.1 [45.1, 51.1] | 49.8 [46.7, 52.9] | 1.7 [0.1, 3.5] |
| night | 41.4 [35.3, 48.0] | 48.7 [39.0, 57.7] | 7.2 [2.3, 10.7] |
| boston_day | 42.2 [38.5, 45.8] | 44.4 [40.5, 48.3] | 2.2 [-0.3, 4.9] |
| singapore_day | 54.3 [50.0, 58.7] | 55.6 [51.4, 60.3] | 1.4 [-1.4, 4.0] |
| singapore_night | 41.4 [35.3, 48.0] | 48.7 [39.0, 57.7] | 7.2 [2.3, 10.7] |

### mAP@[0.5:0.95] over each cell's own eligible classes (%), NOT comparable

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 18.4 [17.1, 19.7] | 20.1 [18.8, 21.6] | 1.8 [1.0, 2.5] |
| day | 18.7 [17.5, 20.1] | 20.3 [18.9, 21.7] | 1.6 [0.8, 2.3] |
| night | 16.4 [13.6, 19.7] | 21.3 [17.3, 25.5] | 4.9 [2.7, 7.0] |
| boston_day | 17.2 [15.7, 18.8] | 18.5 [16.7, 20.3] | 1.3 [0.2, 2.3] |
| singapore_day | 20.1 [18.2, 22.2] | 21.9 [19.9, 24.4] | 1.8 [0.5, 3.1] |
| singapore_night | 16.4 [13.6, 19.7] | 21.3 [17.3, 25.5] | 4.9 [2.7, 7.0] |

### median relative distance error (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 8.1 [7.7, 8.5] | 6.8 [6.5, 7.2] | -1.2 [-1.6, -0.9] |
| day | 8.0 [7.7, 8.3] | 6.8 [6.5, 7.2] | -1.2 [-1.5, -0.8] |
| night | 10.9 [9.0, 12.3] | 6.9 [6.1, 7.8] | -4.0 [-5.2, -2.1] |
| boston_day | 7.6 [7.3, 8.0] | 6.8 [6.4, 7.3] | -0.8 [-1.3, -0.4] |
| singapore_day | 8.6 [8.1, 9.4] | 7.0 [6.5, 7.5] | -1.6 [-2.2, -1.1] |
| singapore_night | 10.9 [9.0, 12.3] | 6.9 [6.1, 7.8] | -4.0 [-5.2, -2.1] |

## Per-class AP@0.5 (%)

`-` = fewer than the minimum number of objects.

| cell | model | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|
| all | baseline | 74.5 [71.5, 77.2] | 41.3 [36.5, 46.7] | 46.2 [41.1, 51.5] | 27.8 [21.0, 35.9] |
| all | fusion | 73.6 [70.4, 76.7] | 45.0 [39.9, 50.6] | 49.1 [43.7, 54.7] | 30.3 [23.2, 37.9] |
| day | baseline | 74.7 [71.6, 77.5] | 41.5 [36.6, 46.6] | 46.4 [41.3, 51.6] | 29.8 [22.0, 38.3] |
| day | fusion | 73.4 [69.9, 76.6] | 45.4 [40.0, 50.9] | 49.2 [43.6, 54.8] | 31.3 [23.2, 39.7] |
| night | baseline | 70.5 [60.9, 79.1] | - | - | 12.4 [4.2, 22.1] |
| night | fusion | 73.5 [64.6, 82.4] | - | - | 23.9 [7.5, 39.6] |
| boston_day | baseline | 74.3 [70.3, 77.6] | 36.5 [31.3, 41.7] | 39.8 [31.8, 47.8] | 18.1 [9.6, 27.9] |
| boston_day | fusion | 72.3 [67.6, 76.2] | 41.1 [35.1, 47.4] | 42.5 [33.0, 50.7] | 21.7 [11.6, 31.4] |
| singapore_day | baseline | 76.0 [70.7, 81.1] | 55.1 [45.0, 64.5] | 51.8 [45.7, 57.5] | 34.2 [24.6, 45.8] |
| singapore_day | fusion | 75.9 [70.5, 81.1] | 57.1 [46.6, 67.2] | 54.8 [48.5, 61.2] | 34.9 [26.0, 45.2] |
| singapore_night | baseline | 70.5 [60.9, 79.1] | - | - | 12.4 [4.2, 22.1] |
| singapore_night | fusion | 73.5 [64.6, 82.4] | - | - | 23.9 [7.5, 39.6] |

## Distance error by true distance (median relative error %, matched count)

| cell | model | 0-20 m | 20-40 m | 40-60 m | 60-inf m |
|---|---|---|---|---|---|
| all | baseline | 9.6 (n=4191) | 7.5 (n=7177) | 7.3 (n=3678) | 9.1 (n=1445) |
| all | fusion | 6.6 (n=4284) | 6.4 (n=7451) | 7.4 (n=3798) | 8.3 (n=1628) |
| day | baseline | 9.3 (n=3774) | 7.5 (n=6938) | 7.3 (n=3589) | 9.1 (n=1393) |
| day | fusion | 6.8 (n=3848) | 6.4 (n=7193) | 7.3 (n=3712) | 8.3 (n=1566) |
| night | baseline | 12.1 (n=417) | 10.8 (n=239) | 7.6 (n=89) | 9.8 (n=52) |
| night | fusion | 5.2 (n=436) | 8.8 (n=258) | 9.9 (n=86) | 9.3 (n=62) |
| boston_day | baseline | 9.0 (n=2118) | 7.0 (n=4468) | 7.2 (n=2476) | 8.7 (n=1046) |
| boston_day | fusion | 6.8 (n=2186) | 6.3 (n=4668) | 7.3 (n=2563) | 8.1 (n=1159) |
| singapore_day | baseline | 9.8 (n=1656) | 8.3 (n=2470) | 7.3 (n=1113) | 10.3 (n=347) |
| singapore_day | fusion | 6.9 (n=1662) | 6.6 (n=2525) | 7.4 (n=1149) | 9.1 (n=407) |
| singapore_night | baseline | 12.1 (n=417) | 10.8 (n=239) | 7.6 (n=89) | 9.8 (n=52) |
| singapore_night | fusion | 5.2 (n=436) | 8.8 (n=258) | 9.9 (n=86) | 9.3 (n=62) |
