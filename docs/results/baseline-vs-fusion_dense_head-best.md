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
| all | 51.2 [47.2, 55.6] | 51.0 [47.0, 55.3] | -0.2 [-2.4, 1.9] |
| day | 52.2 [47.8, 56.9] | 51.9 [47.3, 56.4] | -0.3 [-2.7, 1.9] |
| night | 41.4 [35.3, 48.0] | 44.0 [39.3, 49.5] | 2.6 [-2.2, 8.0] |
| boston_day | 46.2 [41.0, 51.6] | 45.6 [41.6, 49.2] | -0.7 [-5.3, 3.6] |
| singapore_day | 55.1 [49.0, 61.8] | 55.6 [49.6, 62.3] | 0.5 [-2.4, 3.0] |
| singapore_night | 41.4 [35.3, 48.0] | 44.0 [39.3, 49.5] | 2.6 [-2.2, 8.0] |

### mAP@[0.5:0.95] over the common classes (%), comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 21.9 [20.0, 24.0] | 23.4 [21.6, 25.5] | 1.5 [0.4, 2.6] |
| day | 22.5 [20.5, 24.8] | 23.8 [21.7, 26.1] | 1.4 [0.2, 2.6] |
| night | 16.4 [13.6, 19.7] | 19.9 [17.0, 23.1] | 3.5 [1.8, 5.3] |
| boston_day | 21.1 [19.0, 23.5] | 21.4 [19.6, 23.1] | 0.2 [-1.9, 2.2] |
| singapore_day | 22.2 [19.4, 25.4] | 24.8 [21.9, 28.5] | 2.6 [1.1, 4.2] |
| singapore_night | 16.4 [13.6, 19.7] | 19.9 [17.0, 23.1] | 3.5 [1.8, 5.3] |

### mAP@0.5 over each cell's own eligible classes (%), NOT comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 47.5 [44.5, 50.6] | 49.3 [46.1, 52.6] | 1.9 [0.3, 3.3] |
| day | 48.1 [45.1, 51.1] | 49.9 [46.6, 53.1] | 1.8 [0.1, 3.3] |
| night | 41.4 [35.3, 48.0] | 44.0 [39.3, 49.5] | 2.6 [-2.2, 8.0] |
| boston_day | 42.2 [38.5, 45.8] | 43.5 [39.5, 47.3] | 1.3 [-1.6, 4.0] |
| singapore_day | 54.3 [50.0, 58.7] | 56.3 [51.9, 60.8] | 2.0 [0.0, 3.7] |
| singapore_night | 41.4 [35.3, 48.0] | 44.0 [39.3, 49.5] | 2.6 [-2.2, 8.0] |

### mAP@[0.5:0.95] over each cell's own eligible classes (%), NOT comparable

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 18.4 [17.1, 19.7] | 20.9 [19.4, 22.4] | 2.5 [1.6, 3.3] |
| day | 18.7 [17.5, 20.1] | 21.1 [19.6, 22.7] | 2.4 [1.6, 3.3] |
| night | 16.4 [13.6, 19.7] | 19.9 [17.0, 23.1] | 3.5 [1.8, 5.3] |
| boston_day | 17.2 [15.7, 18.8] | 18.5 [16.8, 20.2] | 1.4 [0.0, 2.6] |
| singapore_day | 20.1 [18.2, 22.2] | 23.7 [21.5, 26.2] | 3.5 [2.2, 5.0] |
| singapore_night | 16.4 [13.6, 19.7] | 19.9 [17.0, 23.1] | 3.5 [1.8, 5.3] |

### median relative distance error (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 8.1 [7.7, 8.5] | 4.3 [4.1, 4.5] | -3.8 [-4.2, -3.4] |
| day | 8.0 [7.7, 8.3] | 4.3 [4.1, 4.5] | -3.7 [-4.0, -3.3] |
| night | 10.9 [9.0, 12.3] | 3.6 [3.1, 4.5] | -7.3 [-8.3, -5.4] |
| boston_day | 7.6 [7.3, 8.0] | 4.3 [4.1, 4.5] | -3.3 [-3.8, -3.0] |
| singapore_day | 8.6 [8.1, 9.4] | 4.4 [4.1, 4.7] | -4.2 [-5.0, -3.6] |
| singapore_night | 10.9 [9.0, 12.3] | 3.6 [3.1, 4.5] | -7.3 [-8.3, -5.4] |

## Per-class AP@0.5 (%)

`-` = fewer than the minimum number of objects.

| cell | model | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|
| all | baseline | 74.5 [71.5, 77.2] | 41.3 [36.5, 46.7] | 46.2 [41.1, 51.5] | 27.8 [21.0, 35.9] |
| all | fusion | 73.7 [70.7, 76.9] | 45.4 [39.9, 51.3] | 49.9 [44.0, 55.7] | 28.3 [21.3, 36.7] |
| day | baseline | 74.7 [71.6, 77.5] | 41.5 [36.6, 46.6] | 46.4 [41.3, 51.6] | 29.8 [22.0, 38.3] |
| day | fusion | 73.9 [70.1, 76.8] | 45.6 [40.0, 51.2] | 50.1 [44.1, 55.9] | 30.0 [22.0, 38.1] |
| night | baseline | 70.5 [60.9, 79.1] | - | - | 12.4 [4.2, 22.1] |
| night | fusion | 72.7 [65.0, 80.2] | - | - | 15.3 [8.3, 24.6] |
| boston_day | baseline | 74.3 [70.3, 77.6] | 36.5 [31.3, 41.7] | 39.8 [31.8, 47.8] | 18.1 [9.6, 27.9] |
| boston_day | fusion | 72.8 [68.0, 76.5] | 41.6 [35.3, 47.9] | 41.4 [31.6, 50.3] | 18.4 [12.5, 24.3] |
| singapore_day | baseline | 76.0 [70.7, 81.1] | 55.1 [45.0, 64.5] | 51.8 [45.7, 57.5] | 34.2 [24.6, 45.8] |
| singapore_day | fusion | 76.4 [71.4, 81.1] | 56.8 [46.8, 66.3] | 57.1 [50.3, 63.1] | 34.8 [25.0, 46.4] |
| singapore_night | baseline | 70.5 [60.9, 79.1] | - | - | 12.4 [4.2, 22.1] |
| singapore_night | fusion | 72.7 [65.0, 80.2] | - | - | 15.3 [8.3, 24.6] |

## Distance error by true distance (median relative error %, matched count)

| cell | model | 0-20 m | 20-40 m | 40-60 m | 60-inf m |
|---|---|---|---|---|---|
| all | baseline | 9.6 (n=4191) | 7.5 (n=7177) | 7.3 (n=3678) | 9.1 (n=1445) |
| all | fusion | 4.4 (n=4344) | 4.1 (n=7409) | 4.2 (n=3723) | 5.0 (n=1609) |
| day | baseline | 9.3 (n=3774) | 7.5 (n=6938) | 7.3 (n=3589) | 9.1 (n=1393) |
| day | fusion | 4.5 (n=3906) | 4.1 (n=7146) | 4.2 (n=3631) | 4.9 (n=1538) |
| night | baseline | 12.1 (n=417) | 10.8 (n=239) | 7.6 (n=89) | 9.8 (n=52) |
| night | fusion | 3.2 (n=438) | 4.0 (n=263) | 3.8 (n=92) | 6.6 (n=71) |
| boston_day | baseline | 9.0 (n=2118) | 7.0 (n=4468) | 7.2 (n=2476) | 8.7 (n=1046) |
| boston_day | fusion | 4.3 (n=2194) | 4.1 (n=4662) | 4.3 (n=2514) | 5.0 (n=1153) |
| singapore_day | baseline | 9.8 (n=1656) | 8.3 (n=2470) | 7.3 (n=1113) | 10.3 (n=347) |
| singapore_day | fusion | 4.7 (n=1712) | 4.2 (n=2484) | 4.1 (n=1117) | 4.9 (n=385) |
| singapore_night | baseline | 12.1 (n=417) | 10.8 (n=239) | 7.6 (n=89) | 9.8 (n=52) |
| singapore_night | fusion | 3.2 (n=438) | 4.0 (n=263) | 3.8 (n=92) | 6.6 (n=71) |
