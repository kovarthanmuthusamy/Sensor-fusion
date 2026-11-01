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
| all | 50.1 [46.3, 54.3] | 51.0 [47.2, 54.8] | 0.9 [-1.6, 2.7] |
| day | 51.6 [47.4, 55.6] | 51.7 [47.5, 56.0] | 0.1 [-2.4, 2.2] |
| night | 36.9 [31.4, 43.2] | 44.0 [38.7, 49.5] | 7.1 [2.5, 11.2] |
| boston_day | 45.7 [40.7, 51.2] | 45.1 [40.8, 49.2] | -0.7 [-4.9, 3.1] |
| singapore_day | 54.3 [49.2, 60.5] | 55.5 [49.7, 61.9] | 1.3 [-1.8, 3.8] |
| singapore_night | 36.9 [31.4, 43.2] | 44.0 [38.7, 49.5] | 7.1 [2.5, 11.2] |

### mAP@[0.5:0.95] over the common classes (%), comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 21.9 [20.1, 23.9] | 23.5 [21.7, 25.4] | 1.6 [0.5, 2.5] |
| day | 22.5 [20.6, 24.6] | 23.9 [21.8, 26.1] | 1.4 [0.3, 2.5] |
| night | 16.7 [13.3, 20.2] | 20.2 [17.1, 23.3] | 3.5 [1.7, 5.1] |
| boston_day | 20.9 [19.0, 23.2] | 21.2 [19.3, 23.1] | 0.3 [-1.5, 2.0] |
| singapore_day | 22.7 [20.2, 25.9] | 25.2 [22.2, 28.9] | 2.5 [1.1, 4.0] |
| singapore_night | 16.7 [13.3, 20.2] | 20.2 [17.1, 23.3] | 3.5 [1.7, 5.1] |

### mAP@0.5 over each cell's own eligible classes (%), NOT comparable across cells

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 46.4 [43.5, 49.5] | 49.0 [45.8, 52.2] | 2.5 [0.8, 3.9] |
| day | 47.3 [44.3, 50.4] | 49.4 [46.2, 52.7] | 2.1 [0.4, 3.6] |
| night | 36.9 [31.4, 43.2] | 44.0 [38.7, 49.5] | 7.1 [2.5, 11.2] |
| boston_day | 41.6 [37.7, 45.5] | 42.7 [38.8, 46.5] | 1.2 [-1.9, 3.7] |
| singapore_day | 53.4 [49.4, 57.5] | 56.2 [51.9, 60.4] | 2.8 [0.7, 4.6] |
| singapore_night | 36.9 [31.4, 43.2] | 44.0 [38.7, 49.5] | 7.1 [2.5, 11.2] |

### mAP@[0.5:0.95] over each cell's own eligible classes (%), NOT comparable

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 18.4 [17.1, 19.8] | 20.8 [19.5, 22.5] | 2.5 [1.7, 3.2] |
| day | 18.7 [17.5, 20.1] | 21.1 [19.6, 22.7] | 2.4 [1.6, 3.2] |
| night | 16.7 [13.3, 20.2] | 20.2 [17.1, 23.3] | 3.5 [1.7, 5.1] |
| boston_day | 17.0 [15.5, 18.8] | 18.4 [16.5, 20.2] | 1.3 [0.0, 2.4] |
| singapore_day | 20.5 [18.8, 22.6] | 23.9 [21.8, 26.4] | 3.4 [2.3, 4.4] |
| singapore_night | 16.7 [13.3, 20.2] | 20.2 [17.1, 23.3] | 3.5 [1.7, 5.1] |

### median relative distance error (%)

| cell | baseline | fusion | difference (2nd - 1st) |
|---|---|---|---|
| all | 7.3 [7.0, 7.6] | 4.2 [4.0, 4.3] | -3.2 [-3.5, -2.8] |
| day | 7.3 [7.0, 7.7] | 4.2 [4.0, 4.4] | -3.1 [-3.4, -2.8] |
| night | 7.6 [7.0, 8.6] | 3.7 [3.1, 4.6] | -3.9 [-4.8, -3.2] |
| boston_day | 7.0 [6.6, 7.5] | 4.2 [4.0, 4.4] | -2.8 [-3.2, -2.5] |
| singapore_day | 7.9 [7.3, 8.5] | 4.3 [4.0, 4.5] | -3.6 [-4.2, -3.0] |
| singapore_night | 7.6 [7.0, 8.6] | 3.7 [3.1, 4.6] | -3.9 [-4.8, -3.2] |

## Per-class AP@0.5 (%)

`-` = fewer than the minimum number of objects.

| cell | model | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|
| all | baseline | 72.8 [69.6, 75.9] | 39.7 [34.8, 45.7] | 45.9 [40.5, 51.1] | 27.4 [21.0, 34.9] |
| all | fusion | 73.3 [69.8, 76.4] | 45.0 [39.5, 51.1] | 48.9 [42.8, 54.9] | 28.6 [22.3, 36.1] |
| day | baseline | 73.3 [69.7, 76.2] | 40.0 [34.4, 45.4] | 46.0 [40.7, 51.3] | 30.0 [23.1, 37.4] |
| day | fusion | 73.1 [69.3, 76.4] | 45.2 [39.7, 50.9] | 49.0 [42.9, 54.8] | 30.3 [23.1, 37.9] |
| night | baseline | 68.0 [56.9, 78.5] | - | - | 5.8 [1.8, 12.4] |
| night | fusion | 73.2 [65.3, 81.1] | - | - | 14.8 [6.8, 23.7] |
| boston_day | baseline | 72.5 [68.4, 76.1] | 35.1 [29.6, 40.5] | 39.9 [31.4, 47.6] | 19.0 [10.5, 28.6] |
| boston_day | fusion | 71.8 [66.7, 75.9] | 41.0 [34.9, 47.4] | 39.8 [29.9, 49.1] | 18.3 [12.2, 24.4] |
| singapore_day | baseline | 74.4 [68.9, 79.9] | 54.1 [42.6, 64.5] | 51.0 [45.0, 56.7] | 34.1 [25.8, 44.8] |
| singapore_day | fusion | 76.0 [71.1, 80.7] | 57.1 [46.9, 66.4] | 56.5 [49.7, 62.7] | 35.0 [25.8, 45.9] |
| singapore_night | baseline | 68.0 [56.9, 78.5] | - | - | 5.8 [1.8, 12.4] |
| singapore_night | fusion | 73.2 [65.3, 81.1] | - | - | 14.8 [6.8, 23.7] |

## Distance error by true distance (median relative error %, matched count)

| cell | model | 0-20 m | 20-40 m | 40-60 m | 60-inf m |
|---|---|---|---|---|---|
| all | baseline | 6.9 (n=4240) | 7.3 (n=7179) | 7.4 (n=3591) | 8.8 (n=1425) |
| all | fusion | 4.2 (n=4346) | 4.0 (n=7442) | 4.2 (n=3679) | 4.8 (n=1566) |
| day | baseline | 6.9 (n=3838) | 7.3 (n=6937) | 7.4 (n=3503) | 8.8 (n=1371) |
| day | fusion | 4.3 (n=3907) | 4.0 (n=7187) | 4.2 (n=3587) | 4.8 (n=1506) |
| night | baseline | 6.5 (n=402) | 8.4 (n=242) | 8.5 (n=88) | 8.9 (n=54) |
| night | fusion | 3.2 (n=439) | 4.3 (n=255) | 3.9 (n=92) | 6.1 (n=60) |
| boston_day | baseline | 6.4 (n=2176) | 6.8 (n=4523) | 7.3 (n=2434) | 8.3 (n=1038) |
| boston_day | fusion | 4.1 (n=2202) | 4.0 (n=4678) | 4.2 (n=2488) | 4.8 (n=1118) |
| singapore_day | baseline | 7.5 (n=1662) | 8.1 (n=2414) | 7.4 (n=1069) | 10.8 (n=333) |
| singapore_day | fusion | 4.5 (n=1705) | 4.1 (n=2509) | 4.2 (n=1099) | 4.7 (n=388) |
| singapore_night | baseline | 6.5 (n=402) | 8.4 (n=242) | 8.5 (n=88) | 8.9 (n=54) |
| singapore_night | fusion | 3.2 (n=439) | 4.3 (n=255) | 3.9 (n=92) | 6.1 (n=60) |
