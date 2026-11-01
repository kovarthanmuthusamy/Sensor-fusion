# Service under load

Model `fusion_dense_head`, 15 s per level, one real frame per request (photo + ~35k LiDAR points + calibration), measured 2026-11-01 on the shared lab workstation, client and server on the same machine. Service limits: 2 inferences at once, 4 allowed to wait; beyond that it answers 503 (load shedding). Latency is for answered (200) requests.

| clients | answered | requests/s | p50 ms | p95 ms | p99 ms | shed (503) | errors |
|---|---|---|---|---|---|---|---|
| 1 | 69 | 4.6 | 216 | 228 | 271 | 0 | 0 |
| 2 | 96 | 6.3 | 306 | 331 | 336 | 0 | 0 |
| 4 | 152 | 10.0 | 382 | 392 | 598 | 0 | 0 |
| 8 | 136 | 8.8 | 693 | 746 | 1014 | 134 | 0 |
| 16 | 128 | 8.3 | 1043 | 1143 | 1173 | 167 | 0 |
