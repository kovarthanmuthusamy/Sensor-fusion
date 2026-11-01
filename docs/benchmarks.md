# Latency

Measured 2026-11-01 on a shared lab workstation, batch size 1, 30 timed runs after 5 warm-up runs, one real validation frame (1600x900 photo, 34,720 LiDAR points). **These are workstation numbers, not edge-device numbers.**

- CPU: Intel(R) Xeon(R) w9-3475X (Windows); PyTorch rows use PyTorch's default of 36 threads, service rows the service's 4; ONNX Runtime 1.31.1 manages its own threads
- GPU: Quadro GV100 (PyTorch 2.14.1+cu126)

| model | setup | median ms | p95 ms | mean ms |
|---|---|---|---|---|
| rgb | PyTorch CPU (fp32), network only | 37.0 | 44.3 | 37.9 |
| rgb | ONNX Runtime CPU, network only | 14.0 | 22.9 | 15.3 |
| rgb | PyTorch GPU (fp16), network only | 4.5 | 6.6 | 4.8 |
| rgb | Service call on CPU (prepare + ONNX + decode) | 58.7 | 66.6 | 59.6 |
| fusion_dense_head | PyTorch CPU (fp32), network only | 41.5 | 50.5 | 43.0 |
| fusion_dense_head | ONNX Runtime CPU, network only | 18.7 | 22.8 | 19.0 |
| fusion_dense_head | PyTorch GPU (fp16), network only | 12.8 | 14.7 | 13.2 |
| fusion_dense_head | Service call on CPU (prepare + ONNX + decode) | 112.1 | 128.6 | 114.3 |

## Where a service call spends its time (median ms, CPU)

| model | prepare (projection, thickening, normalising) | network (ONNX) | decode |
|---|---|---|---|
| rgb | 40.2 | 14.7 | 3.2 |
| fusion_dense_head | 89.3 | 20.1 | 3.8 |
