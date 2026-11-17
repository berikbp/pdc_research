# Layer-offload sweep: layer-sweep-7b-v1

2,400 validated requests: 30 configurations, 20 passages, 2 languages, 2 shuffled rounds, 64 generated tokens. Full aligned passages are used, so input length differs by language. Placement is the one the server reported, not the requested `-ngl`.

## Cost of moving blocks to the CPU

Decode time per generated token, least squares over configuration means:

| Term | Estimate, ms | 95% interval |
|---|---:|---:|
| All blocks on GPU | 19.417 | 19.396 to 19.436 |
| Each block moved to CPU | 2.463 | 2.462 to 2.464 |
| First CPU block (fixed handoff) | 1.403 | 1.390 to 1.417 |
| Output layer on CPU | 7.344 | 7.268 to 7.453 |

R² = 0.9997; largest residual 0.768 ms per token. Prompt processing: 0.122 ms per prompt token for each block moved to CPU (R² = 0.9995). Intervals are percentile bootstrap over passages and do not cover variation between sessions.

## Configurations

| GPU layers | Blocks on CPU | Decode, tok/s | Decode, ms/tok | Prefill, ms/tok | Total eng_Latn, s | Total kaz_Cyrl, s | Server VRAM, MiB | CPU cores | GPU util, % | GPU power, W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28 | 10.3 | 97.13 | 3.936 | 6.584 | 7.306 | 174 | 7.3 | 9 | 16.3 |
| 1 | 28 | 11.2 | 89.67 | 3.868 | 6.097 | 6.819 | 612 | 7.3 | 10 | 17.7 |
| 2 | 27 | 11.5 | 87.00 | 3.817 | 5.921 | 6.636 | 758 | 7.2 | 13 | 18.2 |
| 3 | 26 | 11.9 | 84.14 | 3.675 | 5.725 | 6.412 | 904 | 7.2 | 13 | 18.9 |
| 4 | 25 | 12.2 | 81.80 | 3.567 | 5.565 | 6.232 | 1050 | 7.2 | 14 | 19.5 |
| 5 | 24 | 12.5 | 80.01 | 3.400 | 5.438 | 6.067 | 1196 | 7.2 | 15 | 19.9 |
| 6 | 23 | 13.0 | 76.97 | 3.284 | 5.228 | 5.847 | 1342 | 7.1 | 15 | 20.6 |
| 7 | 22 | 13.4 | 74.84 | 3.161 | 5.081 | 5.673 | 1472 | 7.1 | 16 | 21.2 |
| 8 | 21 | 13.7 | 72.80 | 3.065 | 4.937 | 5.520 | 1600 | 7.0 | 19 | 21.2 |
| 9 | 20 | 14.2 | 70.42 | 2.923 | 4.776 | 5.324 | 1746 | 6.9 | 19 | 21.7 |
| 10 | 19 | 14.7 | 68.08 | 2.813 | 4.614 | 5.145 | 1876 | 6.9 | 20 | 22.3 |
| 11 | 18 | 15.2 | 65.92 | 2.731 | 4.468 | 4.983 | 2008 | 6.8 | 19 | 22.9 |
| 12 | 17 | 15.9 | 63.07 | 2.582 | 4.268 | 4.763 | 2156 | 6.7 | 23 | 24.3 |
| 13 | 16 | 16.4 | 60.99 | 2.466 | 4.130 | 4.592 | 2284 | 6.6 | 24 | 25.4 |
| 14 | 15 | 17.2 | 58.14 | 2.350 | 3.937 | 4.377 | 2414 | 6.5 | 25 | 26.8 |
| 15 | 14 | 18.1 | 55.25 | 2.227 | 3.736 | 4.163 | 2560 | 6.4 | 27 | 29.0 |
| 16 | 13 | 18.9 | 52.93 | 2.112 | 3.575 | 3.983 | 2688 | 6.3 | 28 | 30.4 |
| 17 | 12 | 19.8 | 50.50 | 1.990 | 3.408 | 3.794 | 2834 | 6.2 | 32 | 32.0 |
| 18 | 11 | 20.8 | 48.11 | 1.876 | 3.246 | 3.608 | 2964 | 6.1 | 32 | 33.9 |
| 19 | 10 | 22.0 | 45.43 | 1.765 | 3.064 | 3.405 | 3092 | 5.9 | 34 | 36.0 |
| 20 | 9 | 23.2 | 43.04 | 1.617 | 2.896 | 3.214 | 3222 | 5.8 | 37 | 39.9 |
| 21 | 8 | 24.6 | 40.65 | 1.491 | 2.729 | 3.030 | 3354 | 5.6 | 41 | 43.6 |
| 22 | 7 | 26.4 | 37.93 | 1.328 | 2.540 | 2.813 | 3500 | 5.4 | 43 | 48.2 |
| 23 | 6 | 28.1 | 35.54 | 1.214 | 2.376 | 2.628 | 3628 | 5.2 | 47 | 51.7 |
| 24 | 5 | 30.5 | 32.81 | 1.084 | 2.189 | 2.416 | 3774 | 4.9 | 50 | 55.3 |
| 25 | 4 | 32.8 | 30.49 | 0.971 | 2.030 | 2.237 | 3904 | 4.6 | 55 | 60.1 |
| 26 | 3 | 35.6 | 28.09 | 0.861 | 1.864 | 2.056 | 4032 | 4.3 | 62 | 65.4 |
| 27 | 2 | 39.1 | 25.54 | 0.734 | 1.690 | 1.855 | 4178 | 3.8 | 69 | 70.8 |
| 28 | 1 | 43.8 | 22.82 | 0.603 | 1.504 | 1.645 | 4324 | 3.2 | 76 | 79.5 |
| 29 | 0 | 51.5 | 19.42 | 0.464 | 1.273 | 1.388 | 4432 | 1.0 | 93 | 91.9 |

GPU utilization and power are whole-GPU readings sampled every 100 ms; server VRAM is the server process alone. Requests run one at a time with a fixed output budget, so these numbers describe single-request latency, not throughput under load.
