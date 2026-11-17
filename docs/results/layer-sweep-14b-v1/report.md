# Layer-offload sweep: layer-sweep-14b-v1

560 validated requests: 7 configurations, 20 passages, 2 languages, 2 shuffled rounds, 64 generated tokens. Full aligned passages are used, so input length differs by language. Placement is the one the server reported, not the requested `-ngl`.

## Cost of moving blocks to the CPU

Decode time per generated token, least squares over configuration means:

| Term | Estimate, ms | 95% interval |
|---|---:|---:|
| Intercept (extrapolated, includes handoff) | 40.175 | 40.100 to 40.246 |
| Each block moved to CPU | 2.908 | 2.906 to 2.910 |
| Output layer on CPU | 11.544 | 11.269 to 11.840 |

R² = 0.9999; largest residual 0.781 ms per token. Prompt processing: 0.148 ms per prompt token for each block moved to CPU (R² = 0.9999). Intervals are percentile bootstrap over passages and do not cover variation between sessions. No configuration holds every block on the GPU, so the intercept is extrapolated, not a measured or feasible placement. Left out of the fit because extra server options change the regime: `g00-t8-nogpu`.

## Configurations

| GPU layers | Blocks on CPU | Decode, tok/s | Decode, ms/tok | Prefill, ms/tok | Total eng_Latn, s | Total kaz_Cyrl, s | Server VRAM, MiB | CPU cores | GPU util, % | GPU power, W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 `--device none` | 48 | 5.1 | 195.24 | 32.319 | 15.452 | 22.121 | 94 | 8.0 | 0 | 2.8 |
| 0 | 48 | 5.2 | 191.31 | 8.151 | 12.978 | 14.550 | 184 | 7.3 | 9 | 15.4 |
| 1 | 48 | 5.6 | 179.01 | 8.046 | 12.179 | 13.744 | 804 | 7.2 | 11 | 16.3 |
| 13 | 36 | 6.9 | 145.65 | 6.338 | 9.879 | 11.134 | 2826 | 6.8 | 17 | 19.1 |
| 25 | 24 | 9.0 | 110.52 | 4.554 | 7.468 | 8.383 | 4772 | 6.2 | 25 | 25.8 |
| 37 | 12 | 13.3 | 74.91 | 2.753 | 5.017 | 5.605 | 6718 | 5.1 | 42 | 47.0 |
| 41 | 8 | 15.9 | 63.02 | 2.164 | 4.200 | 4.682 | 7378 | 4.4 | 49 | 55.1 |

GPU utilization and power are whole-GPU readings sampled every 100 ms; server VRAM is the server process alone. Requests run one at a time with a fixed output budget, so these numbers describe single-request latency, not throughput under load.
