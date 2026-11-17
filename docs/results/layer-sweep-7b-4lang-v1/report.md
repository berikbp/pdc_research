# Layer-offload sweep: layer-sweep-7b-4lang-v1

1,280 validated requests: 8 configurations, 20 passages, 4 languages, 2 shuffled rounds, 64 generated tokens. Full aligned passages are used, so input length differs by language. Placement is the one the server reported, not the requested `-ngl`.

## Cost of moving blocks to the CPU

Decode time per generated token, least squares over configuration means:

| Term | Estimate, ms | 95% interval |
|---|---:|---:|
| All blocks on GPU | 19.272 | 19.257 to 19.284 |
| Each block moved to CPU | 2.433 | 2.432 to 2.435 |
| First CPU block (fixed handoff) | 1.409 | 1.394 to 1.425 |
| Output layer on CPU | 7.140 | 7.097 to 7.185 |

R² = 0.9996; largest residual 1.001 ms per token. Prompt processing: 0.133 ms per prompt token for each block moved to CPU (R² = 0.9997). Intervals are percentile bootstrap over passages and do not cover variation between sessions.

## Configurations

| GPU layers | Blocks on CPU | Decode, tok/s | Decode, ms/tok | Prefill, ms/tok | Total eng_Latn, s | Total rus_Cyrl, s | Total kaz_Cyrl, s | Total zho_Hans, s | Server VRAM, MiB | CPU cores | GPU util, % | GPU power, W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28 | 10.4 | 95.96 | 4.223 | 6.520 | 6.863 | 7.249 | 6.546 | 174 | 7.4 | 8 | 16.9 |
| 4 | 25 | 12.4 | 80.52 | 3.816 | 5.491 | 5.805 | 6.164 | 5.522 | 1050 | 7.3 | 13 | 19.6 |
| 8 | 21 | 13.8 | 72.26 | 3.279 | 4.918 | 5.184 | 5.500 | 4.932 | 1600 | 7.1 | 16 | 21.6 |
| 14 | 15 | 17.2 | 58.03 | 2.537 | 3.942 | 4.148 | 4.388 | 3.952 | 2414 | 6.6 | 24 | 27.1 |
| 20 | 9 | 23.3 | 42.95 | 1.748 | 2.904 | 3.044 | 3.224 | 2.907 | 3222 | 5.9 | 36 | 39.4 |
| 24 | 5 | 30.5 | 32.74 | 1.159 | 2.195 | 2.289 | 2.421 | 2.200 | 3774 | 5.0 | 50 | 55.1 |
| 28 | 1 | 44.4 | 22.52 | 0.627 | 1.492 | 1.541 | 1.633 | 1.494 | 4324 | 3.2 | 77 | 79.7 |
| 29 | 0 | 51.9 | 19.27 | 0.479 | 1.270 | 1.307 | 1.386 | 1.273 | 4432 | 1.0 | 92 | 93.2 |

GPU utilization and power are whole-GPU readings sampled every 100 ms; server VRAM is the server process alone. Requests run one at a time with a fixed output budget, so these numbers describe single-request latency, not throughput under load.
