# Concurrent requests: parallel-14b-order-check-v1

32 requests per cell, language and round in a closed loop of `slots` clients; 2 rounds, each with a fresh server start; 64 generated tokens per request. Throughput is generated tokens per second of wall time, so prompt processing is inside it. Spread is the range between rounds relative to their mean.

| Cell | GPU layers | Blocks on CPU | Slots | Context/slot | Server VRAM, MiB | Language | Input tokens | Generated tok/s | Requests/s | Latency, s | Spread, % | CPU cores | GPU util, % |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| c256-np16-g39 | 39 | 10 | 16 | 256 | 7352 | eng_Latn | 93.1 | 44.6 | 0.697 | 22.84 | 0.9 | 4.4 | 38.8 |
| c256-np16-g39 | 39 | 10 | 16 | 256 | 7352 | zho_Hans | 92.0 | 47.5 | 0.743 | 21.43 | 4.1 | 4.5 | 43.0 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | kaz_Cyrl | 302.0 | 27.8 | 0.434 | 36.44 | 0.2 | 4.0 | 48.3 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | rus_Cyrl | 170.0 | 33.9 | 0.530 | 29.95 | 5.2 | 4.4 | 43.4 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | eng_Latn | 93.1 | 39.9 | 0.624 | 25.49 | 0.5 | 5.0 | 31.4 |
