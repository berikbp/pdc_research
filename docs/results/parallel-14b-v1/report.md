# Concurrent requests: parallel-14b-v1

32 requests per cell, language and round in a closed loop of `slots` clients; 2 rounds, each with a fresh server start; 64 generated tokens per request. Throughput is generated tokens per second of wall time, so prompt processing is inside it. Spread is the range between rounds relative to their mean.

| Cell | GPU layers | Blocks on CPU | Slots | Context/slot | Server VRAM, MiB | Language | Input tokens | Generated tok/s | Requests/s | Latency, s | Spread, % | CPU cores | GPU util, % |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| c256-np01-g43 | 43 | 6 | 1 | 256 | 7416 | eng_Latn | 93.5 | 15.7 | 0.245 | 4.09 | 0.1 | 4.0 | 53.6 |
| c256-np01-g43 | 43 | 6 | 1 | 256 | 7416 | zho_Hans | 93.9 | 15.6 | 0.244 | 4.09 | 0.3 | 4.0 | 56.9 |
| c256-np04-g42 | 42 | 7 | 4 | 256 | 7388 | eng_Latn | 93.5 | 35.0 | 0.547 | 7.31 | 0.1 | 4.2 | 48.6 |
| c256-np04-g42 | 42 | 7 | 4 | 256 | 7388 | zho_Hans | 93.9 | 35.4 | 0.553 | 7.24 | 0.6 | 4.1 | 45.1 |
| c256-np16-g39 | 39 | 10 | 16 | 256 | 7352 | eng_Latn | 93.5 | 47.5 | 0.742 | 21.45 | 2.8 | 4.6 | 36.2 |
| c256-np16-g39 | 39 | 10 | 16 | 256 | 7352 | zho_Hans | 93.9 | 41.2 | 0.644 | 24.68 | 1.8 | 4.7 | 34.4 |
| c512-np01-g42 | 42 | 7 | 1 | 512 | 7306 | kaz_Cyrl | 302.4 | 13.5 | 0.211 | 4.74 | 0.4 | 4.0 | 56.3 |
| c512-np01-g42 | 42 | 7 | 1 | 512 | 7306 | rus_Cyrl | 170.3 | 14.3 | 0.224 | 4.46 | 0.2 | 4.2 | 54.1 |
| c512-np01-g42 | 42 | 7 | 1 | 512 | 7306 | eng_Latn | 93.5 | 14.9 | 0.234 | 4.28 | 0.0 | 4.3 | 53.5 |
| c512-np04-g41 | 41 | 8 | 4 | 512 | 7380 | kaz_Cyrl | 302.4 | 26.5 | 0.414 | 9.65 | 1.4 | 3.7 | 54.1 |
| c512-np04-g41 | 41 | 8 | 4 | 512 | 7380 | rus_Cyrl | 170.3 | 30.4 | 0.475 | 8.41 | 1.1 | 4.1 | 46.6 |
| c512-np04-g41 | 41 | 8 | 4 | 512 | 7380 | eng_Latn | 93.5 | 33.3 | 0.521 | 7.68 | 1.4 | 4.4 | 43.7 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | kaz_Cyrl | 302.4 | 27.4 | 0.427 | 37.06 | 5.9 | 3.9 | 49.7 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | rus_Cyrl | 170.3 | 35.2 | 0.550 | 28.87 | 0.8 | 4.4 | 39.7 |
| c512-np16-g36 | 36 | 13 | 16 | 512 | 7386 | eng_Latn | 93.5 | 41.1 | 0.642 | 24.79 | 1.9 | 5.1 | 36.1 |
