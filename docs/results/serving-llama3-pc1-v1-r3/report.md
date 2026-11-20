# Serving under load: serving-llama3-pc1-v1-r3 on lab-pc1-rtx5090-llama3

Closed loop of `load` clients; 1 rounds, a fresh server start per cell; 128 generated tokens per request. Throughput is generated tokens per second of wall time. NEO is compared with the same engine run with `--always-use-gpu` (GPU-only) at the same KV-cache budget; vLLM is an external reference. `neo` is NEO as shipped (scheduler CPU launch time 0.8 ms per layer); `neo_tuned` uses the launch time measured on this machine (0.18421131746921068 ms).

## Where NEO overtakes GPU-only

| NEO mode | Memory | Language | Smallest winning load | Gain at each load, % (* = GPU-only saturated) |
|---|---|---|---:|---|
| neo_tuned | b0064 | eng_Latn | 32 | 16: -0.9*, 32: +40.9* |
| neo_tuned | b0064 | kaz_Cyrl | 16 | 16: +65.1*, 32: +79.1* |
| neo_tuned | b0064 | kaz_Cyrl@eng_len | 16 | 16: +35.9*, 32: +50.1* |
| neo_tuned | b0064 | rus_Cyrl | 16 | 16: +80.4*, 32: +67.0* |
| neo_tuned | b0064 | zho_Hans | 16 | 16: +9.9*, 32: +62.2* |
| neo_tuned | b0096 | eng_Latn | 32 | 16: -7.3*, 32: +11.9* |
| neo_tuned | b0096 | kaz_Cyrl | 16 | 16: +13.3*, 32: +17.3* |
| neo_tuned | b0096 | kaz_Cyrl@eng_len | 32 | 16: -19.5*, 32: +9.0* |
| neo_tuned | b0096 | rus_Cyrl | 32 | 16: -9.3*, 32: +16.5* |
| neo_tuned | b0096 | zho_Hans | 32 | 16: -15.8*, 32: +14.1* |

## All cells

| System | Memory | Load | Language | Input tokens | Generated tok/s | Steady tok/s | Spread, % | Latency, s | p95, s | TTFT, s | CPU decode share | Swapped out | CPU cores | GPU util, % |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neo_base | b0064 | 16 | eng_Latn | 92.4 | 457.3 | 471.5 | 0.0 | 4.01 | 4.76 | 2.03 | 0.00 | 59 | 1.1 | 91 |
| neo_base | b0064 | 32 | eng_Latn | 92.6 | 467.0 | 477.5 | 0.0 | 7.84 | 9.47 | 5.82 | 0.00 | 122 | 1.1 | 92 |
| neo_tuned | b0064 | 16 | eng_Latn | 92.4 | 453.3 | 440.8 | 0.0 | 4.45 | 5.26 | 0.08 | 0.63 | 22 | 3.2 | 66 |
| neo_tuned | b0064 | 32 | eng_Latn | 92.6 | 658.2 | 754.8 | 0.0 | 5.71 | 7.04 | 0.13 | 0.78 | 29 | 2.3 | 57 |
| neo_base | b0064 | 16 | kaz_Cyrl | 298.5 | 181.8 | 178.6 | 0.0 | 10.06 | 11.95 | 8.22 | 0.00 | 37 | 1.1 | 92 |
| neo_base | b0064 | 32 | kaz_Cyrl | 300.1 | 182.2 | 182.1 | 0.0 | 19.85 | 23.30 | 18.03 | 0.00 | 71 | 1.1 | 92 |
| neo_tuned | b0064 | 16 | kaz_Cyrl | 298.5 | 300.2 | 306.6 | 0.0 | 6.42 | 9.60 | 0.14 | 0.81 | 9 | 2.4 | 57 |
| neo_tuned | b0064 | 32 | kaz_Cyrl | 300.1 | 326.3 | 340.4 | 0.0 | 11.25 | 15.26 | 0.23 | 0.84 | 15 | 2.6 | 56 |
| neo_base | b0064 | 16 | kaz_Cyrl@eng_len | 92.4 | 459.9 | 474.3 | 0.0 | 4.00 | 4.73 | 2.02 | 0.00 | 60 | 1.1 | 91 |
| neo_base | b0064 | 32 | kaz_Cyrl@eng_len | 92.6 | 465.7 | 474.9 | 0.0 | 7.86 | 9.50 | 5.86 | 0.00 | 121 | 1.1 | 91 |
| neo_tuned | b0064 | 16 | kaz_Cyrl@eng_len | 92.4 | 624.8 | 663.2 | 0.0 | 3.22 | 3.40 | 0.07 | 0.63 | 21 | 7.9 | 90 |
| neo_tuned | b0064 | 32 | kaz_Cyrl@eng_len | 92.6 | 699.0 | 824.0 | 0.0 | 5.38 | 6.73 | 0.13 | 0.78 | 28 | 2.5 | 62 |
| neo_base | b0064 | 16 | rus_Cyrl | 152.8 | 323.8 | 335.7 | 0.0 | 5.58 | 6.40 | 3.74 | 0.00 | 44 | 1.1 | 92 |
| neo_base | b0064 | 32 | rus_Cyrl | 154.0 | 332.4 | 334.1 | 0.0 | 10.96 | 12.76 | 9.12 | 0.00 | 89 | 1.1 | 91 |
| neo_tuned | b0064 | 16 | rus_Cyrl | 152.8 | 584.1 | 619.8 | 0.0 | 3.47 | 3.62 | 0.11 | 0.72 | 14 | 8.0 | 86 |
| neo_tuned | b0064 | 32 | rus_Cyrl | 154.0 | 555.0 | 599.8 | 0.0 | 6.74 | 9.37 | 0.16 | 0.82 | 23 | 2.7 | 60 |
| neo_base | b0064 | 16 | zho_Hans | 117.8 | 395.7 | 406.0 | 0.0 | 4.64 | 5.63 | 2.71 | 0.00 | 56 | 1.1 | 92 |
| neo_base | b0064 | 32 | zho_Hans | 118.8 | 393.9 | 398.1 | 0.0 | 9.14 | 11.08 | 7.20 | 0.00 | 108 | 1.1 | 91 |
| neo_tuned | b0064 | 16 | zho_Hans | 117.8 | 434.8 | 455.4 | 0.0 | 4.67 | 4.84 | 0.10 | 0.68 | 17 | 1.8 | 63 |
| neo_tuned | b0064 | 32 | zho_Hans | 118.8 | 639.0 | 668.7 | 0.0 | 5.93 | 8.15 | 0.14 | 0.81 | 23 | 2.6 | 60 |
| neo_base | b0096 | 16 | eng_Latn | 92.4 | 666.3 | 685.9 | 0.0 | 2.70 | 3.25 | 0.72 | 0.00 | 62 | 1.1 | 93 |
| neo_base | b0096 | 32 | eng_Latn | 92.6 | 673.1 | 709.7 | 0.0 | 5.34 | 6.55 | 3.27 | 0.00 | 117 | 1.1 | 91 |
| neo_tuned | b0096 | 16 | eng_Latn | 92.4 | 617.8 | 670.1 | 0.0 | 3.03 | 4.76 | 0.05 | 0.28 | 33 | 4.9 | 88 |
| neo_tuned | b0096 | 32 | eng_Latn | 92.6 | 753.1 | 841.0 | 0.0 | 5.11 | 6.01 | 0.12 | 0.69 | 35 | 2.4 | 63 |
| neo_base | b0096 | 16 | kaz_Cyrl | 298.5 | 287.4 | 289.6 | 0.0 | 6.34 | 7.86 | 4.62 | 0.00 | 26 | 1.1 | 92 |
| neo_base | b0096 | 32 | kaz_Cyrl | 300.1 | 292.0 | 293.4 | 0.0 | 12.45 | 14.47 | 10.74 | 0.00 | 52 | 1.1 | 92 |
| neo_tuned | b0096 | 16 | kaz_Cyrl | 298.5 | 325.6 | 364.4 | 0.0 | 5.75 | 7.90 | 0.14 | 0.70 | 8 | 2.3 | 59 |
| neo_tuned | b0096 | 32 | kaz_Cyrl | 300.1 | 342.4 | 347.2 | 0.0 | 10.91 | 15.75 | 0.23 | 0.76 | 14 | 2.4 | 53 |
| neo_base | b0096 | 16 | kaz_Cyrl@eng_len | 92.4 | 650.9 | 680.8 | 0.0 | 2.74 | 3.24 | 0.71 | 0.00 | 59 | 1.1 | 91 |
| neo_base | b0096 | 32 | kaz_Cyrl@eng_len | 92.6 | 677.1 | 714.1 | 0.0 | 5.30 | 6.55 | 3.24 | 0.00 | 116 | 1.1 | 92 |
| neo_tuned | b0096 | 16 | kaz_Cyrl@eng_len | 92.4 | 524.2 | 587.2 | 0.0 | 3.51 | 5.72 | 0.06 | 0.27 | 26 | 1.3 | 78 |
| neo_tuned | b0096 | 32 | kaz_Cyrl@eng_len | 92.6 | 738.2 | 838.2 | 0.0 | 5.25 | 6.13 | 0.14 | 0.69 | 30 | 2.3 | 62 |
| neo_base | b0096 | 16 | rus_Cyrl | 152.8 | 493.7 | 501.0 | 0.0 | 3.79 | 4.93 | 1.89 | 0.00 | 42 | 1.1 | 92 |
| neo_base | b0096 | 32 | rus_Cyrl | 154.0 | 494.7 | 489.0 | 0.0 | 7.45 | 8.52 | 5.56 | 0.00 | 82 | 1.1 | 92 |
| neo_tuned | b0096 | 16 | rus_Cyrl | 152.8 | 447.8 | 473.7 | 0.0 | 4.52 | 4.80 | 0.10 | 0.59 | 16 | 1.9 | 66 |
| neo_tuned | b0096 | 32 | rus_Cyrl | 154.0 | 576.4 | 599.9 | 0.0 | 6.66 | 9.71 | 0.16 | 0.76 | 23 | 2.5 | 58 |
| neo_base | b0096 | 16 | zho_Hans | 117.8 | 560.6 | 587.7 | 0.0 | 3.20 | 3.82 | 1.22 | 0.00 | 51 | 1.1 | 92 |
| neo_base | b0096 | 32 | zho_Hans | 118.8 | 581.0 | 581.6 | 0.0 | 6.27 | 7.58 | 4.30 | 0.00 | 98 | 1.1 | 90 |
| neo_tuned | b0096 | 16 | zho_Hans | 117.8 | 472.0 | 504.0 | 0.0 | 4.14 | 4.79 | 0.08 | 0.49 | 20 | 1.7 | 69 |
| neo_tuned | b0096 | 32 | zho_Hans | 118.8 | 663.2 | 772.0 | 0.0 | 5.60 | 7.48 | 0.14 | 0.69 | 29 | 2.4 | 61 |

## Language order within a server start

| Position | Blocks | Throughput relative to the cell mean |
|---:|---:|---:|
| 0 | 8 | 1.000 |
| 1 | 8 | 1.000 |
| 2 | 8 | 1.000 |
| 3 | 8 | 1.000 |
| 4 | 8 | 1.000 |
