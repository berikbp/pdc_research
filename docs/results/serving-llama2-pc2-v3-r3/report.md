# Serving under load: serving-llama2-pc2-v3-r3 on lab-pc2-rtx4080

Closed loop of `load` clients; 1 rounds, a fresh server start per cell; 128 generated tokens per request. Throughput is generated tokens per second of wall time. NEO is compared with the same engine run with `--always-use-gpu` (GPU-only) at the same KV-cache budget; vLLM is an external reference. `neo` is NEO as shipped (scheduler CPU launch time 0.8 ms per layer); `neo_tuned` uses the launch time measured on this machine (0.16955984111215608 ms).

## Where NEO overtakes GPU-only

| NEO mode | Memory | Language | Smallest winning load | Gain at each load, % (* = GPU-only saturated) |
|---|---|---|---:|---|
| neo_tuned | b0064 | eng_Latn | 16 | 8: -6.2*, 16: +40.2* |
| neo_tuned | b0064 | kaz_Cyrl | 8 | 8: +44.4*, 16: +74.0* |
| neo_tuned | b0064 | kaz_Cyrl@eng_len | 16 | 8: -6.5*, 16: +38.8* |
| neo_tuned | b0064 | rus_Cyrl | 8 | 8: +10.1*, 16: +60.7* |
| neo_tuned | b0064 | zho_Hans | 8 | 8: +16.5*, 16: +58.4* |
| neo_tuned | b0096 | eng_Latn | 8 | 8: +0.2*, 16: -3.6* |
| neo_tuned | b0096 | kaz_Cyrl | 8 | 8: +1.9*, 16: +25.5* |
| neo_tuned | b0096 | kaz_Cyrl@eng_len | 8 | 8: +0.1*, 16: -5.1* |
| neo_tuned | b0096 | rus_Cyrl | 16 | 8: -2.7*, 16: +21.7* |
| neo_tuned | b0096 | zho_Hans | 16 | 8: -0.2*, 16: +19.0* |

## All cells

| System | Memory | Load | Language | Input tokens | Generated tok/s | Steady tok/s | Spread, % | Latency, s | p95, s | TTFT, s | CPU decode share | Swapped out | CPU cores | GPU util, % |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neo_base | b0064 | 8 | eng_Latn | 107.1 | 218.7 | 214.3 | 0.0 | 4.31 | 5.78 | 0.67 | 0.00 | 27 | 1.1 | 98 |
| neo_base | b0064 | 16 | eng_Latn | 108.6 | 212.3 | 222.5 | 0.0 | 8.56 | 10.05 | 4.79 | 0.00 | 61 | 1.1 | 98 |
| neo_tuned | b0064 | 8 | eng_Latn | 107.1 | 205.1 | 225.7 | 0.0 | 4.51 | 6.50 | 0.09 | 0.06 | 22 | 2.2 | 97 |
| neo_tuned | b0064 | 16 | eng_Latn | 108.6 | 297.7 | 315.7 | 0.0 | 6.80 | 7.04 | 0.18 | 0.67 | 16 | 8.3 | 88 |
| neo_base | b0064 | 8 | kaz_Cyrl | 279.8 | 100.8 | 104.5 | 0.0 | 9.13 | 11.20 | 5.65 | 0.00 | 19 | 1.1 | 98 |
| neo_base | b0064 | 16 | kaz_Cyrl | 281.6 | 101.1 | 100.1 | 0.0 | 18.13 | 21.70 | 14.69 | 0.00 | 36 | 1.1 | 98 |
| neo_tuned | b0064 | 8 | kaz_Cyrl | 279.8 | 145.5 | 150.8 | 0.0 | 6.98 | 7.26 | 0.27 | 0.68 | 5 | 8.4 | 88 |
| neo_tuned | b0064 | 16 | kaz_Cyrl | 281.6 | 176.0 | 188.9 | 0.0 | 10.77 | 14.62 | 0.29 | 0.77 | 5 | 8.3 | 75 |
| neo_base | b0064 | 8 | kaz_Cyrl@eng_len | 107.1 | 216.6 | 213.5 | 0.0 | 4.32 | 5.77 | 0.71 | 0.00 | 26 | 1.1 | 98 |
| neo_base | b0064 | 16 | kaz_Cyrl@eng_len | 108.6 | 212.7 | 221.4 | 0.0 | 8.55 | 10.07 | 4.80 | 0.00 | 59 | 1.1 | 98 |
| neo_tuned | b0064 | 8 | kaz_Cyrl@eng_len | 107.1 | 202.5 | 222.9 | 0.0 | 4.59 | 6.58 | 0.10 | 0.08 | 23 | 2.5 | 97 |
| neo_tuned | b0064 | 16 | kaz_Cyrl@eng_len | 108.6 | 295.3 | 313.1 | 0.0 | 6.86 | 7.12 | 0.20 | 0.66 | 17 | 8.3 | 88 |
| neo_base | b0064 | 8 | rus_Cyrl | 180.6 | 148.9 | 160.0 | 0.0 | 6.24 | 7.72 | 2.71 | 0.00 | 20 | 1.1 | 98 |
| neo_base | b0064 | 16 | rus_Cyrl | 181.9 | 149.3 | 151.6 | 0.0 | 12.21 | 14.92 | 8.74 | 0.00 | 41 | 1.1 | 98 |
| neo_tuned | b0064 | 8 | rus_Cyrl | 180.6 | 163.9 | 173.9 | 0.0 | 6.07 | 6.51 | 0.17 | 0.53 | 8 | 7.8 | 94 |
| neo_tuned | b0064 | 16 | rus_Cyrl | 181.9 | 240.0 | 268.1 | 0.0 | 8.08 | 8.79 | 0.24 | 0.76 | 8 | 8.0 | 79 |
| neo_base | b0064 | 8 | zho_Hans | 208.8 | 135.2 | 143.3 | 0.0 | 6.89 | 8.97 | 3.51 | 0.00 | 16 | 1.1 | 98 |
| neo_base | b0064 | 16 | zho_Hans | 210.4 | 133.4 | 134.8 | 0.0 | 13.62 | 15.95 | 10.24 | 0.00 | 38 | 1.1 | 98 |
| neo_tuned | b0064 | 8 | zho_Hans | 208.8 | 157.5 | 173.7 | 0.0 | 6.21 | 6.61 | 0.19 | 0.56 | 6 | 7.8 | 93 |
| neo_tuned | b0064 | 16 | zho_Hans | 210.4 | 211.3 | 251.6 | 0.0 | 8.79 | 11.58 | 0.25 | 0.76 | 8 | 7.7 | 78 |
| neo_base | b0096 | 8 | eng_Latn | 107.1 | 267.9 | 342.7 | 0.0 | 3.29 | 4.07 | 0.14 | 0.00 | 8 | 1.1 | 98 |
| neo_base | b0096 | 16 | eng_Latn | 108.6 | 319.3 | 310.0 | 0.0 | 5.97 | 6.78 | 2.12 | 0.00 | 51 | 1.1 | 98 |
| neo_tuned | b0096 | 8 | eng_Latn | 107.1 | 268.5 | 337.7 | 0.0 | 3.32 | 4.36 | 0.11 | 0.00 | 9 | 1.1 | 98 |
| neo_tuned | b0096 | 16 | eng_Latn | 108.6 | 307.8 | 323.5 | 0.0 | 6.27 | 7.74 | 0.17 | 0.45 | 24 | 7.0 | 93 |
| neo_base | b0096 | 8 | kaz_Cyrl | 279.8 | 151.2 | 147.2 | 0.0 | 6.08 | 7.12 | 2.68 | 0.00 | 15 | 1.1 | 98 |
| neo_base | b0096 | 16 | kaz_Cyrl | 281.6 | 152.7 | 157.1 | 0.0 | 11.85 | 14.39 | 8.45 | 0.00 | 29 | 1.1 | 98 |
| neo_tuned | b0096 | 8 | kaz_Cyrl | 279.8 | 154.1 | 169.7 | 0.0 | 6.37 | 7.12 | 0.23 | 0.46 | 7 | 7.3 | 92 |
| neo_tuned | b0096 | 16 | kaz_Cyrl | 281.6 | 191.7 | 198.7 | 0.0 | 9.98 | 15.16 | 0.31 | 0.70 | 10 | 8.3 | 76 |
| neo_base | b0096 | 8 | kaz_Cyrl@eng_len | 107.1 | 268.4 | 343.5 | 0.0 | 3.28 | 4.07 | 0.15 | 0.00 | 8 | 1.1 | 98 |
| neo_base | b0096 | 16 | kaz_Cyrl@eng_len | 108.6 | 321.8 | 311.8 | 0.0 | 5.93 | 6.72 | 2.12 | 0.00 | 49 | 1.1 | 98 |
| neo_tuned | b0096 | 8 | kaz_Cyrl@eng_len | 107.1 | 268.5 | 342.4 | 0.0 | 3.32 | 4.20 | 0.10 | 0.00 | 9 | 1.1 | 97 |
| neo_tuned | b0096 | 16 | kaz_Cyrl@eng_len | 108.6 | 305.5 | 324.9 | 0.0 | 6.31 | 7.69 | 0.16 | 0.47 | 25 | 7.2 | 93 |
| neo_base | b0096 | 8 | rus_Cyrl | 180.6 | 214.7 | 219.4 | 0.0 | 4.36 | 6.38 | 0.89 | 0.00 | 18 | 1.1 | 98 |
| neo_base | b0096 | 16 | rus_Cyrl | 181.9 | 214.1 | 218.8 | 0.0 | 8.43 | 9.74 | 4.73 | 0.00 | 43 | 1.1 | 98 |
| neo_tuned | b0096 | 8 | rus_Cyrl | 180.6 | 208.9 | 216.3 | 0.0 | 4.53 | 6.53 | 0.13 | 0.05 | 18 | 2.0 | 97 |
| neo_tuned | b0096 | 16 | rus_Cyrl | 181.9 | 260.6 | 280.1 | 0.0 | 7.69 | 8.28 | 0.27 | 0.64 | 11 | 8.2 | 84 |
| neo_base | b0096 | 8 | zho_Hans | 208.8 | 185.1 | 199.1 | 0.0 | 4.84 | 6.44 | 1.14 | 0.00 | 21 | 1.1 | 98 |
| neo_base | b0096 | 16 | zho_Hans | 210.4 | 197.6 | 203.7 | 0.0 | 9.39 | 11.97 | 5.87 | 0.00 | 35 | 1.1 | 98 |
| neo_tuned | b0096 | 8 | zho_Hans | 208.8 | 184.8 | 194.9 | 0.0 | 5.11 | 6.43 | 0.15 | 0.14 | 13 | 3.3 | 97 |
| neo_tuned | b0096 | 16 | zho_Hans | 210.4 | 235.2 | 261.2 | 0.0 | 8.31 | 9.24 | 0.27 | 0.66 | 12 | 8.0 | 80 |

## Language order within a server start

| Position | Blocks | Throughput relative to the cell mean |
|---:|---:|---:|
| 0 | 8 | 1.000 |
| 1 | 8 | 1.000 |
| 2 | 8 | 1.000 |
| 3 | 8 | 1.000 |
| 4 | 8 | 1.000 |
