# Supervisor progress summary

Multilingual CPU–GPU inference: progress summary

Supervisor discussion | 26 October 2026 | Initial-machine results

## Question and current answer

RQ1 asks whether language changes the best CPU–GPU configuration after controlling tokenized sequence length. In the controlled 64-input / 64-output-token experiment, we found no stable, practically meaningful language-dependent configuration preference among nine tested settings. This is a restricted negative finding; RQ2 (policy improvement) remains unanswered.

## What was done

Stage 1: 960 aligned-content trials across 12 thread/offload settings. Full layer offload won all 80 input comparisons, but token counts differed and there was one repetition. Kazakh averaged 298.8 input tokens versus 92.8 for English.

Stage 2: 2,160 trials covering the CPU-thread sweep, ten-thread follow-up and GPU-only model baseline. This was useful hardware/backend characterization but partly a detour from RQ1. More CPU threads did not consistently help. A diagnostic found CPU embedding computation under default full offload; explicitly moving the embedding to CUDA corrected the GPU-only model baseline.

Stage 3: 2,052 trials = 19 eligible passage groups × four languages × nine configurations × three shuffled rounds. Every trial used 64 input and 64 output tokens. All records passed validation; zero error records. Total across all stages: 5,172 measured requests, excluding warmups and diagnostics.

## Controlled measurements

| Configuration | Mean inference |
|---|---:|
| GPU-only model graph | 1.433 s |
| Full offload / CPU embedding / 4 threads | 1.434 s |
| 28 GPU layers / 4 threads | 1.677 s |
| 24 GPU layers / 8 threads | 2.494 s |

Means include prompt processing and generation; loading and warmups are excluded. GPU-only model computation permits CPU orchestration/sampling. The GPU-only baseline averaged 1.434 s English, 1.433 s Russian, 1.434 s Kazakh and 1.432 s Chinese.

## Uncertainty and limits

10,000 passage-cluster bootstrap resamples; approximate simultaneous 95% intervals. Of 216 language-by-configuration interactions, 201 intervals lay within the predeclared ±2-percentage-point practical band; 15 were inconclusive. One small interaction excluded zero but stayed below the practical threshold. None met the stable, practically meaningful criterion.

One laptop (Ryzen 7 7435HS / RTX 4060 8 GiB, WSL), one model (Qwen2.5-7B-Instruct Q4_K_M), one short length, curated dev passages and rounds within one session. Prefix clipping loses semantic equivalence. No power/utilization telemetry, held-out policy evaluation, universal equivalence or unique CPU-core optimum is established.

## Decision requested from the supervisor

Recommended: replicate the controlled protocol on teammates’ machines, then test longer contexts if warranted. Alternatively, formally revise the question toward memory-constrained CPU–GPU cooperation. Do not build a language-aware selector unless replicated measurements show a useful signal.

Evidence: docs/current-report.md and docs/results/rq1-controlled-64-v1/. Repository: github.com/berikbp/pdc_research.
