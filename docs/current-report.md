# Current research report from Berik

**Status: 25 October 2026.** All experiments described here are complete. This report records the direction taken, the measurements, the corrections to their interpretation, and what remains unanswered.

## Main finding

> For Qwen2.5-7B-Instruct Q4_K_M on the tested Ryzen 7 7435HS / RTX 4060 laptop, with 64 input and 64 output tokens, we found no stable, practically meaningful evidence that language changes the performance-optimal CPU–GPU configuration among the nine configurations tested.

This is a restricted negative finding for RQ1. It does not prove that language never matters, identify one uniquely optimal thread count, or answer RQ2. Some uncertainty intervals remain too wide to establish practical equivalence.

## Research questions and scope

From the [proposal](../Research_Proposal.md):

- **RQ1:** Do multilingual workloads with different tokenization characteristics exhibit different performance-optimal CPU–GPU execution configurations after controlling for tokenized sequence length?
- **RQ2:** If such differences exist, can a tokenizer-aware or workload-aware configuration policy outperform both a fixed CPU–GPU configuration and a conventional sequence-length-aware policy?

The project studies CPU–GPU execution configuration using multilingual workloads. Finding the best CPU thread count is a component of configuration tuning; by itself, it does not answer whether the best configuration depends on language beyond sequence length. No selector or held-out policy comparison has been evaluated.

## Hardware and common measurement protocol

- AMD Ryzen 7 7435HS: 8 physical cores / 16 logical CPUs; RTX 4060 Laptop GPU, approximately 8 GiB VRAM; WSL. CPU affinity was not pinned.
- Pinned llama.cpp revision: `ecdd29e8b2076bb85da9e9ca7ee3d84264375bd1`.
- Qwen2.5-7B-Instruct, Q4_K_M. Merged model SHA-256: `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49`.
- FLORES-200 dev inputs in English, Russian, Kazakh and Simplified Chinese. The 20 original passages were curated for topic coherence, with four consecutive source entries from a distinct article per group. This is not a representative random sample.
- Bare-text continuation, greedy decoding, fixed 64 output tokens, EOS ignored, prompt cache disabled, one sequential server slot. These are performance workloads, not a factuality or translation-quality assessment.
- Primary latency is server-reported prompt processing plus generation, excluding loading, tokenization setup, warmups and HTTP overhead. Do not interpret it as streaming time-to-first-token.

Power, energy, peak memory, CPU/GPU utilization and GPU clock/temperature telemetry were not measured. Server settings and fingerprints are recorded locally. Physical-core allocation and GPU-core counts were not swept.

## Experiment history

| Stage | Purpose | Measured requests | Role in the research |
|---|---|---:|---|
| 1. Aligned-content pilot | Compare 12 thread/offload settings on equivalent passages | 960 | Correct initial direction; unequal input lengths and one repetition prevent a controlled RQ1 answer |
| 2a. CPU thread sweep | Full layer offload, 1/2/4/6/8/12/16 threads, three repetitions | 1,680 | Hardware characterization; partly a detour from language-by-configuration RQ1 |
| 2b. Ten-thread follow-up | Add a missing thread setting | 240 | Separate-session comparison, not a controlled improvement claim |
| 2c. GPU-only model baseline | Explicit CUDA embedding with minimum thread setting | 240 | Correct placement interpretation; separate-session exploratory baseline |
| 3. Controlled RQ1 test | Equal token counts, nine configurations, three shuffled rounds | 2,052 | Direct but restricted RQ1 test |
| **Total** | | **5,172** | Setup, warmups and short placement diagnostics excluded |

## Stage 1: aligned-content pilot

Run: `berik-pilot-v3`. Twenty passages × four languages × 12 configurations × one repetition = 960 requests. CPU thread settings were 4 and 8; GPU layer settings were 0, 8, 16, 24, 28 and 99. The value 99 requests all available offloadable layers; it is not a GPU core count.

| GPU layer setting | 4 CPU threads, seconds | 8 CPU threads, seconds |
|---|---:|---:|
| 0 | 9.6188 | 8.6741 |
| 8 | 7.8283 | 6.6971 |
| 16 | 5.1404 | 5.1108 |
| 24 | 2.7892 | 2.6817 |
| 28 | 1.7893 | 1.7707 |
| All requested (99) | 1.5134 | 1.5269 |

Each cell averages 80 measured inputs. Full layer offload won all 80 input comparisons, although the winning 4-versus-8-thread setting varied. This indicated no obvious language-specific offload advantage, but it did not isolate language from token length.

| Language | Mean input token count | Mean inference, full offload / 4 threads |
|---|---:|---:|
| English | 92.8 | 1.470 s |
| Russian | 168.2 | 1.513 s |
| Kazakh | 298.8 | 1.603 s |
| Chinese | 93.0 | 1.468 s |

Kazakh required about 3.2 times as many input tokens as English in these selected passages. That is a sample/tokenizer observation, not proof of a language-specific execution effect. Output repetition or factual mistakes were retained and were not scored.

**Assessment:** this was a useful exploratory stage in the right direction. Its limitations were unequal input lengths, only one measurement per input/configuration, and substantial loading overhead from the prompt-first schedule: 20 × 12 model loads, plus setup. The broad GPU placement benefit did not establish a novel scheduling contribution.

## Stage 2: CPU-thread investigation and baseline correction

### Why this stage was partly off direction

The study shifted toward “how many CPU threads help when layers are already on GPU?” That is useful hardware tuning, but it does not directly answer the proposal’s language-by-configuration question after controlling token length. This stage still used unequal-length inputs and collected repetitions within each configuration block. It also did not implement a backend that deliberately overlaps useful CPU and GPU model computation. Increasing `--threads` cannot be described as automatically adding CPU computing power to GPU execution.

All full-layer-offload thread measurements are below. Each repetition cell averages the same 80 inputs. These are three repetitions within a block, not three independently shuffled sessions.

| Placement / CPU threads | Repetition 1, s | Repetition 2, s | Repetition 3, s | Mean, s |
|---|---:|---:|---:|---:|
| Full layers, CPU embedding / 1 | 1.4733 | 1.4756 | 1.4743 | 1.4744 |
| Full layers, CPU embedding / 2 | 1.5011 | 1.5033 | 1.5032 | 1.5025 |
| Full layers, CPU embedding / 4 | 1.5177 | 1.5240 | 1.5201 | 1.5206 |
| Full layers, CPU embedding / 6 | 1.5241 | 1.5240 | 1.5264 | 1.5249 |
| Full layers, CPU embedding / 8 | 1.5217 | 1.5243 | 1.5234 | 1.5231 |
| Full layers, CPU embedding / 10 (follow-up) | 1.4989 | 1.5007 | 1.5042 | 1.5013 |
| Full layers, CPU embedding / 12 | 1.4764 | 1.4767 | 1.4755 | 1.4762 |
| Full layers, CPU embedding / 16 | 1.4770 | 1.4796 | 1.4801 | 1.4789 |
| CUDA embedding / 1 (baseline follow-up) | 1.4868 | 1.4873 | 1.4829 | 1.4857 |

One thread had the lowest initial average, but 12 and 16 were within 0.3%. Such small differences do not establish a unique optimum. Ten threads and the embedding-on-GPU baseline were collected in separate sessions; their differences cannot be attributed solely to the settings.

### What “GPU-only” actually means

The requested baseline permits CPU orchestration but excludes CPU model computation. Source inspection and short graph-placement diagnostics corrected our terminology:

- `--threads 0` automatically selects a hardware thread count; it does not disable inference workers.
- `-ngl 99` alone still placed token-embedding `GET_ROWS` on CPU in the checked build/model.
- Adding `-ot token_embd.weight=CUDA0`, with `-t 1 -tb 1`, assigned all logged model graph nodes to CUDA in the short diagnostic. CPU orchestration, sampling, driver and server activity remain allowed.
- Therefore the full-offload thread sweep must not be labeled “GPU-only model computation.” The explicit CUDA-embedding case is the verified candidate for that label. The diagnostic did not trace every timed request, and verbose diagnostic timings were excluded.

The GPU-model-only baseline averaged 1.4857 s, compared with 1.4744 s for the earlier one-thread CPU-embedding run: about 0.8% difference across sessions. This was not evidence that CPU assistance reliably improves performance. See [placement verification](gpu-baseline.md).

**Assessment:** this stage was useful for understanding the backend and correcting the baseline, but it was partly a detour. It should be presented honestly as supporting characterization rather than as an answer to RQ1 or proof that CPU–GPU cooperation helps or hurts.

## Stage 3: equal-length controlled RQ1 experiment

Run: `berik-rq1-controlled-v1`. Exactly 64 input tokens and 64 generated tokens for every language. Nineteen groups were eligible in all four languages; `dev-0606` was excluded because its Chinese input had only 56 tokens. Token-prefix clipping can cut sentences and destroys exact semantic equivalence. The natural aligned-content pilot and controlled prefix experiment answer different parts of the study.

Nine configurations × 76 inputs × three separately shuffled rounds = 2,052 requests. Each round revisited all configurations, with one model load per block and one unmeasured warmup. Repeated rounds reduce order effects but remain within one session. Settings and the 2% practical threshold were documented before execution in the [protocol](rq1-controlled.md).

### Three rounds side by side

| Configuration | Round 1, s | Round 2, s | Round 3, s | Mean, s |
|---|---:|---:|---:|---:|
| gpu-model-only | 1.4336 | 1.4342 | 1.4312 | 1.4330 |
| g99-t1 | 1.4342 | 1.4349 | 1.4527 | 1.4406 |
| g99-t4 | 1.4353 | 1.4315 | 1.4342 | 1.4337 |
| g99-t8 | 1.4542 | 1.4294 | 1.4343 | 1.4393 |
| g99-t16 | 1.4561 | 1.4331 | 1.4312 | 1.4401 |
| g28-t4 | 1.6792 | 1.6728 | 1.6795 | 1.6772 |
| g28-t8 | 1.6879 | 1.6902 | 1.6887 | 1.6889 |
| g24-t4 | 2.8345 | 2.6039 | 2.7042 | 2.7142 |
| g24-t8 | 2.5049 | 2.4845 | 2.4921 | 2.4938 |

`gpu-model-only`: explicit CUDA embedding, all layers on GPU, thread setting 1. `g99-tN`: full layer offload with default CPU embedding, N CPU threads. `g28-tN` / `g24-tN`: 28 / 24 layers requested on GPU with N CPU threads.

### Language comparison at equal token counts

| Configuration | English, s | Russian, s | Kazakh, s | Chinese, s |
|---|---:|---:|---:|---:|
| gpu-model-only | 1.4335 | 1.4326 | 1.4342 | 1.4318 |
| g99-t1 | 1.4419 | 1.4389 | 1.4453 | 1.4364 |
| g99-t4 | 1.4323 | 1.4312 | 1.4369 | 1.4344 |
| g99-t8 | 1.4443 | 1.4379 | 1.4366 | 1.4384 |
| g99-t16 | 1.4400 | 1.4385 | 1.4448 | 1.4371 |
| g28-t4 | 1.6736 | 1.6792 | 1.6761 | 1.6798 |
| g28-t8 | 1.6847 | 1.6907 | 1.6915 | 1.6888 |
| g24-t4 | 2.7176 | 2.7210 | 2.7105 | 2.7077 |
| g24-t8 | 2.4995 | 2.4906 | 2.4867 | 2.4984 |

GPU-only model execution averaged 1.4330 s; the default CPU-embedding four-thread configuration averaged 1.4337 s, a difference of about 0.05%. Four threads had slightly lower sample averages in English and Russian, while the GPU-only candidate had lower averages in Kazakh and Chinese. Those sample winner changes are too small to establish a useful language-specific preference.

Partial offload was materially slower across languages: 28 layers / four threads averaged 1.6772 s; 24 layers / eight threads averaged 2.4938 s. The 24-layer / four-thread condition also showed notable between-round drift, so attributing every timing difference to a parameter would be unjustified.

### Validation and uncertainty

All 2,052 records were unique and balanced across configuration, language, passage and round. All recorded 64 input / 64 output tokens, no input truncation and zero prompt-cache reuse. There were zero error records and 27 measured configuration loads.

The analysis used 10,000 bootstrap resamples of the 19 passage groups, retaining each sampled group’s languages, configurations and rounds together. It estimated paired relative-latency contrasts and language-by-configuration interactions. Separate simultaneous approximate 95% interval families adjust the 32 baseline comparisons and the 216 interactions. Repetitions were not treated as independent passages.

- **201/216** interaction intervals were wholly inside the predeclared ±2-percentage-point practical band.
- **15/216** were too wide to establish equivalence at that band; this is unresolved uncertainty, not proof of no effect.
- **One** interaction interval excluded zero after family adjustment: the English-versus-Chinese difference in `g99-t1` relative to `g28-t4`, estimated at 0.655 percentage points (simultaneous interval approximately 0.029–1.280). It was below the practical threshold.
- **Zero** interactions met all criteria of practical magnitude at least two points, an adjusted interval excluding zero, and consistent direction across rounds. The largest absolute interaction point estimate was 0.846 points.

The intervals quantify variation over the sampled passage clusters conditional on these rounds. They do not quantify future-session, machine, temperature or power-mode uncertainty. With only 19 curated groups, broad equivalence claims are not justified. See the [detailed controlled report](results/rq1-controlled-64-v1/report.md) and complete [interaction table](results/rq1-controlled-64-v1/language-interactions.csv).

## Answers to the research questions

| Question | Current answer |
|---|---|
| RQ1 | A restricted negative finding: no stable, practically meaningful language-dependent configuration preference detected among these nine settings at 64 input / 64 output tokens on this model/laptop. Some comparisons remain inconclusive for equivalence. |
| RQ2 | Unanswered. No policy was built or evaluated, and current data provides no demonstrated benefit that would justify a language-aware selector in this setting. |

The results do not establish novelty, universal GPU superiority, or a definitive physical-core optimum. Shared model-layer placement is not the same as an asynchronous CPU–GPU cooperative backend. Broader claims would require other context/output lengths, independent sessions, additional hardware, and appropriate prior-work comparison.

## Next work

1. Reproduce the controlled protocol on a teammate’s machine, retaining per-machine results rather than pooling raw times.
2. Extend lengths using workloads long enough in all languages; do not repeat or pad text to manufacture eligibility.
3. If addressing memory-constrained cooperation, select a model/workload that constrains VRAM and explicitly document the execution backend. This is a separate extension, not evidence already obtained.
4. Investigate RQ2 only if replicated results establish a stable, practically useful signal; otherwise report the negative characterization honestly.

## Artifacts and reproducibility

Git includes this report, the frozen protocols/configuration files, analysis code, and compact summary/uncertainty tables. Per-request JSON, generated model outputs, logs, token IDs and downloaded assets remain in ignored local folders. Their omission keeps cloning practical, but public summaries alone do not permit independent auditing of raw timings. Keep a backup; raw data can be shared separately as a research archive if needed.

- [Exploratory configuration summaries](results/history/exploratory-configurations.csv)
- [Exploratory run provenance](results/history/exploratory-provenance.json)
- [Controlled configuration summaries](results/rq1-controlled-64-v1/configurations.csv)
- [Controlled validation and record fingerprint](results/rq1-controlled-64-v1/validation.json)

Regenerate from retained local raw data:

```bash
uv run python scripts/summarize_history.py
uv run pdc-research analyze-controlled --run runs/berik-rq1-controlled-v1 --output runs/berik-rq1-controlled-v1/analysis
```

Analysis produces per-request CSV locally as well as summaries. Do not commit that per-request export. Older completed runs retain their original embedded fingerprints and were not rewritten during the repository refactor. Implementation fingerprints intentionally prevent resuming an old incomplete run after incompatible source changes.
