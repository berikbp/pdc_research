# Experiment protocol and remaining stages

## Current question

Does multilingual content change the best CPU–GPU execution configuration after
input and output token counts are controlled? A dense transformer often executes
the same tensor shapes at equal sequence lengths; residual language effects must
be demonstrated rather than assumed. A negative result is a valid outcome.

## Pilot sample budget

`pilot.json` uses a frozen, curated selection of 20 non-overlapping passages,
each containing four consecutive FLORES dev entries from one source article.
The same IDs appear in all four languages: 80 aligned inputs from 80 distinct
source sentence IDs. English/Kazakh passages were reviewed for topical continuity
and a clear opening subject. This is a purposive sample, not a random or representative
sample of FLORES; source translations and wording remain unchanged.

With 12 machine configurations and one repetition, the initial aligned experiment
has **960 measured requests** (48 per prompt group). Matched token-prefix workloads
are disabled for this phase. Results are saved per request and groups run in order;
see [running and resuming](running-and-resuming.md).

This is an exploratory characterization, not a statistically justified final sample
size. Later controlled workloads can use matched 64/128-token prefixes, retaining a
group only if every language has enough tokens. Repeated trials will be necessary
for uncertainty estimates and stable claims about configuration differences.

## What is controlled

- Model file/quantization and exact tokenizer; backend/build recorded per machine.
- Batch size, microbatch size, context limit, f16 KV cache, Flash Attention off.
- CPU threads for both prefill and decode; GPU layers swept explicitly.
- One server slot, sequential requests, no prompt cache reuse, no context shifting.
- Greedy generation, fixed seed, 64 generated tokens, EOS ignored for the controlled test.
- Configuration and language order randomized within each prompt with the recorded seed.
- At least one unmeasured warm-up request per configuration.

Inputs are bare FLORES text as model token IDs, not chat-template conversations.
Special tokens are added by the model tokenizer once. Matched workloads clip token
prefixes, so semantic equivalence is lost. Keep aligned-content and matched-length
results separate. Actual decoded output is retained, but language quality is not scored.

## Measurements implemented

The real-text runner retains server-reported prompt/decode timings and throughputs,
actual token counts, and client elapsed time. Client time includes local HTTP overhead
but excludes model loading and the separately executed tokenization call. Tokenization
latency, streaming time-to-first-token, peak VRAM/RAM, utilization, and energy are
**not measured yet**. A pre-run `nvidia-smi` snapshot is not a peak-memory measurement.

`llama-bench` measures synthetic inference, excluding tokenization/sampling. Its
results serve as the length-aware hardware baseline rather than proof of language effects.

## Next stages

1. Run smoke and pilot on Berik's laptop; validate timings, cache assertions, offload
   logs, actual sample counts, and runtime. Test teammate setup independently.
2. Add process/GPU telemetry and streaming TTFT if reliable on each OS/backend.
3. Freeze the main study's prompt budget and length strata using pilot variance
   and runtime. Candidate: 100 groups, five repetitions, and short/medium/long
   contexts. This is provisional and needs an eligibility check and compute budget.
4. Use dev for tuning; reserve devtest for held-out evaluation. Split related sentence
   groups together to avoid leakage. Require enough independent groups per stratum.
5. Analyze each machine separately: paired language/configuration comparisons,
   confidence intervals over independent groups, and optimal-configuration stability.
   Repetitions estimate noise; they are not independent new prompts. Correct for
   multiple comparisons where needed; test whether effects exceed practical noise.
6. Compare a fixed configuration and a length-aware selector on held-out data.
   Add tokenizer features only if RQ1 supports them. Include selection and
   configuration-switching costs; model reloads cannot be treated as free.
7. Replicate across machines with the same workload/model hashes. Report OS,
   backend, RAM limits, GPU power modes, and failed/OOM settings explicitly.

Do not pool raw timings from different machines into one language comparison.
Do not select a configuration from incomplete runs without accounting for failures.

## Sources

- [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
- [llama-server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen2.5-7B GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
