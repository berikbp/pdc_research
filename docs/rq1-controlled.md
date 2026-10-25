# RQ1 controlled 64-token experiment

Purpose: test whether language changes the best tested CPU/GPU configuration
when input and output token counts are fixed. This is a restricted RQ1 test on
one machine, one model and one short length; it does not generalize to all
lengths or prove absence of language effects.

Protocol frozen before execution:

- Original 20 curated FLORES passage groups, four languages.
- Clip token-ID prefixes to exactly 64 tokens, including any tokenizer-added
  special token. Generate exactly 64 tokens; greedy decoding, cache disabled.
- Retain groups only when all languages have at least 64 tokens. Expected: 19
  groups / 76 inputs; dev-0606 is excluded because Chinese has 56 tokens.
- Clipping loses semantic equivalence and can cut off text mid-sentence. Original
  source text is retained for provenance; measured input is the token prefix in
  controlled-cases.json, not the full original text.
- Nine configurations in configs/experiments/rq1-controlled.json: GPU model-only
  with explicit CUDA embedding and thread setting 1; default full offload with
  CPU embedding at 1/4/8/16 threads; 28 and 24 GPU layers at 4/8 threads.
- Three separately shuffled rounds. Each round visits every configuration once
  and each input once per configuration. Requests are shuffled within blocks.
  Seeds and exact 27-block schedule are saved. No thread/core affinity.
- 2,052 measured requests, 27 configuration loads, one tokenization setup load
  and one excluded warmup per block on an uninterrupted run. Resuming a partial
  block reloads and warms it again; completed requests are never repeated.
- The GPU model-only candidate matches the previous graph-placement diagnostic.
  Host orchestration and sampling are allowed; parameter 1 does not mean a
  literal zero-thread process. Diagnostic logging is off for timing.

Primary outcome: server prompt_ms + predicted_ms per request; startup excluded.
Secondary outcomes: prompt/decode latency and throughput separately. Do not
interpret client elapsed or startup as time-to-first-token. No power or peak
memory conclusions are supported. Round timestamps capture temporal context;
round shuffling reduces but cannot eliminate temperature/clock/background drift.

Analysis plan:

1. Verify 2,052 unique identities, balanced rounds/languages/configurations,
   64 prompt_n, 64 predicted_n, cache_n zero, no truncation, and error records.
2. Pair configurations within passage, language and round. Summarize round
   variation and configuration rankings by language. Repetitions are repeated
   measurements, not independent passages.
3. Bootstrap the 19 passage groups as clusters, preserving all their languages,
   configurations and rounds, for 95% confidence intervals. Compare paired
   relative latencies and language-by-configuration contrasts. Use simultaneous
   intervals or multiple-comparison adjustment for exploratory contrast families;
   do not select favorable comparisons after seeing results.
4. Use a predeclared 2% relative-latency band as the practical-equivalence
   criterion (an engineering choice, not a universal statistical threshold).
   A stable language-dependent preference requires a practically meaningful
   language-by-configuration difference with uncertainty excluding zero and
   consistent direction across rounds. Inspect held-out replication before
   proposing a selector.
5. Wide intervals mean inconclusive, not no effect. An absence of meaningful
   language differences is supported only to the precision demonstrated by
   intervals. A selector is not justified solely by differences in sample winners.

Run:

```bash
uv run pdc-research run-controlled --machine configs/machines/berik-full-gpu.json --experiment configs/experiments/rq1-controlled.json --manifest data/workloads/pilot-v3.jsonl --output runs/berik-rq1-controlled-v1
```

Append --resume after interruption. Create RUN/STOP to pause safely. Root
status.json holds aggregate progress. Per-configuration checkpoints and results
are under RUN/configurations/ID/prompts/NNN/results. Round numbers are one-based;
repetition identities are zero-based. Use `uv run pdc-research status --output RUN` to reconstruct committed counts
without modifying results. The saved phase does not prove a process is live.
No automatic shutdown is configured. Keep the machine awake and plugged in.

The historical run is complete. See [current results](current-report.md).
After the repository refactor, new run fingerprints include the shared session
module. Completed historical run metadata remains unchanged and can be analyzed;
source-fingerprint changes intentionally prevent resuming an incompatible old run.
