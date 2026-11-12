# Speaker notes and discussion guide

Suggested delivery: 8–10 minutes for slides 1–11. Slides 12–14 are backup tables.

## Slide 1: Does language change the best
CPU–GPU inference configuration?

Open with the question, not a claim of universal GPU superiority. The short answer is a restricted negative finding at equal token counts. State that this is a progress review and request guidance before expanding the study.

## Slide 2: RQ1 asks whether the winner changes by language

RQ1 is not simply which configuration is fastest or which language takes longer. It is whether language changes configuration preferences after controlling token count. RQ2 is conditional: a useful selector requires a reproducible signal and held-out evaluation.

## Slide 3: A reproducible benchmark on one consumer laptop

All results presented were collected on the initial laptop. Other team machines have not yet supplied a replicated controlled result. The FLORES passages are curated, not randomly representative. This is performance measurement, not an accuracy benchmark.

## Slide 4: Three stages, with a correction in direction

Be candid: stage 1 was useful exploration. Stage 2 partly shifted away from RQ1 into thread tuning and baseline clarification. Stage 3 returned to the required equal-length comparison. Counts exclude setup, warmups and diagnostic runs.

## Slide 5: Stage 1: input size differed substantially by language

Full layer offload won all 80 inputs among the tested placements, but each condition had one repetition. Kazakh inputs averaged about 3.2 times the tokens of English for aligned content. This is consistent with workload-size differences; it does not isolate a language effect or prove novelty.

## Slide 6: Stage 2: thread tuning was not the central RQ1 test

The original full-offload thread sweep gave 1.4744 seconds with one thread and 1.4762 with twelve. Those tiny differences and session/block confounds do not identify a definitive optimum. We found default full layer offload still used CPU for embedding lookup. Explicit CUDA embedding made all logged model graph nodes CUDA in a short diagnostic. CPU orchestration and sampling remain allowed. Thread counts are not physical-core allocation.

## Slide 7: Stage 3 equalized the workload and revisited settings

Exactly 64 input and 64 output tokens. Nineteen of twenty groups were eligible in every language; the excluded group had 56 Chinese tokens. Token prefixes lose exact semantic equivalence. Three rounds revisited every configuration in shuffled order, with requests shuffled within blocks. All 2052 records passed validation with no error records. The 2 percent practical band was specified before execution.

## Slide 8: GPU-dominant settings formed the fastest group

The bars show overall means; ticks are means from the three observed rounds, not confidence intervals. Each overall value averages 228 measurements. GPU-only averaged 1.4330 seconds and full offload with four CPU threads 1.4337: roughly 0.05 percent apart. Do not claim a unique winner. Partial offload was clearly slower in this sample. The 24-layer four-thread setting also shifted across rounds.

## Slide 9: Equal-length inputs had very similar GPU-only timings

These are means under the explicit CUDA-embedding GPU-only model candidate. The vertical axis begins at zero. Configuration winner changes among the full-offload settings were numerically tiny. Raw similarity alone is not the full RQ1 analysis; the next slide discusses the language-by-configuration contrasts and uncertainty.

## Slide 10: No interaction met the practical, stable-effect criteria

Ten thousand bootstrap samples resampled the nineteen passage groups, retaining languages, settings and rounds together. Intervals were adjusted within the complete family of 216 interactions. 201 intervals were within plus/minus two percentage points. Fifteen remained inconclusive for equivalence. One small interaction excluded zero, estimated at 0.655 points, but remained below the practical threshold. No interaction met all criteria. These intervals are conditional on observed rounds and do not estimate future-session variability.

## Slide 11: Supervisor decision: replicate, extend, or revise?

Recommend replication of the controlled protocol on teammates hardware, followed by longer contexts if warranted. Keep machines separate in analysis. If the supervisor instead wants a study of memory-constrained cooperative CPU/GPU computation, revise the proposal explicitly and select an appropriate model/backend. Do not build a language-aware selector merely because RQ2 was proposed. Ask which direction best fits the project scope.

## Slide 12: Backup: controlled rounds side by side

Each cell averages 76 inputs; every input has 64 prompt and 64 generated tokens. A round is a revisit to all nine settings, within one session. Full means, source data and all uncertainty tables are versioned. GPU-only graph placement was verified in a short diagnostic rather than traced during every timing trial.

## Slide 13: Backup: the original 12-setting pilot

Each cell averages 80 aligned-content inputs, with one repetition. The offload setting refers to model layers, not GPU cores. Input lengths differ across languages. These numbers should not be pooled with the controlled-prefix run as if they were the same workload.

## Slide 14: Backup: exploratory thread and baseline repetitions

These are blocked repetitions with unequal input lengths. The ten-thread and CUDA-embedding baseline rows are follow-up sessions. The table documents the investigation but must not be used to infer a causal advantage from tiny cross-session differences.

## Suggested opening statement

We tested whether language changes which CPU–GPU inference configuration is fastest beyond token length. Our initial measurements mixed language with input size. We corrected that with a controlled 64-token experiment. In that restricted setting, we found no stable, practically meaningful language-dependent configuration preference. We would like guidance on replication and extending the workload before evaluating a scheduling policy.

## Questions the supervisor may ask

**Is this a universal negative result?** No. It covers one model, laptop, short input/output length, a curated dev sample and three rounds within one session. Fifteen interaction intervals were inconclusive for the practical-equivalence criterion.

**Did CPU and GPU cooperate?** Layer placement and CPU thread settings were varied. This does not establish an asynchronous cooperative backend. Full offload with CPU embedding was distinguished from the explicit CUDA-embedding candidate.

**Why are semantically aligned passages not enough?** Token counts differ; the controlled prefixes equalize computation length but lose exact semantic equivalence. Both experiments have distinct roles.

**What is novel?** Novelty has not been established. The current contribution is reproducible characterization and a bounded negative finding, pending replication and comparison with prior work.

**What about RQ2?** No selector was implemented or evaluated. The current setting shows no demonstrated language-specific benefit that would justify one.

**Does one thread mean zero CPU use?** No. Parameter one avoids additional OpenMP inference workers; CPU orchestration, sampling and other host work remain allowed. The diagnostic verified model graph placement for a short generation, not every measured request.

**Should the study stop?** Discuss whether to replicate and extend the original question or formally revise the project toward memory-constrained CPU–GPU cooperation. They require different experiments.
