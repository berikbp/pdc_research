# Full-GPU CPU thread sweep

Question: which CPU software thread setting minimizes inference latency when
all model layers are offloaded to the GPU on this machine?

Settings: 1, 2, 4, 6, 8, 12, 16 threads; GPU layers 99; same prefill/decode
thread count; original 80 aligned inputs; 64 output tokens; three repetitions.
Total: 1,680 measured requests. No CPU affinity is imposed; this is not a
physical-core allocation experiment. Repetitions measure timing variability,
not output diversity (greedy decoding remains enabled).

Configuration-first execution loads each configuration once on an uninterrupted
run, plus a separate tokenizer setup load. Configurations, prompt groups, and
trials within groups are shuffled reproducibly. Configuration blocks can still
be affected by thermal/time drift; repetitions within a block do not eliminate
that confound. Model startup is recorded separately and excluded from inference.
Every successful request is committed immediately. Prompt summaries remain
partial until all seven configurations have visited each prompt.

Run:

```bash
uv run pdc-research run --machine configs/machines/berik-full-gpu.json --experiment configs/experiments/cpu-threads.json --manifest data/workloads/pilot-v3.jsonl --output runs/berik-cpu-threads-v1
```

Resume with the same command plus `--resume`. Create
`runs/berik-cpu-threads-v1/STOP` for a graceful pause. Configuration-first mode
rejects `--stop-after-prompts`. The old pilot remains unchanged in its own run
folder and should be analyzed separately. No automatic shutdown is configured.
