# Supervisor presentation package

Prepared 26 October 2026 from the versioned report tables. Suggested presentation
length: 8–10 minutes for slides 1–11; slides 12–14 are backup tables.

## Files

- [Editable PowerPoint](supervisor-presentation.pptx): 14 slides with embedded
  speaker notes. Text and tables are editable; the two scientific plots are
  embedded images, with standalone PNG/SVG versions in `figures/`.
- [Slide PDF](supervisor-presentation.pdf): matching fixed-layout version.
- [One-page PDF summary](supervisor-summary.pdf): supervisor handout.
- [Summary text](supervisor-summary.md): editable plain-text version.
- [Speaker notes and discussion guide](speaker-notes.md): slide-by-slide notes,
  an opening statement and likely supervisor questions.
- [Full current report](../current-report.md): detailed history and limitations.

## Presentation message

At 64 input and 64 output tokens, no stable, practically meaningful
language-dependent configuration preference was detected among the nine tested
settings on the initial laptop/model. This is a restricted negative RQ1 finding,
not universal equivalence. Fifteen interaction intervals remain inconclusive for
practical equivalence; RQ2 has not been evaluated. A small statistically detectable
interaction did not meet the practical threshold.

Be explicit that the thread investigation partly drifted from RQ1, that default
full offload left the embedding on CPU, and that the later CUDA-embedding baseline
permits CPU orchestration. Do not call any of these a literal zero-host-thread run.

Ask the supervisor whether to prioritize replication and longer contexts under
the original proposal, or to formally revise the study toward memory-constrained
CPU–GPU cooperation. No further benchmark is part of this presentation build.

## Rebuild

The benchmark has no new runtime dependencies. The document build uses optional
packages in an isolated environment and expects DejaVu Sans fonts under
`/usr/share/fonts/truetype/dejavu` (Linux/WSL):

```bash
uv run --no-project --python 3.12 --with python-pptx==1.0.2 --with reportlab==5.1.0 --with matplotlib==3.11.2 python scripts/build_presentation.py
```

Inputs are `docs/results/history/exploratory-configurations.csv`, the controlled
configuration/language tables and `validation.json`. Slide text is frozen to the
current report; review it when updating the result tables. The builder checks
text-box height and the expected controlled measurement count.

The PDF was inspected as 14 pages and the handout as one page. Numerical cells in
the PowerPoint backup tables were checked against the source CSVs. Plotted ticks
show round means, not confidence intervals. Bootstrap details are in the report
and speaker notes. The charts are descriptions of means; uncertainty is discussed
separately rather than invented as graphical error bars.
