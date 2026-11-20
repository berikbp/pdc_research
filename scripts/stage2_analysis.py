"""Stage 2 summary tables across machines, models and rounds; run from the repo root.

Inputs (versioned under docs/results):
  serving-*/rounds.csv          per-round blocks written by `analyze-serving`
  stage2/journal-blocks.csv     per-block journal windows from scripts/align_journals.py
Outputs (docs/results/stage2):
  ratios.csv        NEO (measured launch time) / GPU-only throughput, per round, cell and language
  crossovers.csv    smallest load where NEO wins in every round, per setup, memory and language
  journals.csv      CPU memory traffic, CPU power, GPU clocks per setup and NEO mode
  energy.csv        energy per generated token (GPU + CPU package), NEO against GPU-only
  latency.csv       time to first token and time per output token at the crossover cells
  summary.json      headline numbers quoted in docs/stage2-report.md

A third round was measured as a separate run (`*-r3`, a new seed) for the cells next to the
crossover; its blocks count as round 3 of the main run.
"""

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path("docs/results")
OUT = RESULTS / "stage2"
SETUPS = [  # (setup id, label, main run, third-round run, journal pc, journal model)
    ("pc2-llama2", "RTX 4080 + Llama-2-7B", "serving-llama2-pc2-v3", "serving-llama2-pc2-v3-r3", "pc2", "llama2"),
    ("pc1-llama3", "RTX 5090 + Llama-3-8B", "serving-llama3-pc1-v1", "serving-llama3-pc1-v1-r3", "pc1", "llama3"),
    ("pc1-llama2", "RTX 5090 + Llama-2-7B", "serving-llama2-pc1-v3", None, "pc1", "llama2"),
]
LANGUAGES = ["eng_Latn", "rus_Cyrl", "zho_Hans", "kaz_Cyrl", "kaz_Cyrl@eng_len"]
CLEAR = 0.05   # a win or loss must exceed 5% in every round; smaller differences count as ties


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def blocks_of(setup) -> list[dict]:
    _, _, main, third, _, _ = setup
    rows = read(RESULTS / main / "rounds.csv")
    if third:
        for row in read(RESULTS / third / "rounds.csv"):
            rows.append({**row, "round": "3"})
    return rows


def verdict(ratios: list[float]) -> str:
    if all(r > 1 + CLEAR for r in ratios):
        return "win"
    if all(r < 1 - CLEAR for r in ratios):
        return "loss"
    return "tie"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ratio_rows, crossover_rows, latency_rows = [], [], []
    summary = {"clear_threshold": CLEAR, "setups": {}}
    for setup in SETUPS:
        sid, label = setup[0], setup[1]
        by = {(r["system"], r["memory"], int(r["load"]), r["language"], r["round"]): r for r in blocks_of(setup)}
        inputs = {lang: st.mean(float(r["mean_input_tokens"]) for k, r in by.items() if k[3] == lang) for lang in LANGUAGES}
        memories = sorted({k[1] for k in by if k[0] == "neo_tuned"}, key=lambda m: (m == "free", m))
        loads = sorted({k[2] for k in by})
        cells = defaultdict(list)
        for (system, memory, load, lang, rnd), row in by.items():
            if system != "neo_tuned" or ("neo_base", memory, load, lang, rnd) not in by:
                continue
            base = by[("neo_base", memory, load, lang, rnd)]
            ratio = float(row["generated_tokens_per_s"]) / float(base["generated_tokens_per_s"])
            cells[(memory, load, lang)].append(ratio)
            ratio_rows.append({"setup": sid, "memory": memory, "load": load, "language": lang, "round": rnd,
                               "neo_tok_s": round(float(row["generated_tokens_per_s"]), 2),
                               "gpu_only_tok_s": round(float(base["generated_tokens_per_s"]), 2),
                               "ratio": round(ratio, 4), "cpu_decode_share": round(float(row["cpu_decode_share"] or 0), 3),
                               "gpu_only_swapped_out": base["swap_out_requests"]})
        setup_summary = {"label": label, "input_tokens": {k: round(v, 1) for k, v in inputs.items()}, "memory": {}}
        for memory in memories:
            per_lang = {}
            for lang in LANGUAGES:
                series = {load: cells.get((memory, load, lang), []) for load in loads}
                first = next((load for load in loads if series[load] and verdict(series[load]) == "win"
                              and all(series[l] and verdict(series[l]) == "win" for l in loads if l > load)), None)
                best = max((st.mean(v), load) for load, v in series.items() if v)
                worst = min((st.mean(v), load) for load, v in series.items() if v)
                crossover_rows.append({
                    "setup": sid, "memory": memory, "language": lang, "input_tokens": round(inputs[lang], 1),
                    "first_winning_load": first if first is not None else "none",
                    "rounds_at_crossover": len(series.get(first, [])) if first else "",
                    "ratio_at_crossover": round(st.mean(series[first]), 3) if first else "",
                    "best_ratio": round(best[0], 3), "best_load": best[1],
                    "worst_ratio": round(worst[0], 3), "worst_load": worst[1],
                    **{f"ratio_c{load}": round(st.mean(series[load]), 3) if series[load] else "" for load in loads}})
                per_lang[lang] = {"first_winning_load": first, "best_ratio": round(best[0], 3),
                                  "worst_ratio": round(worst[0], 3)}
            setup_summary["memory"][memory] = per_lang
        # NEO as shipped (launch time 0.8 ms): best gain anywhere
        shipped = [float(r["generated_tokens_per_s"]) / float(by[("neo_base",) + k[1:]]["generated_tokens_per_s"])
                   for k, r in by.items() if k[0] == "neo" and ("neo_base",) + k[1:] in by]
        setup_summary["neo_as_shipped"] = {"blocks": len(shipped), "best_ratio": round(max(shipped), 3),
                                           "median_ratio": round(st.median(shipped), 3)}
        # Latency at 64 blocks, 32 clients: what the throughput gain costs
        for lang in ("eng_Latn", "kaz_Cyrl"):
            for system in ("neo_base", "neo_tuned"):
                rows = [r for k, r in by.items() if k[:4] == (system, "b0064", 32, lang)]
                if rows:
                    latency_rows.append({
                        "setup": sid, "memory": "b0064", "load": 32, "language": lang, "system": system,
                        "tok_s": round(st.mean(float(r["generated_tokens_per_s"]) for r in rows), 1),
                        "mean_latency_s": round(st.mean(float(r["mean_latency_s"]) for r in rows), 2),
                        "p95_latency_s": round(st.mean(float(r["p95_latency_s"]) for r in rows), 2),
                        "ttft_s": round(st.mean(float(r["mean_ttft_s"]) for r in rows), 2),
                        "time_per_output_token_ms": round(st.mean(float(r["mean_time_per_output_token_ms"]) for r in rows), 1)})
        summary["setups"][sid] = setup_summary

    # Journals: CPU memory traffic and power, GPU clocks and throttling
    journal = read(OUT / "journal-blocks.csv")
    journal_rows, energy_rows = [], []
    for sid, label, _, _, pc, model in SETUPS:
        rows = [r for r in journal if r["pc"] == pc and r["model"] == model]
        for system in ("neo_base", "neo_tuned"):
            for band, test in (("no CPU decode", lambda s: s == 0), ("CPU decode share > 0.4", lambda s: s > 0.4)):
                sel = [r for r in rows if r["system"] == system and test(float(r["cpu_share"] or 0))]
                if not sel:
                    continue
                f = lambda key: st.mean(float(r[key]) for r in sel)  # noqa: E731
                journal_rows.append({
                    "setup": sid, "system": system, "blocks_in": band, "n_blocks": len(sel),
                    "cpu_read_gb_s": round(f("read_gbs"), 2), "cpu_read_gb_s_max": round(max(float(r["read_gbs"]) for r in sel), 2),
                    "predicted_kv_read_gb_s": round(f("kv_read_gbs"), 2), "cpu_package_w": round(f("cpu_w"), 1),
                    "gpu_w": round(f("gpu_w"), 1), "sm_clock_mhz": round(f("pclk")), "sm_busy_pct": round(f("sm"), 1),
                    "power_violation_pct": round(f("pviol"), 2), "thermal_events": sum(float(r["tviol"]) > 0 for r in sel),
                    "pcie_mb_s": round(f("rx") + f("tx"))})
        index = {(r["system"], r["blocks"], r["load"], r["lang"], r["round"]): r for r in rows}
        pairs = []
        for (system, *rest), r in index.items():
            b = index.get(("neo_base", *rest))
            if system != "neo_tuned" or not b:
                continue
            energy = lambda x: (float(x["gpu_w"]) + float(x["cpu_w"])) * float(x["wall"]) / float(x["tokens"])  # noqa: E731
            pairs.append((float(r["tok_s"]) / float(b["tok_s"]), energy(r) / energy(b), energy(r), energy(b)))
        for band, test in (("NEO faster by > 10%", lambda x: x > 1.1), ("within 10%", lambda x: 0.9 <= x <= 1.1),
                           ("NEO slower by > 10%", lambda x: x < 0.9)):
            sel = [p for p in pairs if test(p[0])]
            if sel:
                energy_rows.append({"setup": sid, "blocks_in": band, "n_blocks": len(sel),
                                    "throughput_ratio": round(st.mean(p[0] for p in sel), 3),
                                    "energy_per_token_ratio": round(st.mean(p[1] for p in sel), 3),
                                    "neo_mj_per_token": round(1000 * st.mean(p[2] for p in sel), 1),
                                    "gpu_only_mj_per_token": round(1000 * st.mean(p[3] for p in sel), 1)})
    for name, rows in (("ratios.csv", ratio_rows), ("crossovers.csv", crossover_rows), ("journals.csv", journal_rows),
                       ("energy.csv", energy_rows), ("latency.csv", latency_rows)):
        write(name, rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"ratios {len(ratio_rows)}, crossovers {len(crossover_rows)}, journals {len(journal_rows)}, "
          f"energy {len(energy_rows)}, latency {len(latency_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
