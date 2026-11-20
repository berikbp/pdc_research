"""Draw the stage 2 figures as standalone SVG files; run from the repo root after stage2_analysis.py.

Standard library only. Colours follow the project palette (categorical slots in fixed order),
with a light and a dark variant switched by prefers-color-scheme inside each SVG.
Every marker carries a <title>, so values show on hover when the SVG is opened directly.
"""

import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path("docs/results")
FIGURES = RESULTS / "figures"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LANGUAGES = [  # fixed order = fixed colour slot; the length-matched control also gets a dashed line
    ("eng_Latn", "English", "s1", ""), ("rus_Cyrl", "Russian", "s2", ""), ("zho_Hans", "Chinese", "s4", ""),
    ("kaz_Cyrl", "Kazakh", "s3", ""), ("kaz_Cyrl@eng_len", "Kazakh cut to English length", "s5", "6 4")]
SETUPS = [("pc2-llama2", "RTX 4080 + Llama-2-7B"), ("pc1-llama3", "RTX 5090 + Llama-3-8B"),
          ("pc1-llama2", "RTX 5090 + Llama-2-7B")]
STYLE = """<style>
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .muted{fill:#52514e} .grid{stroke:#e4e3df} .axis{stroke:#8a8984}
  .ref{stroke:#52514e} .ring{stroke:#fcfcfb}
  .s1{stroke:#2a78d6;fill:#2a78d6} .s2{stroke:#eb6834;fill:#eb6834} .s3{stroke:#1baf7a;fill:#1baf7a}
  .s4{stroke:#eda100;fill:#eda100} .s5{stroke:#e87ba4;fill:#e87ba4}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .muted{fill:#c3c2b7} .grid{stroke:#383835} .axis{stroke:#6b6a64}
    .ref{stroke:#c3c2b7} .ring{stroke:#1a1a19}
    .s1{stroke:#3987e5;fill:#3987e5} .s2{stroke:#d95926;fill:#d95926} .s3{stroke:#199e70;fill:#199e70}
    .s4{stroke:#c98500;fill:#c98500} .s5{stroke:#d55181;fill:#d55181}
  }
  text{font-family:""" + FONT + """}
</style>"""


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def svg(width: int, height: int, title: str, body: list[str]) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' "
            f"height='{height}' role='img' aria-label='{title}'>{STYLE}<title>{title}</title>"
            f"<rect class='bg' width='{width}' height='{height}'/>" + "".join(body) + "</svg>")


def text(x, y, value, cls="ink", size=12, anchor="start", weight="normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' class='{cls}' font-size='{size}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{value}</text>")


def crossover() -> None:
    """NEO / GPU-only throughput against load: setups in columns, KV budget in rows."""
    rows = read(RESULTS / "stage2" / "ratios.csv")
    data = defaultdict(list)
    for r in rows:
        data[(r["setup"], r["memory"], int(r["load"]), r["language"])].append(float(r["ratio"]))
    loads = [1, 4, 8, 16, 32, 64]
    memories = [("b0064", "64 blocks of KV cache (1 024 tokens)"), ("b0096", "96 blocks (1 536 tokens)")]
    pw, ph, gap_x, gap_y = 268, 210, 44, 78
    left, top = 58, 150
    width = left + 3 * pw + 2 * gap_x + 24
    height = top + 2 * ph + gap_y + 64
    y_min, y_max = 0.5, 2.1
    body = [text(left, 30, "Where NEO overtakes GPU-only", size=18, weight="600"),
            text(left, 52, "Throughput of NEO with its CPU launch time measured on the machine, divided by the same "
                           "engine run GPU-only at the same KV budget.", "muted", 12),
            text(left, 68, "Mean of all rounds (3 rounds at the crossover cells, otherwise 2); above the line NEO is "
                           "faster. Closed loop of N clients, 128 output tokens.", "muted", 12)]
    # legend, one row
    lx = left
    for _, name, cls, dash in LANGUAGES:
        body.append(f"<line x1='{lx}' y1='98' x2='{lx + 22}' y2='98' class='{cls}' stroke-width='2' "
                    f"stroke-dasharray='{dash}'/><circle cx='{lx + 11}' cy='98' r='4' class='{cls} ring' "
                    f"style='stroke-width:2'/>")
        body.append(text(lx + 28, 102, name, "ink", 12))
        lx += 34 + 7.0 * len(name) + 16
    for row, (memory, memory_label) in enumerate(memories):
        for col, (setup, setup_label) in enumerate(SETUPS):
            x0 = left + col * (pw + gap_x)
            y0 = top + row * (ph + gap_y)
            px = lambda i: x0 + 10 + i * (pw - 20) / (len(loads) - 1)  # noqa: E731
            py = lambda v: y0 + ph - (v - y_min) / (y_max - y_min) * ph  # noqa: E731
            body.append(text(x0, y0 - 24, setup_label, "ink", 13, weight="600"))
            body.append(text(x0, y0 - 9, memory_label, "muted", 11))
            for v in (0.5, 1.0, 1.5, 2.0):
                body.append(f"<line x1='{x0}' x2='{x0 + pw}' y1='{py(v):.1f}' y2='{py(v):.1f}' class='grid'/>")
                if col == 0:
                    body.append(text(x0 - 8, py(v) + 4, f"{v:.1f}×", "muted", 11, "end"))
            body.append(f"<line x1='{x0}' x2='{x0 + pw}' y1='{py(1):.1f}' y2='{py(1):.1f}' class='ref' "
                        f"stroke-width='1.2' stroke-dasharray='4 3'/>")
            if col == 2:
                body.append(text(x0 + pw - 2, py(1) - 6, "GPU-only", "muted", 11, "end"))
            for i, load in enumerate(loads):
                body.append(text(px(i), y0 + ph + 18, str(load), "muted", 11, "middle"))
            if row == 1:
                body.append(text(x0 + pw / 2, y0 + ph + 36, "concurrent clients", "muted", 11, "middle"))
            for lang, name, cls, dash in LANGUAGES:
                points = [(i, sum(v) / len(v), len(v)) for i, load in enumerate(loads)
                          if (v := data.get((setup, memory, load, lang)))]
                if not points:
                    continue
                path = " ".join(f"{'M' if k == 0 else 'L'}{px(i):.1f},{py(v):.1f}" for k, (i, v, _) in enumerate(points))
                body.append(f"<path d='{path}' class='{cls}' style='fill:none' stroke-width='2' stroke-dasharray='{dash}' "
                            f"stroke-linejoin='round'/>")
                for i, v, n in points:
                    body.append(f"<circle cx='{px(i):.1f}' cy='{py(v):.1f}' r='4' class='{cls} ring' "
                                f"style='stroke-width:2'><title>{name}, {loads[i]} clients: {v:.2f}× "
                                f"({n} rounds)</title></circle>")
    (FIGURES / "stage2-crossover.svg").write_text(svg(width, height, "Where NEO overtakes GPU-only", body),
                                                  encoding="utf-8")


def memory_traffic() -> None:
    """Measured CPU memory reads against the KV-cache bytes CPU attention must read."""
    rows = [r for r in read(RESULTS / "stage2" / "journal-blocks.csv")
            if r["system"] == "neo_tuned" and float(r["cpu_share"] or 0) > 0.05]
    series = [("pc2", "llama2", "4080 + Llama-2 (i9-13900K)", "s1"),
              ("pc1", "llama3", "5090 + Llama-3 (Ultra 9 285K)", "s2"),
              ("pc1", "llama2", "5090 + Llama-2 (Ultra 9 285K)", "s3")]
    left, top, pw, ph = 70, 116, 560, 360
    width, height = left + pw + 250, top + ph + 70
    top_v = 40.0
    px = lambda v: left + v / top_v * pw  # noqa: E731
    py = lambda v: top + ph - v / top_v * ph  # noqa: E731
    body = [text(left, 30, "CPU attention reads the KV cache it needs, and no more", size=18, weight="600"),
            text(left, 52, "One dot per measured language block in which NEO decoded on the CPU. x: KV-cache bytes "
                           "the CPU must read", "muted", 12),
            text(left, 68, "(CPU-decoded tokens/s × context × KV bytes per token). "
                           "y: reads counted by the memory controllers during the block.", "muted", 12),
            text(left, 84, "Peak memory bandwidth was not measured; a single-thread copy on PC1 moved 49 GB/s "
                           "(reads + writes).", "muted", 12)]

    for v in range(0, 41, 10):
        body.append(f"<line x1='{left}' x2='{left + pw}' y1='{py(v):.1f}' y2='{py(v):.1f}' class='grid'/>")
        body.append(text(left - 8, py(v) + 4, f"{v}", "muted", 11, "end"))
        body.append(f"<line y1='{top}' y2='{top + ph}' x1='{px(v):.1f}' x2='{px(v):.1f}' class='grid'/>")
        body.append(text(px(v), top + ph + 18, f"{v}", "muted", 11, "middle"))
    body.append(text(left + pw / 2, top + ph + 40, "expected KV-cache reads, GB/s", "muted", 12, "middle"))
    body.append(f"<text transform='translate({left - 46},{top + ph / 2}) rotate(-90)' class='muted' font-size='12' "
                f"text-anchor='middle'>measured memory reads, GB/s</text>")
    body.append(f"<line x1='{px(0)}' y1='{py(0)}' x2='{px(top_v)}' y2='{py(top_v)}' class='ref' stroke-width='1.2' "
                f"stroke-dasharray='4 3'/>")
    body.append(text(px(38), py(38) - 8, "measured = expected", "muted", 11, "end"))
    for pc, model, label, cls in series:
        for r in rows:
            if r["pc"] != pc or r["model"] != model:
                continue
            x, y = float(r["kv_read_gbs"]), float(r["read_gbs"])
            body.append(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='4' class='{cls} ring' style='stroke-width:1.5' "
                        f"fill-opacity='0.85'><title>{label}: {r['blocks']} blocks, {r['load']} clients, {r['lang']}, "
                        f"round {r['round']}: expected {x:.1f}, measured {y:.1f} GB/s</title></circle>")
    ly = top + 8
    for _, _, label, cls in series:
        body.append(f"<circle cx='{left + pw + 30}' cy='{ly}' r='5' class='{cls} ring' style='stroke-width:1.5'/>")
        body.append(text(left + pw + 42, ly + 4, label, "ink", 12))
        ly += 24
    (FIGURES / "stage2-memory-traffic.svg").write_text(
        svg(width, height, "CPU memory traffic against expected KV-cache reads", body), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    crossover()
    memory_traffic()
    print("wrote", FIGURES / "stage2-crossover.svg", "and", FIGURES / "stage2-memory-traffic.svg")


if __name__ == "__main__":
    main()
