"""Draw the report figures as standalone SVG files; run from the repo root.

Reads the versioned result tables under docs/results and writes docs/results/figures.
Uses only the standard library, like the rest of the analysis.
"""

import csv
import json
from pathlib import Path

RESULTS = Path("docs/results")
FIGURES = RESULTS / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # first three categorical slots
WIDTH, HEIGHT = 820, 440
LEFT, RIGHT, TOP, BOTTOM = 64, 250, 56, 52
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ticks(top: float, count: int = 5) -> list[float]:
    raw = top / count
    magnitude = 10 ** len(str(int(raw))) / 10
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [step * i for i in range(int(top / step) + 2) if step * i <= top + step]


class Chart:
    """Line chart with a zero-based y axis, direct end labels and a legend."""

    def __init__(self, title: str, subtitle: str, x_label: str, y_label: str,
                 x_values: list[float], y_max: float, x_names: list[str] | None = None):
        self.x_values, self.x_names = x_values, x_names
        self.y_ticks = ticks(y_max)
        self.y_top = self.y_ticks[-1]
        self.parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {WIDTH} {HEIGHT}' width='{WIDTH}' "
            f"height='{HEIGHT}' role='img' aria-label='{title}'>",
            f"<rect width='{WIDTH}' height='{HEIGHT}' fill='{SURFACE}'/>",
            f"<text x='{LEFT}' y='24' {FONT} font-size='15' font-weight='600' fill='{INK}'>{title}</text>",
            f"<text x='{LEFT}' y='42' {FONT} font-size='12' fill='{MUTED}'>{subtitle}</text>"]
        for value in self.y_ticks:
            y = self.y(value)
            self.parts.append(f"<line x1='{LEFT}' x2='{WIDTH - RIGHT}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>")
            self.parts.append(f"<text x='{LEFT - 8}' y='{y + 4:.1f}' {FONT} font-size='11' fill='{MUTED}' "
                              f"text-anchor='end'>{value:g}</text>")
        positions = x_values if x_names is None else list(range(len(x_names)))
        labels = [f"{v:g}" for v in x_values] if x_names is None else x_names
        for position, label in zip(positions, labels):
            self.parts.append(f"<text x='{self.x(position):.1f}' y='{HEIGHT - BOTTOM + 18}' {FONT} font-size='11' "
                              f"fill='{MUTED}' text-anchor='middle'>{label}</text>")
        self.parts.append(f"<text x='{(LEFT + WIDTH - RIGHT) / 2}' y='{HEIGHT - 12}' {FONT} font-size='12' "
                          f"fill='{MUTED}' text-anchor='middle'>{x_label}</text>")
        self.parts.append(f"<text x='14' y='{(TOP + HEIGHT - BOTTOM) / 2}' {FONT} font-size='12' fill='{MUTED}' "
                          f"text-anchor='middle' transform='rotate(-90 14 {(TOP + HEIGHT - BOTTOM) / 2})'>"
                          f"{y_label}</text>")
        self.legend_y = TOP + 6

    def x(self, value: float) -> float:
        if self.x_names is not None:
            low, high = -0.35, len(self.x_names) - 0.65
        else:
            low, high = min(self.x_values), max(self.x_values)
        return LEFT + (value - low) / (high - low) * (WIDTH - LEFT - RIGHT)

    def y(self, value: float) -> float:
        return HEIGHT - BOTTOM - value / self.y_top * (HEIGHT - TOP - BOTTOM)

    def series(self, name: str, points: list[tuple[float, float]], color: str, *, dashed: bool = False,
               markers: bool = True, line: bool = True, legend: bool = True,
               end_label: str | None = None) -> None:
        path = " ".join(f"{'M' if i == 0 else 'L'}{self.x(px):.1f},{self.y(py):.1f}"
                        for i, (px, py) in enumerate(points))
        dash = " stroke-dasharray='6 4'" if dashed else ""
        if line:
            self.parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'{dash}/>")
        if markers:
            for px, py in points:
                fill = SURFACE if dashed else color
                self.parts.append(f"<circle cx='{self.x(px):.1f}' cy='{self.y(py):.1f}' r='4' fill='{fill}' "
                                  f"stroke='{color}' stroke-width='2'><title>{name}: {py:.1f}</title></circle>")
        if end_label:
            px, py = points[-1]
            self.parts.append(f"<text x='{self.x(px) + 9:.1f}' y='{self.y(py) + 4:.1f}' {FONT} font-size='11' "
                              f"fill='{INK}'>{end_label}</text>")
        if not legend:
            return
        x = WIDTH - RIGHT + 96
        self.parts.append(f"<line x1='{x}' x2='{x + 22}' y1='{self.legend_y}' y2='{self.legend_y}' "
                          f"stroke='{color}' stroke-width='2'{dash}/>")
        self.parts.append(f"<text x='{x + 28}' y='{self.legend_y + 4}' {FONT} font-size='11' "
                          f"fill='{INK}'>{name}</text>")
        self.legend_y += 18

    def note(self, x: float, y: float, text: str, *, below: bool = False) -> None:
        offset = 18 if below else -10
        self.parts.append(f"<text x='{self.x(x):.1f}' y='{self.y(y) + offset:.1f}' {FONT} font-size='10' "
                          f"fill='{MUTED}' text-anchor='middle'>{text}</text>")

    def write(self, path: Path) -> None:
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


def block_cost() -> None:
    chart = Chart("Decode time grows linearly with blocks left on the CPU",
                  "One request at a time, 64 generated tokens; lines are the fitted cost model",
                  "Transformer blocks on the CPU", "Decode time, ms per token", [0, 12, 24, 36, 48], 190)
    for name, folder, color in (("Qwen2.5-14B", "layer-sweep-14b-v1", ORANGE), ("Qwen2.5-7B", "layer-sweep-7b-v1", BLUE)):
        rows = [r for r in read_csv(RESULTS / folder / "layers.csv") if not r["server_args"]]
        model = json.loads((RESULTS / folder / "cost-model.json").read_text())["decode_ms_per_token"]
        slope, top = model["per_cpu_block_ms"], max(int(r["blocks_on_cpu"]) for r in rows)
        start = model["base_ms"] + model["handoff_ms"]
        chart.series(f"{name} fit", [(0, start), (top, start + slope * top)], color, markers=False,
                     legend=False, end_label=f"+{slope:.2f} ms per block")
        measured = sorted({(int(r["blocks_on_cpu"]), float(r["decode_ms_per_token"])) for r in rows
                           if r["output_on_gpu"] == "True"})
        chart.series(name, measured, color, line=False)
    chart.write(FIGURES / "block-cost.svg")


def throughput(cells: list[dict], cell: str, language: str) -> float:
    return next(float(r["generated_tokens_per_s"]) for r in cells if r["cell"] == cell and r["language"] == language)


def parallel_7b() -> None:
    cells = read_csv(RESULTS / "parallel-7b-v1" / "cells.csv")
    slots = [1, 4, 16]
    chart = Chart("Under load, partial offload falls further behind full offload",
                  "Qwen2.5-7B, closed loop of concurrent requests, 64 generated tokens each",
                  "Concurrent requests (server slots)", "Generated tokens per second",
                  slots, 250, [str(s) for s in slots])
    for layers, placement, dashed in ((29, "all on GPU", False), (24, "5 blocks on CPU", True)):
        for language, short, color in (("eng_Latn", "English", BLUE), ("kaz_Cyrl", "Kazakh", ORANGE)):
            points = [(i, throughput(cells, f"g{layers}-np{s:02d}", language)) for i, s in enumerate(slots)]
            chart.series(f"{short}, {placement}", points, color, dashed=dashed, end_label=f"{points[-1][1]:.0f}")
    chart.write(FIGURES / "parallel-7b.svg")


def parallel_14b() -> None:
    cells = read_csv(RESULTS / "parallel-14b-v1" / "cells.csv")
    slots = [1, 4, 16]
    chart = Chart("On a model that does not fit, longer inputs push layers off the GPU",
                  "Qwen2.5-14B; numbers at the points are layers that still fit on the GPU (of 49)",
                  "Concurrent requests (server slots)", "Generated tokens per second",
                  slots, 50, [str(s) for s in slots])
    plans = (("English, 256-token slots", "c256", "eng_Latn", BLUE, False),
             ("English, 512-token slots", "c512", "eng_Latn", BLUE, True),
             ("Kazakh, 512-token slots", "c512", "kaz_Cyrl", ORANGE, False))
    for name, prefix, language, color, dashed in plans:
        points = []
        for index, count in enumerate(slots):
            row = next(r for r in cells if r["cell"].startswith(f"{prefix}-np{count:02d}") and r["language"] == language)
            points.append((index, float(row["generated_tokens_per_s"])))
            if not dashed:
                chart.note(index, points[-1][1], row["gpu_layers"], below=language == "kaz_Cyrl")
        chart.series(name, points, color, dashed=dashed, end_label=f"{points[-1][1]:.0f}")
    chart.write(FIGURES / "parallel-14b.svg")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    block_cost()
    parallel_7b()
    parallel_14b()


if __name__ == "__main__":
    main()
