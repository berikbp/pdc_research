"""Construct aligned groups without downloading or silently replacing source data."""

import random
from collections.abc import Callable
from pathlib import Path

from .common import append_row, sha256, write_json


def prepare(experiment: dict, source: str | Path, output: str | Path) -> None:
    languages = experiment["languages"]
    if len(languages) != len(set(languages)) or not languages:
        raise ValueError("languages must be nonempty and unique")
    sources = {lang: Path(source) / experiment["split"] / f"{lang}.{experiment['split']}"
               for lang in languages}
    texts = {lang: path.read_text(encoding="utf-8").splitlines() for lang, path in sources.items()}
    sizes = {len(lines) for lines in texts.values()}
    if len(sizes) != 1 or not all(all(line.strip() for line in lines) for lines in texts.values()):
        raise ValueError("FLORES files must have equal line counts and no empty sentences")
    width, count = experiment["sentences_per_group"], experiment["groups"]
    if width < 1 or count < 1:
        raise ValueError("groups and sentences_per_group must be positive")
    # Consecutive blocks preserve local order; random sample selects identical IDs for every language.
    blocks = list(range(0, next(iter(sizes)) - width + 1, width))
    if count > len(blocks):
        raise ValueError(f"Requested {count} non-overlapping groups; only {len(blocks)} available")
    starts = random.Random(experiment["seed"]).sample(blocks, count)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as stream:
        for start in starts:
            ids = list(range(start + 1, start + width + 1))
            for lang in languages:
                append_row(stream, {"group_id": f"{experiment['split']}-{start + 1:04d}",
                                    "sentence_ids": ids, "language": lang, "split": experiment["split"],
                                    "text": "\n".join(texts[lang][start:start + width])})
    write_json(str(output) + ".meta.json", {"experiment": experiment,
               "source_sha256": {lang: sha256(path) for lang, path in sources.items()},
               "manifest_sha256": sha256(output), "rows": count * len(languages)})


def make_cases(rows: list[dict], experiment: dict, tokenize: Callable[[str], list[int]]) -> list[dict]:
    expected = set(experiment["languages"])
    groups = {}
    for row in rows:
        members = groups.setdefault(row["group_id"], set())
        if row["language"] in members:
            raise ValueError("Duplicate language within a workload group")
        members.add(row["language"])
    if not groups or any(members != expected for members in groups.values()):
        raise ValueError("Every workload group must contain exactly the configured languages")
    cases = []
    for row in rows:
        tokens = tokenize(row["text"])
        if not tokens:
            raise ValueError("Tokenizer returned no tokens")
        common = {**row, "source_token_count": len(tokens)}
        cases.append({**common, "mode": "aligned", "tokens": tokens})
        for length in experiment["matched_tokens"]:
            if length < 1:
                raise ValueError("matched_tokens must be positive")
            # Prefix clipping changes semantics. Never pad or repeat short inputs.
            if len(tokens) >= length:
                cases.append({**common, "mode": f"matched-{length}", "tokens": tokens[:length]})
    # Keep only groups eligible in every language at a given matched length.
    languages = set(experiment["languages"])
    eligible = {}
    for case in cases:
        key = (case["group_id"], case["mode"])
        eligible.setdefault(key, set()).add(case["language"])
    return [case for case in cases if eligible[(case["group_id"], case["mode"])] == languages]
