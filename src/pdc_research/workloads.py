"""Construct aligned groups without downloading or silently replacing source data."""

import csv
import random
from collections.abc import Callable
from pathlib import Path

from .common import append_row, read_json, sha256, write_json


def _select_groups(experiment: dict, source: Path, size: int) -> tuple[list[dict], dict]:
    width, count = experiment["sentences_per_group"], experiment["groups"]
    if not experiment.get("selection_file"):
        blocks = list(range(0, size - width + 1, width))
        if count > len(blocks):
            raise ValueError(f"Requested {count} non-overlapping groups; only {len(blocks)} available")
        starts = random.Random(experiment["seed"]).sample(blocks, count)
        return [{"sentence_ids": list(range(start + 1, start + width + 1))} for start in starts], {}

    selection_path = Path(experiment["selection_file"])
    selection = read_json(selection_path)
    if selection["split"] != experiment["split"] or len(selection["groups"]) != count:
        raise ValueError("Frozen selection must match the experiment split and group count")
    metadata_path = source / f"metadata_{experiment['split']}.tsv"
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    if len(metadata) != size:
        raise ValueError("FLORES metadata and sentence files must have equal row counts")
    used = set()
    articles = set()
    for group in selection["groups"]:
        ids = group["sentence_ids"]
        if (len(ids) != width or any(type(index) is not int or not 1 <= index <= size for index in ids)
                or ids != list(range(ids[0], ids[0] + width))):
            raise ValueError("Each selected passage must contain consecutive in-range sentence IDs")
        if used.intersection(ids):
            raise ValueError("Selected passages must not overlap")
        used.update(ids)
        urls = {metadata[index - 1]["URL"] for index in ids}
        if len(urls) != 1 or group["source_url"] not in urls:
            raise ValueError("Selected passage crosses an article boundary or has the wrong source URL")
        if group["source_url"] in articles:
            raise ValueError("Select one passage per source article")
        articles.add(group["source_url"])
    return selection["groups"], {
        "selection_sha256": sha256(selection_path), "metadata_sha256": sha256(metadata_path),
    }


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
    groups, provenance = _select_groups(experiment, Path(source), next(iter(sizes)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as stream:
        for group in groups:
            ids = group["sentence_ids"]
            for lang in languages:
                append_row(stream, {**group, "group_id": f"{experiment['split']}-{ids[0]:04d}",
                                    "language": lang, "split": experiment["split"],
                                    "text": "\n".join(texts[lang][index - 1] for index in ids)})
    write_json(str(output) + ".meta.json", {"experiment": experiment,
               "source_sha256": {lang: sha256(path) for lang, path in sources.items()},
               "manifest_sha256": sha256(output), "rows": count * len(languages), **provenance})


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
