"""Tokenize a workload manifest once for the stage 2 serving runner.

The runner sends token IDs, so it needs no tokenizer library and every system gets
exactly the same input. Needs the `tokenizers` package; the project itself stays
dependency-free. A derived language such as `kaz_Cyrl@eng_len` truncates each Kazakh
passage to the token length of the English passage of the same group, which gives a
length-matched control for RQ2.

    python scripts/tokenize_workload.py --tokenizer PATH/tokenizer.json \
        --source NousResearch/Llama-2-7b-hf@8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
        --manifest data/workloads/pilot-v3.jsonl --output data/workloads/pilot-v3-llama2.json
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truncated(cases: dict, language: str, length_of: str) -> list[dict]:
    """`language` passages cut to the token length of `length_of` passages of the same group."""
    lengths = {case["group_id"]: len(case["tokens"]) for case in cases[length_of]}
    result = []
    for case in cases[language]:
        limit = lengths[case["group_id"]]
        if limit > len(case["tokens"]):
            raise ValueError(f"{case['group_id']}: {language} is shorter than {length_of}")
        result.append({"group_id": case["group_id"], "tokens": case["tokens"][:limit]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json of the served model")
    parser.add_argument("--source", required=True, help="Model repository and revision, recorded as provenance")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--derive", action="append", default=["kaz_Cyrl@eng_len=kaz_Cyrl:eng_Latn"],
                        help="NAME=LANGUAGE:LENGTH_OF (default: Kazakh cut to English length)")
    args = parser.parse_args()

    from tokenizers import Tokenizer  # Imported here so the module is importable without it.

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    cases: dict[str, list[dict]] = {}
    for row in rows:
        tokens = tokenizer.encode(row["text"], add_special_tokens=True).ids
        cases.setdefault(row["language"], []).append({"group_id": row["group_id"], "tokens": tokens})
    derived = {}
    for rule in args.derive:
        name, _, spec = rule.partition("=")
        language, _, length_of = spec.partition(":")
        cases[name] = truncated(cases, language, length_of)
        derived[name] = {"language": language, "truncated_to_length_of": length_of}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "tokenizer": {"source": args.source, "tokenizer_json_sha256": sha256(args.tokenizer)},
        "manifest_sha256": sha256(args.manifest), "add_special_tokens": True,
        "derived": derived, "cases": cases,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    for language, items in cases.items():
        lengths = [len(item["tokens"]) for item in items]
        print(f"{language}: {len(items)} passages, mean {sum(lengths) / len(lengths):.1f} tokens "
              f"({min(lengths)}-{max(lengths)})")


if __name__ == "__main__":
    main()
