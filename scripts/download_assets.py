"""Download the pinned GGUF model and original FLORES-200; run from the repo root."""

import argparse
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pdc_research.backend import binary
from pdc_research.common import read_json, sha256, write_json


def verify(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Unexpected file size: {path}")
    if sha256(path) != asset["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def download(url: str, path: Path, asset: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path, asset)
        print(f"Verified existing {path}", flush=True)
        return path
    partial = path.with_name(path.name + ".part")
    if not partial.exists() or partial.stat().st_size < asset["size_bytes"]:
        print(f"Downloading {path}", flush=True)
        subprocess.run([
            "curl", "-4", "-fLsS", "--retry", "5", "--retry-all-errors",
            "--retry-delay", "3", "--connect-timeout", "15",
            "--speed-time", "60", "--speed-limit", "1024",
            "-C", "-", url, "-o", str(partial),
        ], check=True)
    verify(partial, asset)
    partial.rename(path)
    print(f"Downloaded and verified {path}", flush=True)
    return path


def prepare_model(config: dict) -> None:
    model = config["model"]
    base = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}"

    def shard(asset: dict) -> Path:
        return download(f"{base}/{asset['filename']}?download=true",
                        Path("models") / asset["filename"], asset)

    with ThreadPoolExecutor(max_workers=2) as executor:
        shards = list(executor.map(shard, model["files"]))
    merged = Path("models") / model["merged_filename"]
    record = merged.with_suffix(".metadata.json")
    if merged.exists():
        provenance = read_json(record) if record.exists() else {}
        if provenance.get("source") != model or provenance.get("merged_sha256") != sha256(merged):
            raise ValueError(f"Existing merged model lacks matching provenance: {merged}")
        print(f"Verified existing merged model {merged}")
        return
    temporary = merged.with_name(merged.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete merge exists; inspect before retrying: {temporary}")
    tool = binary({"llama_bin": "vendor/llama.cpp/build/bin"}, "llama-gguf-split")
    subprocess.run([tool, "--merge", str(shards[0]), str(temporary)], check=True)
    merged_hash = sha256(temporary)
    temporary.rename(merged)
    write_json(record, {"source": model, "merged_sha256": merged_hash,
                       "merged_bytes": merged.stat().st_size,
                       "merge_tool_sha256": sha256(tool),
                       "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Ready: {merged}", flush=True)


def prepare_dataset(config: dict) -> None:
    dataset = config["dataset"]
    archive = download(dataset["url"], Path("data/raw") / dataset["filename"], dataset)
    target = Path("data/raw/flores200_dataset")
    if not target.exists():
        with tarfile.open(archive, "r:gz") as stream:
            stream.extractall(archive.parent, filter="data")
    # Refuse an incomplete or modified extraction, including when reusing a folder.
    with tarfile.open(archive, "r:gz") as stream:
        members = {str(PurePosixPath(member.name)): member for member in stream.getmembers()}
        for split in ("dev", "devtest"):
            for language in ("eng_Latn", "rus_Cyrl", "kaz_Cyrl", "zho_Hans"):
                relative = f"flores200_dataset/{split}/{language}.{split}"
                member = stream.extractfile(members[relative])
                if member is None or (archive.parent / relative).read_bytes() != member.read():
                    raise ValueError(f"Extraction does not match pinned archive: {relative}")
    write_json("data/raw/flores200-source.json", {
        "source_url": dataset["url"], "archive_sha256": dataset["sha256"],
        "archive_bytes": dataset["size_bytes"],
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Ready: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["model", "dataset"])
    args = parser.parse_args()
    config = read_json("configs/assets.json")
    if args.only != "dataset":
        prepare_model(config)
    if args.only != "model":
        prepare_dataset(config)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Asset setup failed: {exc}") from exc
