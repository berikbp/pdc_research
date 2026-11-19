"""Check that NEO computes a model correctly before measuring it: greedy tokens from NEO's GPU path,
NEO's CPU-attention path and vLLM must agree on the same prompt.

Two steps in separate processes (each needs the GPU to itself):

    python scripts/check_neo_model.py neo  --machine M.json --cases W.json --output DIR
    python scripts/check_neo_model.py vllm --machine M.json --cases W.json --output DIR
    python scripts/check_neo_model.py compare --output DIR      # exit code 0 only if they agree

The prompt is the first English passage of the tokenized workload, so no tokenizer is needed.
fp16 rounding may flip a late greedy token, so agreement is required on the first AGREE tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path

GENERATED = 17   # prefill token + 16 decode steps, as in NEO's examples/example.py
AGREE = 8


def prompt(cases_path: str) -> list[int]:
    return json.loads(Path(cases_path).read_text())["cases"]["eng_Latn"][0]["tokens"]


def run_neo(machine: dict, tokens: list[int], output: Path) -> None:
    sys.path[:0] = [machine["neo_dir"], f"{machine['neo_dir']}/csrc"]
    import swiftllm  # noqa: PLC0415  (only importable inside the NEO environment)

    config = swiftllm.EngineConfig(
        model_path=machine["model"], use_dummy=False, block_size=16, gpu_mem_utilization=0.9,
        num_gpu_blocks_override=200, swap_space=4, max_seqs_in_block_table=10, max_blocks_per_seq=100,
        max_batch_size=10, max_tokens_in_batch=1024, library_path=machine["neo_library"],
        profile_result_path=str(output / "neo-profile") + "/", extra_layer_for_cprf=True,
        tensor_parallel_degree=1)
    engine = swiftllm.Engine(config)
    engine.initialize()
    # Same layout as examples/example.py: requests 0 and 2 decode on the GPU, 1 and 3 on the CPU.
    reqs = [swiftllm.create_request(tokens, i) for i in range(4)]
    batch = swiftllm.SubBatch()
    for i in (0, 2):
        batch.add_pref(reqs[i], is_gpu=True)
    engine.step([batch])
    for i in (1, 3):
        batch = swiftllm.SubBatch()
        batch.add_pref(reqs[i], is_gpu=False)
        engine.step([batch])
    for _ in range(GENERATED - 1):
        batches = [swiftllm.SubBatch() for _ in range(2)]
        batches[0].add_gdec(reqs[0])
        batches[1].add_cdec(reqs[1])
        batches[1].add_gdec(reqs[2])
        batches[0].add_cdec(reqs[3])
        engine.step(batches)
    result = {"gpu": [list(reqs[0].output_token_ids), list(reqs[2].output_token_ids)],
              "cpu": [list(reqs[1].output_token_ids), list(reqs[3].output_token_ids)]}
    (output / "neo.json").write_text(json.dumps(result))


def run_vllm(machine: dict, tokens: list[int], output: Path) -> None:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(model=machine["model"], dtype="float16", gpu_memory_utilization=0.6, max_model_len=1024,
              enforce_eager=True)
    out = llm.generate([{"prompt_token_ids": tokens}],
                       SamplingParams(temperature=0, max_tokens=GENERATED, ignore_eos=True))
    (output / "vllm.json").write_text(json.dumps({"tokens": list(out[0].outputs[0].token_ids)}))


def compare(output: Path) -> int:
    neo = json.loads((output / "neo.json").read_text())
    reference = json.loads((output / "vllm.json").read_text())["tokens"]
    paths = {"neo_gpu_0": neo["gpu"][0], "neo_gpu_2": neo["gpu"][1],
             "neo_cpu_1": neo["cpu"][0], "neo_cpu_3": neo["cpu"][1]}
    agree = {name: tokens[:AGREE] == reference[:AGREE] for name, tokens in paths.items()}
    common = {name: next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b),
                         min(len(tokens), len(reference))) for name, tokens in paths.items()}
    verdict = {"agree_first_tokens": AGREE, "agree": agree, "matching_prefix_with_vllm": common,
               "vllm": reference, **paths, "ok": all(agree.values())}
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in ("ok", "agree", "matching_prefix_with_vllm")}))
    return 0 if verdict["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["neo", "vllm", "compare"])
    parser.add_argument("--machine")
    parser.add_argument("--cases")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.step == "compare":
        return compare(args.output)
    machine = json.loads(Path(args.machine).read_text())
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(machine["gpu_index"]))
    if machine.get("omp_threads"):
        os.environ["OMP_NUM_THREADS"] = str(machine["omp_threads"])
    (run_neo if args.step == "neo" else run_vllm)(machine, prompt(args.cases), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
