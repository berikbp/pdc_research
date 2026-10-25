# Setup and first run

Run all commands from the repository root. Python code uses only the standard library;
llama.cpp is a separate native build. Linux/WSL is the initial documented workflow.
macOS can use Metal, and other supported GPUs can use Vulkan. Keep backend and OS
differences in the results rather than treating different machines as interchangeable.

## 1. Python and hardware

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
uv run pdc-research doctor --output runs/my-machine-hardware.json
```

Without uv, use `python -m venv .venv`, activate it, and run `python -m pip install -e .`.
The remaining commands can then use `pdc-research` directly.

The hardware report includes failures: a missing GPU tool must not silently turn into
a claim that the machine has no GPU. `nvidia-smi`'s CUDA version describes driver
compatibility; `nvcc --version` checks the installed CUDA compiler.

## 2. Build the pinned llama.cpp

Install Git, CMake, a C++ compiler, and the development dependencies for your backend.
For Ubuntu/WSL, the basic compiler packages are `build-essential` and `cmake`.
For CUDA, install an appropriate CUDA toolkit inside WSL, following NVIDIA's WSL
instructions; use the Windows NVIDIA driver rather than installing a Linux display driver.
Check `nvidia-smi` and `nvcc --version` in your terminal.

```bash
uv run python scripts/setup_llama.py --backend cuda
vendor/llama.cpp/build/bin/llama-server --version
vendor/llama.cpp/build/bin/llama-server --list-devices
```

Use `--backend cpu`, `metal`, or `vulkan` as appropriate. The script pins
`configs/toolchain.json`, refuses dirty source checkouts, and limits compilation
to four jobs by default. It downloads source and builds tools; it does not install
drivers, system packages, models, or datasets. See the upstream
[build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
For native Windows, use a compatible compiler/CMake environment and adjust `llama_bin`
if binaries land in `build/bin/Release`.

### Optional project-local CUDA toolkit (Linux x86-64)

If a system toolkit is unavailable, NVIDIA's wheels provide a local alternative.
This downloads roughly 0.5 GiB and installs under ignored `vendor/`, without
installing a display driver or requiring sudo. CUDA 13.1 requires a compatible
host compiler and NVIDIA driver. Other machines should use a compatible toolkit
for their drivers instead of blindly copying this version.

```bash
uv pip install --python .venv/bin/python --target vendor/cuda -r configs/cuda-linux.txt
uv run python scripts/setup_llama.py --backend cuda --cuda-root vendor/cuda/nvidia/cu13 --cuda-arch 89
```

Compute capability `89` targets the RTX 4060. Omit `--cuda-arch` for automatic
detection on other GPUs. The wheel layout uses `nvidia/cu13` as toolkit root.
The setup script adds missing linker aliases inside this project-local installation
so CMake can find the libraries.
The compiler/runtime are CUDA 13.1; cuBLAS and its NVRTC dependency use compatible
CUDA 13 packages already cached on the initial machine. Keep their versions recorded
when comparing builds. See [NVIDIA's package description](https://pypi.org/project/cuda-toolkit/13.1.0/).

On Ubuntu 26.04, CUDA 13.1 can fail its compiler check with conflicting `rsqrt`
exception declarations in newer glibc. For this specific error, add
`--cuda-glibc-workaround` to the build command. It adjusts only the two declarations
in the toolkit under `vendor/cuda`, preserves the original header, and records
the option in `vendor/llama.cpp/build/pdc-build.json`. It does not edit system headers.
The compatibility issue and declaration fix are described in the
[NVIDIA developer forum](https://forums.developer.nvidia.com/t/fedora-43-and-nvcc-cuda13-1-error-exception-specification-is-incompatible-rsqrt-rsqrtf/354510/6).

## 3. Model

For the pinned assets used by this project, run:

```bash
uv run python scripts/download_assets.py
```

This retrieves and verifies the two official GGUF shards, merges them using the
built tool, and downloads/extracts the original FLORES-200 archive. Revisions,
sizes, and SHA-256 values are pinned in `configs/assets.json`. Partial downloads
are resumable; model files and metadata remain under ignored `models/` and `data/`.
Use `--only model` or `--only dataset` for a single asset. Keep the shards until
the merged model is verified; this workflow uses roughly 9.4 GB for model assets.

The shared model used in the reported experiments is **Qwen2.5-7B-Instruct, Q4_K_M**, a dense model with
28 transformer layers. This is an execution benchmark, not an evaluation of its
Kazakh generation quality. Validate the model choice during the pilot.

Alternatively, download the Q4_K_M GGUF file(s) from the
[official model repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF).
Keep all shards if split. Merge using the built `llama-gguf-split --merge FIRST_SHARD OUTPUT`
and set `model` to the resulting file. The default path is:

```text
models/qwen2.5-7b-instruct-q4_k_m.gguf
```

Agree on the exact file before collecting results across machines. Runners hash the
model and binaries; do not compare different hashes as the same model. Downloading
several gigabytes is a separate setup step, not an implicit part of a benchmark.

## 4. FLORES-200

Download the **original FLORES-200** release linked by the
[official README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).
Record the archive URL, download date, and archive SHA-256 in your experiment notes.
Extract locally; the expected layout is:

```text
data/raw/flores200_dataset/dev/eng_Latn.dev
data/raw/flores200_dataset/dev/rus_Cyrl.dev
data/raw/flores200_dataset/dev/kaz_Cyrl.dev
data/raw/flores200_dataset/dev/zho_Hans.dev
```

Preserve sentence order. The preparation command validates line counts, uses
one-based sentence IDs, and records source file hashes. Do not substitute FLORES+
or another revision without creating a new dataset version. Follow the dataset's
license and citation requirements when distributing it.

```bash
uv run pdc-research prepare --source data/raw/flores200_dataset
```

This creates `data/workloads/pilot-v3.jsonl` and its metadata sidecar. The pilot uses
the frozen single-article selection in `configs/workloads/coherent-dev-v1.json`;
preparation rejects article-boundary crossings, overlapping IDs, and source URL mismatches.
Existing manifests
are refused; use a new output path to change settings. Share this exact manifest
and sidecar with teammates, or verify matching hashes after regeneration.

## 5. Machine configuration

Berik's initial profile is `configs/machines/berik-wsl.json`. Teammates should copy
`configs/machines/example.json` to `configs/machines/NAME.local.json`, set a unique
machine ID, and edit paths and feasible settings. Local profiles are ignored by Git;
each run embeds its profile. Commit reviewed shared profiles when useful.

The example tests 4 and 8 CPU threads, and 0, 8, 16, 24, 28, and 99 GPU layers.
99 requests full offload, including any additional offloadable output layer. It is
a request, not a claim of 99 physical layers. Keep server logs to inspect actual placement.
Configurations that do not fit are failures, not slow successful trials.

```bash
uv run pdc-research plan --machine configs/machines/berik-wsl.json
```

Before the whole pilot, run the provided smoke experiment: one frozen coherent group,
one repetition, 16 output tokens, one thread count and GPU settings `[0, 99]`.
Use a reduced copy of your own profile for the smoke run.

```bash
uv run pdc-research prepare --experiment configs/experiments/smoke.json --source data/raw/flores200_dataset --output data/workloads/smoke.jsonl
uv run pdc-research run --machine configs/machines/berik-smoke.json --experiment configs/experiments/smoke.json --manifest data/workloads/smoke.jsonl --output runs/berik-smoke-001
```

Teammates should use their own machine profile with similarly reduced settings.

## 6. Baseline and real-text runs

```bash
uv run pdc-research baseline --machine configs/machines/berik-wsl.json --output runs/berik-baseline-001
uv run pdc-research run --machine configs/machines/berik-wsl.json --output runs/berik-pilot-001
```

Baseline uses synthetic `llama-bench` pp and tg tests. Its tg test is separate from
the FLORES context; do not interpret it as multilingual decoding. Real-text runs
start a fresh server for each configuration, reuse the loaded model for sequential
requests, and stop that server when finished. Keep port 8080 free or change the profile.
Only one benchmark should run on a machine at a time.

Real-text runs now save durable per-request JSON files and per-prompt summaries.
See [running and resuming](running-and-resuming.md) for the artifact layout, pause
controls, and resume commands. Existing runs require explicit `--resume`, and
model, workload, settings, and backend fingerprints must match.

Record `doctor` output beside the run. Keep the laptop plugged in, use the same power
mode, stop competing workloads, and record cooling/power conditions. Only run one
benchmark on a machine at a time. A process lock also protects each run directory.

## Troubleshooting

- **Agent cannot access GPU, terminal can:** run `doctor` and benchmarks in your terminal;
  do not replace working drivers because a sandbox denies GPU access.
- **No CUDA compiler:** install the toolkit; a successful `nvidia-smi` is insufficient.
- **Out of memory:** remove that offload setting or reduce context/batch settings in a
  new profile. Record the change; use identical settings for comparisons requiring them.
- **No matched inputs at a length:** four sentences can be too short. Increase group
  width in a new experiment, then regenerate. Never pad by repeating tokens.
- **Build/API incompatibility:** use the pinned commit, and check server logs. Upgrade
  the pin in a reviewed change with a fresh smoke run before collecting new results.
- **Windows binary path:** set `llama_bin` to the directory containing both executables.

## Development checks

```bash
uv run python -m unittest discover -s tests -v
```

Tests use tiny artificial files and mocked responses. They do not download models
or claim to validate GPU inference.

## 7. Replicate the controlled RQ1 experiment

The earlier commands document the exploratory pilot. The current controlled
protocol and its nine settings are in [rq1-controlled.md](rq1-controlled.md).
After preparing `data/workloads/pilot-v3.jsonl`, run:

```bash
uv run pdc-research plan --machine configs/machines/NAME.local.json --experiment configs/experiments/rq1-controlled.json
uv run pdc-research run-controlled --machine configs/machines/NAME.local.json --experiment configs/experiments/rq1-controlled.json --output runs/NAME-rq1-001
uv run pdc-research status --output runs/NAME-rq1-001
uv run pdc-research analyze-controlled --run runs/NAME-rq1-001 --output runs/NAME-rq1-001/analysis
```

Use your own profile, not Berik's hardware metadata. The controlled matrix overrides
its thread/layer lists with the nine explicit entries in the experiment config.
The planner reports an upper bound before tokenization; the historical model and
manifest yield 19 eligible groups, 76 inputs and 2,052 requests. The current
64-token protocol explicitly checks that eligibility and refuses a changed
sample. Investigate tokenizer/model differences rather than weakening that check
silently. The GPU-only entry explicitly names CUDA0; other backends require new
placement verification. Settings that do not fit VRAM must be reported as failures.

Append `--resume` to the same `run-controlled` command after interruption. Creating
`runs/NAME-rq1-001/STOP` pauses after the current request is saved. No command
schedules shutdown. The code can run in your terminal; systemd is not required.

`doctor --output runs/NAME-hardware.json` should be outside the future run folder,
which must not already exist when starting a new experiment. A plain `status`
reports the last saved phase and reconstructs committed results; it does not
prove that a process is live.

For sharing, commit your reviewed report and small summary tables. Keep raw JSON,
model generations, logs and per-request CSV exports in ignored `runs/`. Copy the
whole run directory to a separate backup if it is needed for research verification.
