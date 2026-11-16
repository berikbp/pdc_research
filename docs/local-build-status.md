# Initial laptop build

The pinned llama.cpp tools were built successfully for Berik's WSL laptop.

- Revision: `ecdd29e8b2076bb85da9e9ca7ee3d84264375bd1`.
- Release build with GNU 15.2.0, project-local CUDA 13.1.80, architecture `89`.
- Tools: `llama-server`, `llama-bench`, `llama-gguf-split` under `vendor/llama.cpp/build/bin/`.
- Server device check: NVIDIA GeForce RTX 4060 Laptop GPU, approximately 8 GiB VRAM.
- Historical initial build check passed; the current test suite is documented in development notes.

Build reproduction:

```bash
uv pip install --python .venv/bin/python --target vendor/cuda -r configs/cuda-linux.txt
uv run python scripts/setup_llama.py --backend cuda --cuda-root vendor/cuda/nvidia/cu13 --cuda-arch 89 --cuda-glibc-workaround
vendor/llama.cpp/build/bin/llama-server --list-devices
```

The local wheel installation needs linker aliases and a build runtime library path;
the setup script handles both. Ubuntu 26.04 also required the documented CUDA/glibc
declaration workaround. The script preserves the original header and saves build
options in `vendor/llama.cpp/build/pdc-build.json`. Keep the `vendor/cuda` directory
alongside this build because its shared libraries are used at runtime.

No model inference or FLORES performance results were collected during this build.
The subsequent asset setup downloaded and verified the pinned model and FLORES-200,
and prepared the workload manifests. The pilot and subsequent experiments are now
complete; see [the current report](report-2026-10-wsl.md). This page records only the
initial native build, not current experiment progress.

## Native Linux build (Pop!_OS 24.04, November 2026)

The same pinned revision was rebuilt on the same laptop under native Linux.

- Driver 580.173.02 reports CUDA 13.0 compatibility, so the 13.1 wheel list used
  under WSL does not apply; `configs/cuda-linux-13.0.txt` pins toolkit 13.0.88.
- GNU 13.3.0 and glibc 2.39: the `--cuda-glibc-workaround` option is not needed.
- CMake and Ninja came from `uv tool install cmake ninja`; no system packages
  or sudo were required.
- Port 8080 is held by a system service on this machine; the native profiles use 18080.

```bash
uv tool install cmake && uv tool install ninja
uv sync --locked
uv pip install --python .venv/bin/python --target vendor/cuda -r configs/cuda-linux-13.0.txt
uv run python scripts/setup_llama.py --backend cuda --cuda-root vendor/cuda/nvidia/cu13 --cuda-arch 89 --jobs 12
vendor/llama.cpp/build/bin/llama-server --list-devices
uv run python scripts/download_assets.py
```

Machine profiles: `configs/machines/berik-popos.json` and `berik-popos-smoke.json`,
with `machine_id` `berik-7435hs-rtx4060-popos`. Native Linux sees 23 GiB RAM against
roughly 11 GiB under WSL, and the driver has no silent system-memory fallback, so
results from the two environments must not be pooled.

Before measuring: `system76-power profile performance`, AC power, lid open, no other
workloads. Each run records these conditions in `conditions.jsonl`.
