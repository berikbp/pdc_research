# Initial laptop build

The pinned llama.cpp tools were built successfully for Berik's WSL laptop.

- Revision: `ecdd29e8b2076bb85da9e9ca7ee3d84264375bd1`.
- Release build with GNU 15.2.0, project-local CUDA 13.1.80, architecture `89`.
- Tools: `llama-server`, `llama-bench`, `llama-gguf-split` under `vendor/llama.cpp/build/bin/`.
- Server device check: NVIDIA GeForce RTX 4060 Laptop GPU, approximately 8 GiB VRAM.
- Five Python pipeline tests passed.

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
Next: obtain the GGUF model and FLORES-200, then run the documented smoke experiment.
