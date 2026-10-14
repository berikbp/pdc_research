"""Fetch the pinned llama.cpp revision and build its benchmark tools.

Usage: python scripts/setup_llama.py --backend cuda
Run from the repository root. Requires Git, CMake and a C++ compiler.
"""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda", "metal", "vulkan"], required=True)
    parser.add_argument("--jobs", type=int, default=4, help="Limit build RAM consumption")
    parser.add_argument("--cuda-root", type=Path, help="Optional project-local CUDA toolkit root")
    parser.add_argument("--cuda-arch", help="Optional CUDA compute capability, e.g. 89 for RTX 4060")
    parser.add_argument("--cuda-glibc-workaround", action="store_true",
                        help="Fix CUDA 13.1 rsqrt declarations for newer glibc; project-local toolkit only")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.cuda_root and not (args.cuda_root / "bin/nvcc").is_file():
        parser.error(f"CUDA compiler missing at {args.cuda_root / 'bin/nvcc'}; install toolkit first")
    if args.cuda_glibc_workaround and (
        not args.cuda_root
        or not args.cuda_root.resolve().is_relative_to(Path("vendor/cuda").resolve())
    ):
        parser.error("--cuda-glibc-workaround requires a toolkit inside vendor/cuda")
    return args


def checkout_source(root: Path, pin: dict) -> None:
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "remote", "add", "origin", pin["llama_cpp_repository"]],
            check=True,
        )
    has_head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True
    ).returncode == 0
    if has_head and subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("llama.cpp has local changes; preserve them before changing revisions")
    has_commit = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", pin["llama_cpp_commit"] + "^{commit}"],
        capture_output=True,
    ).returncode == 0
    if not has_commit:
        subprocess.run(
            ["git", "-C", str(root), "fetch", "--depth", "1", "origin", pin["llama_cpp_commit"]],
            check=True,
        )
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", pin["llama_cpp_commit"]], check=True)


def prepare_cuda_toolkit(cuda_root: Path, glibc_workaround: bool) -> None:
    # NVIDIA Linux wheels omit the unversioned linker names and lib64 alias.
    # Adapt only project-local wheels; never mutate a system CUDA installation.
    if cuda_root.is_relative_to(Path("vendor/cuda").resolve()) and (cuda_root / "lib").is_dir():
        lib64 = cuda_root / "lib64"
        if not lib64.exists():
            lib64.symlink_to("lib", target_is_directory=True)
        for library in (cuda_root / "lib").glob("lib*.so.*"):
            alias = library.with_name(library.name.split(".so.")[0] + ".so")
            if not alias.exists():
                alias.symlink_to(library.name)
    if glibc_workaround:
        header = cuda_root / "include/crt/math_functions.h"
        original = header.read_text()
        patched = original
        for declaration in ("rsqrt(double x)", "rsqrtf(float x)"):
            patched = patched.replace(declaration + ";", declaration +
                      "\n#if __NV_GLIBC_PROVIDES_IEC_60559_FUNCS\n__THROW\n#endif\n;")
        if patched != original:
            backup = header.with_suffix(".h.pdc-original")
            if not backup.exists():
                backup.write_text(original)
            header.write_text(patched)


def build_configuration(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    options = [f"-DGGML_{backend.upper()}={'ON' if backend == args.backend else 'OFF'}"
               for backend in ("cuda", "metal", "vulkan")]
    environment = os.environ.copy()
    if "microsoft" in platform.release().lower():
        # Windows PATH entries cause slow optional-library searches in Linux CMake.
        environment["PATH"] = os.pathsep.join(part for part in environment["PATH"].split(os.pathsep)
                                               if not part.startswith("/mnt/"))
    if args.cuda_root:
        cuda_root = args.cuda_root.resolve()
        prepare_cuda_toolkit(cuda_root, args.cuda_glibc_workaround)
        options.extend([f"-DCUDAToolkit_ROOT={cuda_root}", f"-DCMAKE_CUDA_COMPILER={cuda_root / 'bin/nvcc'}"])
        options.append(f"-DCMAKE_BUILD_RPATH={cuda_root / 'lib'}")
        environment["PATH"] = str(cuda_root / "bin") + os.pathsep + environment["PATH"]
    if args.cuda_arch:
        options.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
    return options, environment


def write_build_record(root: Path, pin: dict, args: argparse.Namespace, options: list[str]) -> None:
    metadata = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                "commit": pin["llama_cpp_commit"], "backend": args.backend,
                "cuda_root": str(args.cuda_root.resolve()) if args.cuda_root else None,
                "cuda_arch": args.cuda_arch, "cmake_options": options,
                "cuda_glibc_workaround": args.cuda_glibc_workaround,
                "cmake_version": subprocess.check_output(["cmake", "--version"], text=True)}
    (root / "build" / "pdc-build.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    pin = json.loads(Path("configs/toolchain.json").read_text(encoding="utf-8"))
    root = Path("vendor/llama.cpp")
    checkout_source(root, pin)
    options, environment = build_configuration(args)
    subprocess.run(
        ["cmake", "-S", str(root), "-B", str(root / "build"),
         "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", *options],
        check=True, env=environment,
    )
    subprocess.run(
        ["cmake", "--build", str(root / "build"), "--config", "Release",
         "--parallel", str(args.jobs), "--target", "llama-server", "llama-bench", "llama-gguf-split"],
        check=True, env=environment,
    )
    write_build_record(root, pin, args, options)
    print("Built pinned llama.cpp. On multi-configuration generators, binaries may be in build/bin/Release.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Build failed: {exc}") from exc
