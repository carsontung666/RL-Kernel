# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Load T06 CUDA attention + DetGemm when pip _C cannot be built (nvcc 11 vs torch 12)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
_MERGED = None


def resolve_cuda_home() -> str:
    for home in (
        os.environ.get("T06_CUDA_HOME"),
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-11.8",
        "/usr/local/cuda",
    ):
        if home and (Path(home) / "bin" / "nvcc").is_file():
            return home
    raise RuntimeError("no CUDA toolkit with nvcc found")


def _prepare_env() -> None:
    import torch

    cuda_home = resolve_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CC"] = os.environ.get("CC", "gcc-11")
    os.environ["CXX"] = os.environ.get("CXX", "g++-11")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        str(Path(torch.__file__).parent / "lib")
        + os.pathsep
        + str(Path(cuda_home) / "lib64")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _load(name: str, sources: list[str], extra_include_paths: list[str] | None = None):
    from torch.utils.cpp_extension import load

    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths or [],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-ccbin=g++-11",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _merge(*modules):
    names = {}
    for mod in modules:
        for key in dir(mod):
            if not key.startswith("_"):
                names[key] = getattr(mod, key)
    return SimpleNamespace(**names)


def ensure_native_kernels() -> str:
    """JIT-install attention + DetGemm and patch rl_engine.kernels.ops.base._C."""

    global _MERGED
    import torch

    import rl_engine.kernels.ops.base as base
    from rl_engine.kernels.p2.attention import mqa_joint_attention_sink as mqa

    native_ok = (
        base._EXT_AVAILABLE
        and base._C is not None
        and hasattr(base._C, "det_gemm_fwd_rhs_transposed")
        and hasattr(base._C, "mqa_joint_attention_sink_forward")
    )
    if native_ok:
        mqa._C = base._C
        mqa._EXT_AVAILABLE = True
        return "native"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    if _MERGED is not None:
        return "jit"

    _prepare_env()
    attn = _load(
        "mqa_t06_verify",
        [
            str(_ROOT / "csrc/cuda/attention/mqa_joint_attention_sink.cu"),
            str(_ROOT / "csrc/cuda/attention/mqa_joint_attention_sink_jitbind.cpp"),
        ],
    )
    gemm = _load(
        "det_gemm_t06_verify",
        [
            str(_ROOT / "csrc/cuda/gemm/det_gemm_kernel.cu"),
            str(_ROOT / "csrc/cuda/gemm/det_gemm_jitbind.cpp"),
        ],
        extra_include_paths=[str(_ROOT / "csrc/cuda/gemm")],
    )
    merged = _merge(attn, gemm)
    _MERGED = merged
    base._C = merged
    base._EXT_AVAILABLE = True
    mqa._C = merged
    mqa._EXT_AVAILABLE = True
    return "jit"


def ensure_t06_cuda_kernel() -> str:
    return ensure_native_kernels()
