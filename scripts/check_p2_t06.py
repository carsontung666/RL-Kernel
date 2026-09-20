#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""T06 local verification runner.

CPU oracle + negatives + recorded block always run.
CUDA: uses rl_engine._C if present; otherwise JIT-compiles the T06 kernel
with g++-11 / nvcc (this machine's working toolchain) and injects it.

This is recorded-operator verification, not a live DSV4 / Qwen3 training run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_pytest(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def _resolve_cuda_home() -> str:
    candidates = [
        os.environ.get("T06_CUDA_HOME"),
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-11.8",
        "/usr/local/cuda",
    ]
    for home in candidates:
        if home and (Path(home) / "bin" / "nvcc").is_file():
            return home
    raise RuntimeError("no CUDA toolkit with nvcc found (tried CUDA_HOME and /usr/local/cuda-11.8)")


def _jit_cuda_module():
    cuda_home = _resolve_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CC"] = os.environ.get("CC", "gcc-11")
    os.environ["CXX"] = os.environ.get("CXX", "g++-11")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    torch_lib = Path(sys.prefix)
    # site-packages torch lib
    import torch

    os.environ["LD_LIBRARY_PATH"] = (
        str(Path(torch.__file__).parent / "lib")
        + os.pathsep
        + str(Path(cuda_home) / "lib64")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    from torch.utils.cpp_extension import load

    return load(
        name="mqa_t06_verify",
        sources=[
            str(ROOT / "csrc/cuda/attention/mqa_joint_attention_sink.cu"),
            str(ROOT / "csrc/cuda/attention/mqa_joint_attention_sink_jitbind.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-ccbin=g++-11",
        ],
        extra_cflags=["-std=c++17"],
        verbose=True,
    )


def _inject_cuda(mod) -> None:
    import rl_engine.kernels.p2.attention.mqa_joint_attention_sink as mqa

    mqa._C = mod
    mqa._EXT_AVAILABLE = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {
        "task_id": "T06",
        "contract_version": "p2-task-contract.v1",
        "scope": "recorded operators; not live DSV4/Qwen3 training",
        "verdicts": {},
    }

    cpu_args = [
        "tests/p2",
        "--ignore=tests/p2/test_mqa_joint_attention_cuda.py",
    ]
    cpu_rc = _run_pytest(cpu_args)
    report["verdicts"]["cpu_pytest"] = "PASS" if cpu_rc == 0 else "FAIL"
    if cpu_rc != 0:
        _write(report, args.json_out)
        return cpu_rc

    if args.skip_cuda:
        report["verdicts"]["cuda"] = "SKIP"
        _write(report, args.json_out)
        print(json.dumps(report, indent=2))
        return 0

    import torch

    from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import cuda_kernel_available

    cuda_source = "unavailable"
    if not torch.cuda.is_available():
        report["verdicts"]["cuda"] = "SKIP"
        report["verdicts"]["cuda_reason"] = "no GPU"
    else:
        try:
            if cuda_kernel_available():
                cuda_source = "rl_engine._C"
            else:
                print("[t06] _C missing mqa symbols; JIT-compiling CUDA reference", flush=True)
                mod = _jit_cuda_module()
                _inject_cuda(mod)
                cuda_source = "jit_gcc11"
            import pytest
            import rl_engine.kernels.p2.attention.mqa_joint_attention_sink as mqa

            assert mqa.cuda_kernel_available()
            # Must run in-process so the JIT injection is visible to tests.
            cuda_rc = pytest.main(["-q", "--tb=line", str(ROOT / "tests/p2/test_mqa_joint_attention_cuda.py")])
            report["verdicts"]["cuda_pytest"] = "PASS" if cuda_rc == 0 else "FAIL"
            report["verdicts"]["cuda_source"] = cuda_source
            if cuda_rc != 0:
                _write(report, args.json_out)
                return cuda_rc
            report["verdicts"]["cuda"] = "PASS"
        except Exception as exc:
            report["verdicts"]["cuda"] = "FAIL"
            report["verdicts"]["cuda_error"] = f"{type(exc).__name__}: {exc}"
            _write(report, args.json_out)
            print(json.dumps(report, indent=2))
            return 1

    _write(report, args.json_out)
    print(json.dumps(report, indent=2), flush=True)
    print("T06 verification", report["verdicts"], flush=True)
    return 0


def _write(report: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
