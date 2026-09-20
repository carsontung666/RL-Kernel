# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import (
    MqaJointAttentionSinkOp,
    cuda_kernel_available,
)
from rl_engine.kernels.p2.attention.oracle import mqa_joint_attention_sink_bwd, mqa_joint_attention_sink_fwd
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import CUDA_VS_ORACLE_BWD_ATOL, CUDA_VS_ORACLE_FWD_ATOL
from rl_engine.kernels.p2.fixtures.catalog import named_attn_catalog

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def test_shipped_cuda_extension_exposes_t06_symbols():
    from rl_engine import _C

    assert hasattr(_C, "mqa_joint_attention_sink_forward")
    assert hasattr(_C, "mqa_joint_attention_sink_backward")
    assert cuda_kernel_available()


def _to_cuda(case):
    plan = CandidatePlan(
        layer_type=case.plan.layer_type,
        n_compressed=case.plan.n_compressed,
        n_recent=case.plan.n_recent,
        valid=case.plan.valid.cuda(),
        kind=None if case.plan.kind is None else case.plan.kind.cuda(),
    )
    return case.q.cuda(), case.k.cuda(), case.v.cuda(), case.sink.cuda(), plan


@pytest.mark.parametrize("name", ["candidate_empty", "c0_recent_1", "csa_selected_c4", "candidate_partial"])
def test_cuda_matches_oracle_fp32(name):
    cpu = named_attn_catalog(device="cpu")[name]
    oracle = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    q, k, v, sink, plan = _to_cuda(cpu)
    out = MqaJointAttentionSinkOp(backend="cuda").forward_fp32(
        q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate
    )
    # CPU torch.exp vs CUDA expf is not bitwise; budget is in contract.py.
    assert torch.allclose(out.o.cpu(), oracle.o, atol=CUDA_VS_ORACLE_FWD_ATOL, rtol=0)
    assert out.provenance.backend == "cuda"
    assert "pragma_unroll_8" in out.provenance.unroll
    assert out.provenance.fallback is False


def test_cuda_forward_is_bitwise_deterministic():
    cpu = named_attn_catalog(device="cpu")["csa_selected_c4"]
    q, k, v, sink, plan = _to_cuda(cpu)
    op = MqaJointAttentionSinkOp(backend="cuda")
    a = op.forward_fp32(q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate)
    b = op.forward_fp32(q, k, v, sink, plan, compare=True, state_gate=cpu.state_gate)
    assert torch.equal(a.o, b.o)


@pytest.mark.parametrize("name", ["c0_recent_1", "candidate_empty", "csa_selected_c4"])
def test_cuda_backward_matches_oracle(name):
    cpu = named_attn_catalog(device="cpu")[name]
    saved = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    d_o = torch.randn_like(saved.o)
    ref = mqa_joint_attention_sink_bwd(d_o, saved, sink_was_shared=cpu.sink.dim() == 1)
    q, k, v, sink, plan = _to_cuda(cpu)
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    out.backward(d_o.cuda())
    assert torch.allclose(q.grad.cpu(), ref.dq, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    if k.numel():
        assert torch.allclose(k.grad.cpu(), ref.dk, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
        assert torch.allclose(v.grad.cpu(), ref.dv, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    assert torch.allclose(sink.grad.cpu(), ref.dsink, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)


def test_cuda_shared_1d_sink_backward():
    cpu = named_attn_catalog(device="cpu")["shared_sink"]
    assert cpu.sink.dim() == 1
    saved = mqa_joint_attention_sink_fwd(cpu.q, cpu.k, cpu.v, cpu.sink, cpu.plan)
    d_o = torch.randn_like(saved.o)
    ref = mqa_joint_attention_sink_bwd(d_o, saved, sink_was_shared=True)
    q, k, v, sink, plan = _to_cuda(cpu)
    assert sink.dim() == 1
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    out.backward(d_o.cuda())
    assert sink.grad.shape == (64,)
    assert torch.allclose(sink.grad.cpu(), ref.dsink, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)
    assert torch.allclose(q.grad.cpu(), ref.dq, atol=CUDA_VS_ORACLE_BWD_ATOL, rtol=0)


def test_cuda_bf16_q_fp32_output_backward():
    cpu = named_attn_catalog(device="cpu")["c0_recent_1"]
    q, k, v, sink, plan = _to_cuda(cpu)
    q = q.to(torch.bfloat16).clone().requires_grad_(True)
    k = k.to(torch.bfloat16).clone().requires_grad_(True)
    v = v.to(torch.bfloat16).clone().requires_grad_(True)
    sink = sink.clone().requires_grad_(True)
    out = MqaJointAttentionSinkOp(backend="cuda").apply_autograd(
        q, k, v, sink, plan, output_fp32=True, compare=True, state_gate=cpu.state_gate
    )
    assert out.dtype == torch.float32
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert q.grad.dtype == torch.bfloat16


def test_eager_cuda_graph_byte_equal_or_fail_closed():
    """Capture in a child process so a failed capture cannot poison later tests."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = r"""
import torch
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import named_attn_catalog
from rl_engine.kernels.p2.four_mode import eager_vs_cuda_graph_attention

cpu = named_attn_catalog(device="cpu")["c0_recent_1"]
q = cpu.q.cuda(); k = cpu.k.cuda(); v = cpu.v.cuda(); sink = cpu.sink.cuda()
plan = CandidatePlan(
    layer_type=cpu.plan.layer_type,
    n_compressed=cpu.plan.n_compressed,
    n_recent=cpu.plan.n_recent,
    valid=cpu.plan.valid.cuda(),
)
try:
    eager, graph = eager_vs_cuda_graph_attention(q, k, v, sink, plan, cpu.state_gate)
except P2FailClosedError as exc:
    assert exc.status is P2Status.UNSUPPORTED_CAPABILITY, exc.status
    print("GRAPH_STATUS=UNSUPPORTED_CAPABILITY")
    raise SystemExit(0)
assert torch.equal(eager, graph)
print("GRAPH_STATUS=BYTE_EQUAL")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "GRAPH_STATUS=" in (proc.stdout + proc.stderr)
