# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Drive shipped OProjGroupedOp DetGemm path on the installed _C extension."""

import pytest
import torch

from rl_engine import _C
from rl_engine.kernels.p2.contract import HIDDEN_SIZE
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import make_oproj_case
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def test_installed_extension_exposes_det_gemm_symbols():
    assert hasattr(_C, "det_gemm_fwd_rhs_transposed")
    assert hasattr(_C, "det_gemm_fwd")
    assert hasattr(_C, "det_gemm_db_transposed")
    assert _C.det_gemm_sm90_compiled() is False


def test_o_proj_det_gemm_runs_on_installed_extension():
    case = make_oproj_case("det", tokens=1, seed=0)
    o = case.o.cuda().to(torch.bfloat16)
    w_a = case.w_a.cuda().to(torch.bfloat16)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    cos = case.cos.cuda()
    sin = case.sin.cuda()
    result = OProjGroupedOp(backend="det_gemm").forward(o, w_a, w_b, cos, sin)
    assert result.y.shape == (1, HIDDEN_SIZE)
    assert result.y.dtype == torch.bfloat16
    assert torch.isfinite(result.y).all()
    assert result.provenance.backend == "det_gemm"
    assert "sm90" not in result.provenance.kernel_id.lower()
    assert "sm90" not in (result.provenance.build_fingerprint or "").lower()
    ref = OProjGroupedOp(backend="torch_fp32").forward_fp32(
        o.float(), w_a.float(), w_b.float(), cos, sin
    )
    delta = (result.y.float() - ref.y.float()).abs()
    assert float(delta.max()) < 5e-3
    assert o.data_ptr() == result.saved.o.data_ptr()


def test_o_proj_det_gemm_refuses_silent_fp32_downcast():
    case = make_oproj_case("fp32", tokens=1, seed=1)
    with pytest.raises(P2FailClosedError) as exc:
        OProjGroupedOp(backend="det_gemm").forward(
            case.o.cuda(),
            case.w_a.cuda(),
            case.w_b.cuda(),
            case.cos.cuda(),
            case.sin.cuda(),
        )
    assert exc.value.status is P2Status.ROUND_POINT_MISMATCH


def test_o_proj_det_gemm_forward_fp32_returns_fp32():
    case = make_oproj_case("fp32out", tokens=1, seed=3)
    o = case.o.cuda().to(torch.bfloat16)
    w_a = case.w_a.cuda().to(torch.bfloat16)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    result = OProjGroupedOp(backend="det_gemm").forward_fp32(
        o, w_a, w_b, case.cos.cuda(), case.sin.cuda()
    )
    assert result.y.dtype == torch.float32
    assert result.y.shape == (1, HIDDEN_SIZE)
    assert torch.isfinite(result.y).all()


def test_o_proj_det_gemm_op_backward():
    case = make_oproj_case("opbwd", tokens=1, seed=4)
    o = case.o.cuda().to(torch.bfloat16)
    w_a = case.w_a.cuda().to(torch.bfloat16)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    cos = case.cos.cuda()
    sin = case.sin.cuda()
    op = OProjGroupedOp(backend="det_gemm")
    result = op.forward(o, w_a, w_b, cos, sin)
    d_y = torch.randn_like(result.y, dtype=torch.float32)
    d_o, d_w_a, d_w_b = op.backward(d_y, result.saved, cos, sin)
    assert d_o.shape == o.shape
    assert d_w_a.shape == w_a.shape
    assert d_w_b.shape == w_b.shape
    assert torch.isfinite(d_o).all()
    assert torch.isfinite(d_w_a).all()
    assert torch.isfinite(d_w_b).all()


def test_o_proj_det_gemm_backward_finite():
    case = make_oproj_case("bwd", tokens=1, seed=2)
    o = case.o.cuda().to(torch.bfloat16).clone().requires_grad_(True)
    w_a = case.w_a.cuda().to(torch.bfloat16).clone().requires_grad_(True)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    y = OProjGroupedOp(backend="det_gemm").forward(
        o, w_a, w_b, case.cos.cuda(), case.sin.cuda()
    ).y
    y.float().sum().backward()
    assert o.grad is not None and torch.isfinite(o.grad).all()
    assert w_a.grad is not None and torch.isfinite(w_a.grad).all()
