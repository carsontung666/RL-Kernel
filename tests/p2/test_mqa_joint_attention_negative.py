# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import make_attn_case


def test_wrong_head_dim_fails():
    q = torch.randn(1, 64, 128)
    k = torch.randn(2, 128)
    v = torch.randn(2, 128)
    sink = torch.zeros(64)
    plan = CandidatePlan(layer_type="C0", n_compressed=0, n_recent=2, valid=torch.ones(2, dtype=torch.bool))
    with pytest.raises(P2FailClosedError) as exc:
        MqaJointAttentionSinkOp(backend="oracle").forward_fp32(q, k, v, sink, plan)
    assert exc.value.status in {P2Status.SCHEMA_MISMATCH}


def test_dequant_manifest_is_unsupported():
    from rl_engine.kernels.p2.candidate_plan import CandidatePlan

    with pytest.raises(P2FailClosedError) as exc:
        CandidatePlan(
            layer_type="C0",
            n_compressed=0,
            n_recent=1,
            valid=torch.ones(1, dtype=torch.bool),
            dequant_manifest={"fmt": "fp8"},
        )
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY


def test_cuda_backend_without_kernel_is_unsupported():
    case = make_attn_case("neg", layer_type="C0", tokens=1, n_compressed=0, n_recent=1)
    op = MqaJointAttentionSinkOp(backend="cuda")
    with pytest.raises(P2FailClosedError) as exc:
        op.forward_fp32(case.q, case.k, case.v, case.sink, case.plan)
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY


def test_atomic_reduction_flag():
    plan = CandidatePlan(
        layer_type="C0",
        n_compressed=0,
        n_recent=1,
        valid=torch.ones(1, dtype=torch.bool),
        atomic_reduction=True,
    )
    k = torch.zeros(1, 512)
    v = torch.zeros(1, 512)
    with pytest.raises(P2FailClosedError) as exc:
        plan.validate_for_kv(k, v)
    assert exc.value.status is P2Status.FORBIDDEN_ATOMIC_REDUCTION


def test_missing_global_visibility():
    plan = CandidatePlan(
        layer_type="C4",
        n_compressed=1,
        n_recent=1,
        valid=torch.ones(2, dtype=torch.bool),
        global_visible=False,
    )
    with pytest.raises(P2FailClosedError) as exc:
        plan.validate_for_kv(torch.zeros(2, 512), torch.zeros(2, 512))
    assert exc.value.status is P2Status.MISSING_GLOBAL_VISIBILITY
