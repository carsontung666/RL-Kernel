# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Eager vs CUDA-graph decode for T06 attention.

The WS1 materializing kernel allocates scores every launch, so CUDA Graph
capture is fail-closed UNSUPPORTED_CAPABILITY (not a silent eager replay).
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.state_gate import StateGateVerdict


def eager_vs_cuda_graph_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    state_gate: StateGateVerdict,
) -> tuple[Tensor, Tensor]:
    if not q.is_cuda:
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "CUDA-graph four-mode requires CUDA tensors",
        )
    op = MqaJointAttentionSinkOp(backend="cuda")
    static_q = q.detach().clone()
    static_k = k.detach().clone()
    static_v = v.detach().clone()
    static_sink = sink.detach().clone()
    eager = op.forward_fp32(
        static_q, static_k, static_v, static_sink, plan, compare=True, state_gate=state_gate
    )
    static_out = eager.o.detach().clone()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = op.forward_fp32(
                    static_q,
                    static_k,
                    static_v,
                    static_sink,
                    plan,
                    compare=True,
                    state_gate=state_gate,
                )
                static_out.copy_(warm.o)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op.forward_fp32(
                static_q,
                static_k,
                static_v,
                static_sink,
                plan,
                compare=True,
                state_gate=state_gate,
            )
            static_out.copy_(captured.o)
        graph.replay()
    except RuntimeError as exc:
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "WS1 materializing attention allocates per launch; CUDA Graph capture is unsupported "
            f"until T07 static workspace fusion ({type(exc).__name__}: {exc})",
        ) from exc
    if not torch.equal(eager.o, static_out):
        raise P2FailClosedError(
            P2Status.BYTE_MISMATCH,
            "CUDA-graph decode bytes differ from eager",
        )
    return eager.o, static_out
