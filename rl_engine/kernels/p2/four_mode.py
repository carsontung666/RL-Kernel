# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Eager vs CUDA-graph decode for T06 attention with static workspace."""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import (
    cuda_forward_into,
    make_cuda_fwd_workspace,
)
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.state_gate import StateGateVerdict, require_state_gate


def eager_vs_cuda_graph_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    state_gate: StateGateVerdict,
) -> tuple[Tensor, Tensor]:
    """Capture a CUDA graph on static addresses and compare to eager bytes."""

    if not q.is_cuda:
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "CUDA-graph four-mode requires CUDA tensors",
        )
    require_state_gate(state_gate)
    plan.validate_for_kv(k, v)

    q_s = q.detach().contiguous()
    k_s = k.detach().contiguous()
    v_s = v.detach().contiguous()
    workspace = make_cuda_fwd_workspace(q_s, k_s, sink, plan.valid, output_fp32=True)

    cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    eager = workspace.out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    graph.replay()
    if not torch.equal(eager, workspace.out):
        raise P2FailClosedError(
            P2Status.BYTE_MISMATCH,
            "CUDA-graph decode bytes differ from eager",
        )
    return eager, workspace.out.clone()
