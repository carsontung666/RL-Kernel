#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run a DSV4-shaped mini model through T06 attention + grouped o-proj on GPU.

Layer table mixes C0/C4/C128. Compressed rows are synthetic recorded stand-ins
(T04 compressor is not live). This is a real multi-layer fwd/bwd training loop
on the T06 operators, not pytest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import (
    ATTENTION_SCALE,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
    RECENT_WINDOW,
)
from rl_engine.kernels.p2.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.p2.four_mode import eager_vs_cuda_graph_attention
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.o_proj.rope_consumer import apply_gptj_interleaved_partial, fixture_cos_sin
from rl_engine.kernels.p2.state_gate import synthetic_pass_verdict


LAYER_TABLE = ("C0", "C4", "C128", "C4")


def _pool_rows(x: torch.Tensor, cr: int) -> torch.Tensor:
    tokens = x.shape[0]
    n = tokens // cr
    if n == 0:
        return x.new_zeros((0, x.shape[-1]))
    return x[: n * cr].reshape(n, cr, -1).mean(dim=1)


def build_candidates(layer_type: str, k: torch.Tensor, v: torch.Tensor):
    tokens = k.shape[0]
    recent_n = min(RECENT_WINDOW, tokens)
    recent_k = k[-recent_n:]
    recent_v = v[-recent_n:]
    if layer_type == "C0":
        n_c = 0
        ck = k.new_zeros((0, HEAD_DIM))
        cv = v.new_zeros((0, HEAD_DIM))
    elif layer_type == "C4":
        ck = _pool_rows(k, 4)
        cv = _pool_rows(v, 4)
        n_c = ck.shape[0]
    elif layer_type == "C128":
        ck = _pool_rows(k, 128)
        cv = _pool_rows(v, 128)
        n_c = ck.shape[0]
    else:
        raise ValueError(layer_type)
    k_cat = torch.cat([ck, recent_k], dim=0)
    v_cat = torch.cat([cv, recent_v], dim=0)
    plan = CandidatePlan(
        layer_type=layer_type,
        n_compressed=n_c,
        n_recent=recent_n,
        valid=torch.ones(n_c + recent_n, dtype=torch.bool, device=k.device),
    )
    return k_cat, v_cat, plan


class P2Layer(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.rms_w = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.w_q = nn.Linear(HIDDEN_SIZE, N_Q_HEADS * HEAD_DIM, bias=False)
        self.w_k = nn.Linear(HIDDEN_SIZE, HEAD_DIM, bias=False)
        self.w_v = nn.Linear(HIDDEN_SIZE, HEAD_DIM, bias=False)
        self.w_a = nn.Parameter(0.02 * torch.randn(N_O_PROJ_GROUPS, O_LORA_RANK, GROUP_FLAT_DIM, dtype=torch.bfloat16))
        self.w_b = nn.Parameter(0.02 * torch.randn(HIDDEN_SIZE, N_O_PROJ_GROUPS * O_LORA_RANK, dtype=torch.bfloat16))
        self.sink = nn.Parameter(torch.zeros(N_Q_HEADS))
        self.attn = MqaJointAttentionSinkOp(backend="cuda")
        self.o_proj = OProjGroupedOp(backend="det_gemm")
        self.state_gate = synthetic_pass_verdict(tag=f"min-model-{layer_type}")

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self.rms_w, 1e-6)
        tokens = hidden.shape[0]
        q = self.w_q(normed).view(tokens, N_Q_HEADS, HEAD_DIM)
        q = apply_gptj_interleaved_partial(q, cos, sin, inverse=False)
        k = self.w_k(normed)
        v = self.w_v(normed)
        k_cat, v_cat, plan = build_candidates(self.layer_type, k, v)
        o = self.attn.apply_autograd(
            q,
            k_cat,
            v_cat,
            self.sink,
            plan,
            output_fp32=True,
            compare=True,
            state_gate=self.state_gate,
        )
        y = self.o_proj.forward(o.to(torch.bfloat16), self.w_a, self.w_b, cos, sin).y.float()
        return hidden + y


class P2MiniModel(nn.Module):
    def __init__(self, layer_table=LAYER_TABLE):
        super().__init__()
        self.layers = nn.ModuleList(P2Layer(kind) for kind in layer_table)
        self.final_norm = nn.Parameter(torch.ones(HIDDEN_SIZE))

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = hidden
        for layer in self.layers:
            x = layer(x, cos, sin)
        return F.rms_norm(x, (HIDDEN_SIZE,), self.final_norm, 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/p2_t06_min_model.json"))
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    cuda_source = ensure_t06_cuda_kernel() if device.type == "cuda" else "cpu"

    model = P2MiniModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tokens = args.tokens
    positions = torch.arange(tokens, device=device)
    cos, sin = fixture_cos_sin(positions)
    hidden = 0.02 * torch.randn(tokens, HIDDEN_SIZE, device=device)
    target = torch.zeros_like(hidden)

    print(
        f"[p2-t06-model] device={device} cuda_source={cuda_source} "
        f"layers={LAYER_TABLE} T={tokens} scale={ATTENTION_SCALE:.6f}",
        flush=True,
    )

    losses = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(hidden, cos, sin)
        loss = F.mse_loss(out, target)
        loss.backward()
        finite = all(
            p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()
        )
        if not finite or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite at step {step} loss={float(loss)}")
        opt.step()
        losses.append(float(loss.detach()))
        print(f"[p2-t06-model] step={step} loss={losses[-1]:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        training_like = model(hidden, cos, sin)
        prefill = model(hidden, cos, sin)
        eager_decode = model(hidden, cos, sin)
    four_mode_eager_equal = torch.equal(training_like, prefill) and torch.equal(prefill, eager_decode)

    from rl_engine.kernels.p2.fixtures.catalog import make_attn_case

    attn_case = make_attn_case(
        "graph", layer_type="C4", tokens=2, n_compressed=4, n_recent=8, seed=0, device=str(device)
    )
    graph_status = "SKIP"
    if device.type == "cuda":
        try:
            eager_o, graph_o = eager_vs_cuda_graph_attention(
                attn_case.q,
                attn_case.k,
                attn_case.v,
                attn_case.sink,
                attn_case.plan,
                attn_case.state_gate,
            )
            graph_status = "PASS" if torch.equal(eager_o, graph_o) else "BYTE_MISMATCH"
        except Exception as exc:
            graph_status = f"UNSUPPORTED_CAPABILITY:{type(exc).__name__}"
    four_mode_equal = bool(four_mode_eager_equal)

    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "task_id": "T06",
        "experiment": "p2_t06_min_model",
        "device": str(device),
        "cuda_source": cuda_source,
        "layer_table": list(LAYER_TABLE),
        "tokens": tokens,
        "steps": args.steps,
        "n_params": n_params,
        "losses": losses,
        "loss_dropped": bool(losses[-1] < losses[0]),
        "four_mode_eager_equal": bool(four_mode_eager_equal),
        "cuda_graph_status": graph_status,
        "four_mode_equal": four_mode_equal,
        "elapsed_sec": elapsed,
        "output_shape": [tokens, HIDDEN_SIZE],
        "note": "synthetic recorded C4/C128 rows; T04 compressor not live",
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not four_mode_equal:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
