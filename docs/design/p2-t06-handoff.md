# T06 Handoff Payload

```
task_id: T06
owner: T06 workspace
reviewer: T05 (attention), T02 (o-proj inverse RoPE)
contract_version: p2-task-contract.v1
schema_version: p2.t06.mqa_joint_attention_sink.v1 / p2.t06.o_proj_grouped.v1
foundation_abi: foundation-attention-boundary.v1
target backend: CUDA reference + CPU FP32 sequential oracle
supported shapes: Hq=64 Hkv=1 D=512; T and N from fixtures
supported dtypes: fp32 oracle; fp16/bf16 CUDA I/O with FP32 trees
execution modes: training / prefill / eager / graph share the same arithmetic
actual kernel: sequential d=0..511, sink-then-sequential j, sequential t-then-h for dK/dV
actual tile: none (WS1 materializing reference)
split_kv: disabled
debug path: MqaJointAttentionSinkOp(..., debug=True)
state gate: require_state_gate() before attention compare
input fixtures: tests/p2 + rl_engine/kernels/p2/fixtures/catalog.py (synthetic_recorded)
known unsupported: fused kernels, live P1 replacement, TP>1 real collective, Ascend profile
gpu_slice: CUDA kernel compiles with nvcc 11.8 + g++-11. Same CUDA launch is bitwise. CPU oracle vs CUDA is FP32 ULP (expf vs torch.exp), budget CUDA_VS_ORACLE_FWD_ATOL=1e-7 / BWD_ATOL=1e-5. Backward matches independent torch.softmax autograd.
performance evidence: none (correctness-only; Performance Gate waits on WS1 strict bytes)
downstream: T07 WS2/performance, T01 provider pack
```

## Commands

```bash
# Full T06 recorded-operator verification (CPU + CUDA JIT if _C missing)
python scripts/check_p2_t06.py

python -m pytest tests/p2 -q

# DSV4-shaped mini-model: 4 layers C0/C4/C128/C4, T=128, GPU fwd/bwd
python examples/p2_t06_min_model.py --tokens 128 --steps 3
```

T01 may alias these commands. Do not treat screenshots as evidence.
