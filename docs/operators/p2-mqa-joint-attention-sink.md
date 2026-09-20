# P2 MQA Joint Attention with Sink

T06 operator for DeepSeek-V4 CSA/HCA attention. One softmax denominator over compressed prefix + recent-128 + sink. Sink has no V.

## Entry Point

```python
from rl_engine.kernels.p2 import MqaJointAttentionSinkOp, require_state_gate

op = MqaJointAttentionSinkOp(backend="oracle")  # or "cuda" / "auto"
result = op.forward_fp32(q, k, v, sink, plan, compare=True, state_gate=verdict, debug=True)
o = result.o  # [T, 64, 512]
```

`compare=True` requires a PASS state-byte verdict. State failure is `STATE_BYTES_MISMATCH` and must not be attributed to attention.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CPU/GPU sequential oracle | `MqaJointAttentionSinkOp(backend="oracle")` | none | WS1 golden; sequential `d`/`j`/`h` trees |
| CUDA reference | `MqaJointAttentionSinkOp(backend="cuda")` | `_C.mqa_joint_attention_sink_forward/backward` | Required on CUDA tensors; missing kernel is `UNSUPPORTED_CAPABILITY` |
| auto | CUDA tensors require the compiled kernel; CPU tensors use oracle | — | GPU auto never falls back to oracle |

## Tensor Contract

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q` | `[T, 64, 512]` | fp32/bf16/fp16 |
| `k`, `v` | `[N, 512]` | same as q (MQA packed) |
| `sink` | `[64]` or `[T, 64]` | fp32 |
| `plan.valid` | `[N]` bool | invalid → `-inf` logits |
| `plan.n_compressed` | int | prefix length; rest is recent |
| output `O` | `[T, 64, 512]` | declared output dtype |
| `dKV` | `[N, 512]` | `dK + dV` |

Constants: `Hq=64`, `Hkv=1`, `D=512`, `scale=512^-0.5`. Candidate order is compressed prefix then recent-128. Softmax is one denominator: `Z = e_sink + sum_j e[j]`.

## Accuracy

The oracle is the sequential-tree FP32 ground truth (TF32/autocast off). Same CUDA launch is bitwise equal. CUDA vs CPU oracle uses `CUDA_VS_ORACLE_FWD_ATOL=1e-7` / `BWD_ATOL=1e-5` (`expf` vs `torch.exp`), not byte-equal. Debug mode exposes `m/Z/p/p_sink` without changing `O`. Actual unroll is `pragma_unroll_8_sequential_d_0_511`.

Forbidden (fail-closed, not numeric error): two-softmax merge, Split-KV, sink-as-V, atomic partials, missing state gate.

## Tests

```bash
python -m pytest tests/p2/test_mqa_joint_attention_oracle.py tests/p2/test_mqa_joint_attention_negative.py tests/p2/test_candidate_plan.py tests/p2/test_state_gate.py -q
python -m pytest tests/p2/test_mqa_joint_attention_cuda.py -q
```
