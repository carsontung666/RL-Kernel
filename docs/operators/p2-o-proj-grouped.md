# P2 Grouped Output Projection

T06-B operator: out-of-place inverse GPT-J partial RoPE, then 8-group `wo_a` / `wo_b`.

## Entry Point

```python
from rl_engine.kernels.p2 import OProjGroupedOp

op = OProjGroupedOp(backend="oracle")
result = op.forward_fp32(o, w_a, w_b, cos, sin)
y = result.y  # [T, 4096]
dO, dW_a, dW_b = op.backward(dY, result.saved, cos, sin)
```

## Backends

| Backend | GEMM | RoPE |
| --- | --- | --- |
| oracle | sequential-K FP32 | T02 `rope_gptj_interleaved_partial` if present, else T06-private apply on caller tables |
| det_gemm | `_C.det_gemm_fwd_rhs_transposed` (fixed-K; Ampere naive, not SM90) | same consumer |
| torch_fp32 | declared FP32 matmul, TF32 off | same consumer |
| auto | CUDA requires DetGemm symbols; CPU uses oracle | no silent mix |

T06 does not register a second public RoPE. NeoX / in-place mutation fail closed (`INVALID_ROPE_VARIANT`).

## Tensor Contract

```
O_tilde = RoPE_inverse(O)          # range [448:512], NoPE copied, out-of-place
8 groups × 8 heads, flatten to 4096
Z_g = O_tilde_g @ W_a[g]^T         # W_a: [8, 1024, 4096]
Z   = concat(Z_0..Z_7)             # [T, 8192]
Y   = Z @ W_b^T                    # W_b: [4096, 8192]
```

## Tests

```bash
python -m pytest tests/p2/test_o_proj_grouped.py tests/p2/test_o_proj_grouped_negative.py tests/p2/test_o_proj_det_gemm.py -q
```
