# Free-Energy-Mixer

## Quick Start

```python
# Minimal softmax prior layer (copy-paste ready). Assumes fem.py is available:
#   from fem import FreeEnergyMixer, FreeEnergyMixerConfig

import torch
from fem import FreeEnergyMixer, FreeEnergyMixerConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32

B, T, D, H = 2, 128, 512, 8
x = torch.randn(B, T, D, device=device, dtype=dtype)

# Softmax prior, paper-aligned budget:
#   fem_ratio=0.5  -> fem_dim ≈ D/2
#   p_t_to_fem_ratio=4.0 -> |Q|+|K| = 4*fem_dim = 2*D (matches standard attention)
cfg = FreeEnergyMixerConfig(
    n_embd=D, n_head=H,
    prior_type="softmax",
    softmax_backend="auto",

    # Recommended sizing that matches the parameter budget of standard attention:
    #   (A) fem_ratio = 0.5, p_t_to_fem_ratio = 4.0
    #       → fem_dim ≈ D/2; total (Q+K) ≈ 2D  (same Q+K size as standard attention)
    #   (B) fem_ratio = 2/3, p_t_to_fem_ratio = 2.0
    #       → fem_dim ≈ 2D/3; total (Q+K) ≈ 4D/3  (more balanced: wider value path, lighter Q/K)
    fem_ratio=2/3,            # value path width ≈ 2D/3
    p_t_to_fem_ratio=2.0,     # total Q+K size relative to fem_dim

    use_temperature=True,     # enable FEM free-energy branch (β + LSE)
    use_lse=True,             # keep LSE branch when β=1
    use_outer_gate=True,      # multiplicative outer gate
    use_conv=True,            # lightweight time-decay conditioning (TDC)
    use_rope=True             # apply RoPE to Q/K
)

layer = FreeEnergyMixer(cfg).to(device=device, dtype=dtype)
with torch.no_grad():
    y = layer(x)     # (B,T,D)
print(y.shape)

```

# More Configurations and Different Selection Prior Types

```python
# minimal_usage_fem.py
# Minimal, ready-to-run usage guide for FreeEnergyMixer
# -----------------------------------------------------
# Assumes your model code lives in `fem.py` alongside this file:
#   from fem import FreeEnergyMixer, FreeEnergyMixerConfig
#
# This script shows how to instantiate FEM for different priors,
# with paper-aligned parameter budgets and safe defaults.
#
# Run:
#   python minimal_usage_fem.py

import torch
from fem import FreeEnergyMixer, FreeEnergyMixerConfig

# -----------------------------
# Environment & synthetic input
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32  # bfloat16 preferred on CUDA

B, T, D, H = 2, 128, 512, 8          # (batch, seq, hidden, heads) — D must be divisible by H
x = torch.randn(B, T, D, device=device, dtype=dtype)
# Optional padding mask (1/True = keep token, 0/False = pad token)
mask = None  # or: (torch.rand(B, T, device=device) > 0.1)

# --------------------------------------------------------------------
# Helper: build common config fields with concise one-line comments
# --------------------------------------------------------------------
def base_cfg(
    prior_type: str,
    fem_ratio: float,
    p_t_to_fem_ratio: float,
    qk_dim: int | None = None,
    use_temperature: bool = True,
    use_lse: bool = True,
    use_outer_gate: bool = True,
    use_rope: bool = True,
    use_conv: bool = True,
):
    """
    prior_type           : 'softmax' | 'linear' | 'gla' | 'rnn_softmax' | 'mamba'
    fem_ratio            : fem_dim = round(n_embd * fem_ratio) aligned to n_head (paper default: 0.5)
    p_t_to_fem_ratio     : total param size for (Q+K) relative to fem_dim (paper default: 4.0 for attn)
    qk_dim               : override per-token Q/K dim; if None, we use qk_dim = 0.5 * p_t_to_fem_ratio * fem_dim
    use_temperature      : enable β and the log-sum-exp branch (recommended for FEM)
    use_lse              : if use_temperature=False, enable LSE-only branch (β = 1)
    use_outer_gate       : multiplicative outer gate g∈(0,1) (controlled by a linear head)
    use_rope             : apply RoPE to (Q,K) — useful for softmax/linear/GLA/rnn_softmax
    use_conv             : lightweight time-decay conditioning (TDC) to modulate Q/K/V after parameterization
    """
    return FreeEnergyMixerConfig(
        # Core sizes
        n_embd=D,                    # model width (hidden size)
        n_head=H,                    # number of heads
        fem_dim=None,                # let fem_ratio decide fem_dim (kept head-aligned)
        fem_ratio=fem_ratio,         # fem_dim ≈ D * fem_ratio (rounded to multiple of n_head)

        # Parameterization sizes
        qk_dim=qk_dim,               # per-token dim of Q or K; if None we derive from p_t_to_fem_ratio
        p_t_to_fem_ratio=p_t_to_fem_ratio,  # total (Q+K) = p_t_to_fem_ratio * fem_dim

        # Prior & backend
        prior_type=prior_type,       # choose the prior family
        softmax_backend="auto",      # 'auto' uses SDPA (supports v_dim != qk_dim); 'flash' if v_dim==qk_dim

        # Runtime
        dropout=0.0,                 # set >0 during training
        causal=True,                 # causal masking (set False for encoder use-cases)
        bias=True,                   # include bias in linear layers

        # FEM options
        use_temperature=use_temperature,
        use_lse=use_lse,
        use_outer_gate=use_outer_gate,
        value_logexp_cap=40.0,       # numeric stabilization cap on |β·v| via pre-clip
        eps=1e-6,                    # numeric epsilon

        # RoPE & TDC
        use_rope=use_rope,
        rope_theta=10000.0,
        use_conv=use_conv,
        conv_hidden=64,              # TDC hidden width
        conv_norm_first=True,
        conv_bidirectional=False,    # set True if you prefer a bi-directional TDC feature

        # Mamba (unused unless prior_type='mamba')
        ssm_rank=16,
        mamba_normalize=True,
    )

# -------------------------------------------------------------
# Presets per prior (paper-aligned parameter budgets & flags)
# -------------------------------------------------------------
# 1) Softmax Attention — paper default: fem_ratio=0.5, p_t_to_fem_ratio=4.0
cfg_softmax = base_cfg(
    prior_type="softmax",
    fem_ratio=0.5,             # fem_dim = D/2
    p_t_to_fem_ratio=4.0,      # |Q|+|K| = 4 * fem_dim = 2 * D (matches standard attention)
    qk_dim=None,               # let it be derived, resulting in qk_dim ≈ D
    use_temperature=True,      # enable FEM free-energy branch
    use_lse=True,
    use_outer_gate=True,
    use_conv=True,
    use_rope=True,
)

# 2) Linear Attention (kernel) — fem_ratio=0.5, p_t_to_fem_ratio=4.0
cfg_linear = base_cfg(
    prior_type="linear",
    fem_ratio=0.5,
    p_t_to_fem_ratio=4.0
)

# 3) Gated Linear Attention (GLA) — fem_ratio=0.5, p_t_to_fem_ratio=4.0
cfg_gla = base_cfg(
    prior_type="gla",
    fem_ratio=0.5,
    p_t_to_fem_ratio=4.0
)

# 4) Linear RNN (AFT-like) — fem_ratio=1, p_t_to_fem_ratio=2.0
cfg_rnn = base_cfg(
    prior_type="rnn_softmax",
    fem_ratio=1,
    p_t_to_fem_ratio=2.0,
)

# 5) (Optional) Mamba SSM — FEM still applies on value path; Q/K dims are irrelevant
cfg_mamba = FreeEnergyMixerConfig(
    n_embd=D, n_head=H,
    fem_dim=None, fem_ratio=0.5,     # keep value path ~D/2
    prior_type="mamba",
    dropout=0.0, causal=True, bias=True,
    use_temperature=True, use_lse=True, use_outer_gate=True,
    use_rope=False,                   # RoPE not used for SSM-style prior
    use_conv=True, conv_hidden=64,
    ssm_rank=16, mamba_normalize=True,
)

# --------------------------
# Build and run single pass
# --------------------------
def run_one(cfg: FreeEnergyMixerConfig, name: str):
    fem = FreeEnergyMixer(cfg).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        y = fem(x, attention_mask=mask)
    print(f"{name:18s} -> out: {tuple(y.shape)}, dtype: {y.dtype}")

if __name__ == "__main__":
    print(f"[env] device={device} dtype={dtype}, B={B}, T={T}, D={D}, H={H}")
    run_one(cfg_softmax, "softmax")
    run_one(cfg_linear,  "linear")
    run_one(cfg_gla,     "gla")
    run_one(cfg_rnn,     "rnn_softmax")
    run_one(cfg_mamba,   "mamba")

    # Tips:
    # - To switch softmax to FlashAttention, set cfg.softmax_backend="flash" and ensure:
    #     1) CUDA + flash-attn installed; 2) no temperature/LSE concat; 3) qk_dim == fem_dim (v_dim == qk_dim).
    # - To disable the free-energy branch and use pure μ:
    #     use_temperature=False, use_lse=False
    # - To disable TDC or RoPE:
    #     use_conv=False, use_rope=False
```
