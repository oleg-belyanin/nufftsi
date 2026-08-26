# Spectral NUFFT Interpolator API

Recover a regular complex 1D/2D/3D grid from irregular samples. Two backends share the same estimator and a similar interpolator-style API (`fit` → `reconstruct_grid` / `__call__`):

| Module | I/O | NUFFT | Wavelet L1 | Device |
|--------|-----|-------|------------|--------|
| `nufftsi` / `nufftsi.numpy` | NumPy | `sigpy.linop.NUFFT` | `sigpy.linop.Wavelet` | CPU |
| `nufftsi.torch` | `torch.Tensor` | torchkbnufft | ptwt | CPU or CUDA |

Install: `pip install "nufftsi[sigpy]"` or `pip install "nufftsi[torch]"`.

## Problem

Coordinates must be in `[0, 1)` per axis. Observations `values[j]` are complex. The grid `x` has shape `shape`; its centered spectrum is

```text
c = fftshift(fftn(x)) / prod(shape)
```

The fit minimizes mean `|NUFFT(c)(points) - values|^2` plus `lambda * (||W real(c)||_1 + ||W imag(c)||_1)`. The wavelet prior is on the spectrum, not on `x`.

## Parameters

```python
SpectralNUFFTInterpolator(
    shape,                      # (N,), (H, W), or (D, H, W)
    regularization_lambda=1e-2,
    wavelet="db4",
    wavelet_level=None,
    n_iter=1000,
    lr=1.5e-2,
    init_method="interpolate",  # numpy default; torch default is "adjoint"
    device="cpu",
    random_state=42,
    lambda_candidates=None,     # if set, pick lambda on a validation split
    validation_fraction=0.2,
    validation_iter=None,
)
```

| `init_method` | NumPy / sigpy | Torch |
|---------------|---------------|-------|
| `"zeros"` | yes | yes |
| `"adjoint"` | yes | yes |
| `"interpolate"` | yes (PCHIP / `griddata`) | no |

Point shapes: `(M,)` or `(M, 1)` in 1D; `(M, 2)` / `(M, 3)` in 2D/3D.

## NumPy

```python
import numpy as np
from nufftsi import SpectralNUFFTInterpolator

t = np.sort(np.random.default_rng(0).uniform(0.0, 1.0, 128)).astype(np.float32)
values = (np.sin(2 * np.pi * 3 * t) + 0.5j * np.cos(2 * np.pi * 5 * t)).astype(np.complex64)

interp = SpectralNUFFTInterpolator(
    shape=(256,),
    regularization_lambda=1e-2,
    n_iter=400,
    init_method="adjoint",
).fit(t, values)

grid = interp.reconstruct_grid()
pred = interp(np.array([0.1, 0.2, 0.3], dtype=np.float32))
```

After `fit`: `grid_`, `spectrum_`, `selected_lambda_`, `data_nrmse_`. Prefer `reconstruct_grid()` over reading `grid_` directly.

2D:

```python
points = rng.uniform(0.0, 1.0, (2000, 2)).astype(np.float32)
interp = SpectralNUFFTInterpolator(shape=(64, 64), n_iter=1200, init_method="adjoint")
interp.fit(points, values)
```

`fit_spectral_nufft_interpolator(points, values, shape=...)` is the same as constructing the class and calling `fit`.

## Torch

Inputs and outputs are `torch.Tensor` (`grid_tensor_`, `spectrum_tensor_`). There are no NumPy `grid_` / `spectrum_` attributes. Use `init_method="adjoint"` or `"zeros"`.

```python
import torch
from nufftsi.torch import SpectralNUFFTInterpolator

t = torch.linspace(0.0, 0.95, 128)
values = torch.sin(2 * torch.pi * 3 * t).to(torch.complex64)
values = values + 0.5j * torch.cos(2 * torch.pi * 5 * t)

interp = SpectralNUFFTInterpolator(
    shape=(256,),
    regularization_lambda=1e-2,
    n_iter=400,
    init_method="adjoint",
).fit(t, values)

grid_t = interp.reconstruct_grid()
pred_t = interp(torch.tensor([0.1, 0.2, 0.3]))
```

## Limits

- Iterative; slower than PCHIP / `griddata` on easy problems.
- One signal at a time (no batch API).
- NumPy backend is CPU-only.
- `lambda` matters; use `lambda_candidates` or a prior when you care about the grid, not only data fit.
