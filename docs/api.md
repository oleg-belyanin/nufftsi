# Spectral NUFFT Interpolator API

Recover a regular complex 1D/2D/3D grid from irregular samples. The public API is exposed through two modules:

- `nufftsi.numpy` - NumPy inputs/outputs with a **sigpy** backend (`linop.NUFFT` + `linop.Wavelet`, internally backed by PyWavelets).
- `nufftsi.torch` - `torch.Tensor` inputs/outputs with a **torch** + `torchkbnufft` + `ptwt` backend.

Both interfaces are intentionally interpolator-like: create the object with grid parameters, fit it with `fit(points, values)`, then either call it on new irregular points or recover the full regular grid with `reconstruct_grid()`.

Related docs:

- backend comparison: `docs/backend_parity.md`
- paper draft: `paper/`

## Backends

| Module | I/O | NUFFT | Wavelet L1 | Optimizer | GPU |
|--------|-----|-------|------------|-----------|-----|
| `..._numpy` | `numpy` | `sigpy.linop.NUFFT` | `sigpy.linop.Wavelet` (pywt) | NumPy Adam | no (`device="cpu"`) |
| `..._torch` | `torch.Tensor` | `torchkbnufft` | `ptwt` | `torch.optim.Adam` | yes, when CUDA is available |

Both solve the same spectral objective. Notes on backend parity live in `docs/backend_parity.md` (lab-only).

## Method idea

Assume the irregular measurements are

```text
points[j] = coordinate in [0, 1)^D
values[j] = complex signal value at that point
```

The goal is to recover a complex signal `x` on a regular grid `shape`. Its spectrum is defined as

```text
c = fftshift(fftn(x)) / prod(shape)
```

The observations are predicted through NUFFT:

```text
values[j] ~= sum_k c[k] * exp(+2pi i <k, points[j]>)
```

The optimization problem is

```text
mean(|NUFFT(c)(points) - values|^2)
+ lambda * (||W real(c)||_1 + ||W imag(c)||_1)
```

Important: in this setup, both `ptwt` and `sigpy.linop.Wavelet` / pywt operate on real-valued arrays, so `||Wc||_1` is computed separately for the real and imaginary parts of the spectrum. This is the same in 1D, 2D, and 3D.

This is a **spectral** prior (`W` applied to `c`), not the image-domain `||W x||_1` used in common sigpy MRI examples such as `L1WaveletRecon`.

## Dependency installation

Recommended research environment for this interpolator:

```bash
python -m venv .venv && source .venv/bin/activate
```

Minimum dependencies for the NumPy branch:

```bash
python -m pip install sigpy scipy
# PyWavelets and numba come in transitively with sigpy
```

For the torch branch and backend comparisons:

```bash
python -m pip install torch torchkbnufft ptwt
```

## Quick start: 1D complex (NumPy / sigpy)

```python
import numpy as np

from nufftsi.numpy import SpectralNUFFTInterpolator

rng = np.random.default_rng(0)

# Irregular coordinates in [0, 1)
t = np.sort(rng.uniform(0.0, 1.0, 128)).astype(np.float32)

# Complex observations
values = (
    np.sin(2 * np.pi * 3 * t)
    + 0.5j * np.cos(2 * np.pi * 5 * t)
).astype(np.complex64)

interp = SpectralNUFFTInterpolator(
    shape=(256,),
    regularization_lambda=1e-2,
    n_iter=1000,
    init_method="interpolate",  # or "adjoint" / "zeros"
).fit(t, values)

# Reconstructed regular grid
grid = interp.reconstruct_grid()

# Values at new irregular points
new_t = np.array([0.1, 0.2, 0.3], dtype=np.float32)
pred = interp(new_t)
```

## Torch module API

The `nufftsi.torch` module accepts and returns only `torch.Tensor` objects:

- input `points` and `values` must be tensors;
- `reconstruct_grid()` returns a `torch.Tensor`;
- `predict(...)` / `__call__(...)` returns a `torch.Tensor`;
- internal attributes are `grid_tensor_` and `spectrum_tensor_`;
- NumPy attributes `grid_` and `spectrum_` do not exist in this module.

For a fully tensor-native torch workflow, use `init_method="adjoint"` or `init_method="zeros"`.

- `"adjoint"` - fast initialization via `KbNufftAdjoint` (usually better than zeros).
- `"zeros"` - initializes from a zero complex grid.
- `"interpolate"` - scipy/NumPy-based; **only** available in `nufftsi.numpy`.

```python
import torch

from nufftsi.torch import SpectralNUFFTInterpolator

t = torch.linspace(0.0, 0.95, 128)
values = torch.sin(2 * torch.pi * 3 * t).to(torch.complex64)
values = values + 0.5j * torch.cos(2 * torch.pi * 5 * t)

interp = SpectralNUFFTInterpolator(
    shape=(256,),
    regularization_lambda=1e-2,
    n_iter=1000,
    init_method="adjoint",
).fit(t, values)

grid_t = interp.reconstruct_grid()  # torch.Tensor, complex64
pred_t = interp(torch.tensor([0.1, 0.2, 0.3]))
```

## Quick start: 2D complex (NumPy / sigpy)

```python
import numpy as np

from nufftsi.numpy import SpectralNUFFTInterpolator

rng = np.random.default_rng(1)

points = rng.uniform(0.0, 1.0, (2000, 2)).astype(np.float32)
x = points[:, 0]
y = points[:, 1]

values = (
    np.sin(2 * np.pi * 2 * x)
    + 1j * np.cos(2 * np.pi * 3 * y)
).astype(np.complex64)

interp = SpectralNUFFTInterpolator(
    shape=(64, 64),
    regularization_lambda=1e-2,
    n_iter=1200,
    wavelet="db4",
    wavelet_level=3,
    init_method="adjoint",
).fit(points, values)

image = interp.reconstruct_grid()  # shape == (64, 64), complex
sampled = interp(points[:10])      # shape == (10,)
```

## Torch module: 2D complex

```python
import torch

from nufftsi.torch import SpectralNUFFTInterpolator

device = "cpu"

points = torch.rand(2000, 2, device=device)
x = points[:, 0]
y = points[:, 1]

values = torch.sin(2 * torch.pi * 2 * x).to(torch.complex64)
values = values + 1j * torch.cos(2 * torch.pi * 3 * y)

interp = SpectralNUFFTInterpolator(
    shape=(64, 64),
    regularization_lambda=1e-2,
    n_iter=1200,
    wavelet="db4",
    wavelet_level=3,
    init_method="adjoint",
    device=device,
).fit(points, values)

image_t = interp.reconstruct_grid()  # torch.Tensor, shape == (64, 64), complex64
sampled_t = interp(points[:10])      # torch.Tensor, shape == (10,)
```

## 3D

The 3D API follows the same pattern:

```python
interp = SpectralNUFFTInterpolator(
    shape=(32, 32, 32),
    regularization_lambda=3e-2,
    n_iter=600,
    wavelet="db2",
    wavelet_level=2,
    init_method="adjoint",
).fit(points_3d, values)
```

`points_3d` must have shape `(M, 3)`. On CPU, 3D is substantially heavier than 2D, so use smaller `shape` and `n_iter` for quick checks.

## Torch module: 3D complex

```python
import torch

from nufftsi.torch import SpectralNUFFTInterpolator

device = "cpu"

points = torch.rand(5000, 3, device=device)
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

values = (
    torch.sin(2 * torch.pi * 2 * x)
    + 0.5 * torch.cos(2 * torch.pi * (y + z))
).to(torch.complex64)
values = values + 1j * torch.cos(2 * torch.pi * (x - 2 * z))

interp = SpectralNUFFTInterpolator(
    shape=(32, 32, 32),
    regularization_lambda=3e-2,
    n_iter=600,
    wavelet="db2",
    wavelet_level=2,
    init_method="adjoint",
    device=device,
).fit(points, values)

volume_t = interp.reconstruct_grid()  # torch.Tensor, shape == (32, 32, 32)
sampled_t = interp(points[:16])       # torch.Tensor, shape == (16,)
```

## Functional API

Each module also exposes a small convenience wrapper:

```python
from nufftsi.numpy import fit_spectral_nufft_interpolator

interp = fit_spectral_nufft_interpolator(
    points,
    values,
    shape=(64, 64),
    regularization_lambda=1e-2,
    n_iter=1000,
)
```

This is equivalent to:

```python
SpectralNUFFTInterpolator(shape=(64, 64), ...).fit(points, values)
```

## `SpectralNUFFTInterpolator` parameters

```python
SpectralNUFFTInterpolator(
    shape,
    regularization_lambda=1e-2,
    wavelet="db4",
    wavelet_level=None,
    n_iter=1000,
    lr=1.5e-2,
    init_method="interpolate",  # NumPy default; torch default = "adjoint"
    device="cpu",
    random_state=42,
    lambda_candidates=None,
    validation_fraction=0.2,
    validation_iter=None,
)
```

Main parameters:

- `shape`: regular grid shape. Supported dimensions are 1D/2D/3D: `(N,)`, `(H, W)`, `(D, H, W)`.
- `regularization_lambda`: regularization strength for `||W real(c)||_1 + ||W imag(c)||_1`.
- `wavelet`: wavelet filter name (`"db4"`, `"db2"`, ...) - implemented via sigpy/pywt in NumPy and via ptwt in torch.
- `wavelet_level`: decomposition level. `None` lets the wavelet library choose.
- `n_iter`: number of Adam iterations for the final fit.
- `lr`: Adam learning rate.
- `init_method`: `"interpolate"` | `"adjoint"` | `"zeros"` (see table below).
- `device`: `"cpu"` only in the NumPy module; `"cpu"` or a CUDA device in torch.
- `lambda_candidates`: if provided, `lambda` is selected using a validation split.
- `validation_fraction`: fraction of observations reserved for lambda selection.
- `validation_iter`: number of iterations used for trial fits during lambda selection.

### `init_method`

| Value | NumPy / sigpy | Torch |
|-------|---------------|-------|
| `"zeros"` | yes | yes |
| `"adjoint"` | yes (`NUFFT` adjoint + scale) | yes (`KbNufftAdjoint` + scale) |
| `"interpolate"` | yes (scipy PCHIP / `griddata`) | no |

## Choosing a module

- `..._numpy` - use this when your surrounding code is NumPy-based and should stay **independent from torch**.
- `..._torch` - use this for a tensor-native pipeline, GPU support, or when torch is already in the environment.
- `SpectralNUFFTInterpolator` and `fit_spectral_nufft_interpolator` keep the same names across modules; only the type contract differs.
- In the torch module, only `init_method="zeros"` or `"adjoint"` are supported.

## Attributes after `fit`

After a successful `fit(...)`, the following are available:

- `grid_`, `spectrum_` - in the NumPy module (`np.ndarray`);
- `grid_tensor_`, `spectrum_tensor_` - in the torch module;
- `selected_lambda_`, `data_nrmse_`, `shape_` - in both.

`spectrum_*` contains centered Fourier coefficients `fftshift(fftn(grid)) / prod(shape)`.

Recommended public access:

```python
grid = interp.reconstruct_grid()
spectrum = interp.spectrum_          # NumPy module
fit_error = interp.data_nrmse_
```

In the torch module:

```python
grid_t = interp.reconstruct_grid()
spectrum_t = interp.spectrum_tensor_
```

## Coordinates

All coordinates must be normalized to `[0, 1)` on every axis.

1D accepts two shapes:

```python
points.shape == (M,)
points.shape == (M, 1)
```

2D/3D:

```python
points.shape == (M, 2)
points.shape == (M, 3)
```

The coordinate order is user-defined: `(x)`, `(x, y)`, `(x, y, z)`. Internally, that order maps directly to the axes of the regular grid `shape`.

Inside the NumPy/sigpy backend, coordinates are converted to sigpy grid units as `coord = -points * shape` (with NUFFT scale `sqrt(prod(shape))` to match torchkbnufft conventions). This is transparent to the user.

## When the method is useful

This method is useful when:

- values are complex;
- points are irregular;
- a regular 1D/2D/3D grid is needed;
- ordinary local interpolation behaves poorly because of gaps, clustering, or noise;
- the spectrum is expected to be well described by wavelet regularization applied to `real(c)` and `imag(c)`.

In experiments:

- in dense 1D cases, PCHIP often remains a strong baseline;
- in 2D complex scenarios, spectral NUFFT + `||Wc||_1` outperformed `griddata`;
- one-shot `NUFFT adjoint + cutoff + IFFT` is usually just a baseline;
- NumPy/sigpy and torch backends achieve similar data fit; see `docs/backend_parity.md`.

Backend comparison:

```bash
python -m venv .venv && source .venv/bin/activate
# see docs/backend_parity.md (lab-only)
```

## Limits

- This is an optimization-based method, not a cheap local interpolator. On CPU, 2D/3D can be noticeably slower than `griddata` (in our benchmarks, the sigpy path is often faster than torch on CPU).
- The complex spectrum is regularized as two real-valued problems: `real(c)` and `imag(c)`.
- Coordinates must be in `[0, 1)`. Normalize physical coordinates first.
- One signal at a time; no batch API.
- NumPy backend: CPU only.
- Lambda selection matters; for production use, prefer a validation split, prior knowledge, or a dedicated benchmark.

## Minimal check

```bash
source .venv (see README)
python -m pytest tests/test_spectral_nufft_interpolator.py -q
```

Expected result:

```text
9 passed
```
