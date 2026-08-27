# nufftsi

Spectral **NUFFT** interpolator for irregular **complex** 1D/2D/3D samples.

Recover a regular complex grid from nonuniform points by fitting a centered spectrum with wavelet $`\ell_1`$ regularization on $`\Re c`$ and $`\Im c`$:

```math
\min_x \; \mathrm{mean}\bigl(|A\mathcal{F}(x)-y|^2\bigr)
+ \lambda\bigl(\|W\Re c\|_1 + \|W\Im c\|_1\bigr),\quad
c=\mathrm{fftshift}(\mathrm{fftn}(x))/\prod\mathrm{shape}.
```

Two backends share the same public API:

| Import | I/O | Stack |
|--------|-----|-------|
| `nufftsi` / `nufftsi.numpy` | NumPy | **sigpy** NUFFT + Wavelet (no PyTorch) |
| `nufftsi.torch` | `torch.Tensor` | torch + **torchkbnufft** + **ptwt** |

## Install

```bash
# NumPy / sigpy path (recommended default)
pip install "nufftsi[sigpy]"

# Torch path
pip install "nufftsi[torch]"

# Both + pytest
pip install "nufftsi[dev]"
```

Alpha (v0.1). Install from GitHub until the package is published to PyPI:

```bash
pip install "nufftsi[sigpy] @ git+https://github.com/oleg-belyanin/nufftsi.git"
```

## Quick start (NumPy)

```python
import numpy as np
from nufftsi import SpectralNUFFTInterpolator

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0.0, 1.0, 128)).astype(np.float32)
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

Coordinates must lie in `[0, 1)` per axis. See [docs/api.md](docs/api.md) for 2D/3D, torch API, and parameters.

## When it helps

- Complex values on irregular points
- Gaps / clustering where local interpolators struggle
- Spectrum well described by wavelet sparsity on $`\Re c,\Im c`$

Dense 1D cases may still favor PCHIP; 2D complex with gaps is where spectral NUFFT typically wins over `griddata`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
