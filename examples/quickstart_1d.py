"""Minimal 1D complex example for the numpy/sigpy backend."""

from __future__ import annotations

import numpy as np

from nufftsi import SpectralNUFFTInterpolator


def main() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 1.0, 128)).astype(np.float32)
    values = (
        np.sin(2 * np.pi * 3 * t) + 0.5j * np.cos(2 * np.pi * 5 * t)
    ).astype(np.complex64)

    interp = SpectralNUFFTInterpolator(
        shape=(256,),
        regularization_lambda=1e-2,
        n_iter=200,
        wavelet_level=3,
        init_method="adjoint",
    ).fit(t, values)

    grid = interp.reconstruct_grid()
    pred = interp(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    print(f"grid shape={grid.shape}, dtype={grid.dtype}")
    print(f"data_nrmse={interp.data_nrmse_:.4f}, lambda={interp.selected_lambda_}")
    print(f"pred={pred}")


if __name__ == "__main__":
    main()
