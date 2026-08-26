from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sigpy")

from nufftsi.numpy import (
    SpectralNUFFTInterpolator as NumpySpectralNUFFTInterpolator,
    fit_spectral_nufft_interpolator as fit_numpy_spectral_nufft_interpolator,
)


def _torch_stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchkbnufft")
    pytest.importorskip("ptwt")
    from nufftsi.torch import (
        SpectralNUFFTInterpolator as TorchSpectralNUFFTInterpolator,
    )

    return torch, TorchSpectralNUFFTInterpolator


def test_spectral_nufft_interpolator_1d_complex_api() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 1.0, 48)).astype(np.float32)
    values = np.sin(2 * np.pi * 3 * t) + 0.5j * np.cos(2 * np.pi * 5 * t)

    interp = fit_numpy_spectral_nufft_interpolator(
        t,
        values,
        shape=(32,),
        regularization_lambda=1e-2,
        n_iter=8,
        wavelet_level=2,
        init_method="interpolate",
    )

    grid = interp.reconstruct_grid()
    pred = interp(t[:7])

    assert grid.shape == (32,)
    assert pred.shape == (7,)
    assert np.iscomplexobj(grid)
    assert interp.spectrum_ is not None
    assert interp.spectrum_.shape == (32,)
    assert interp.data_nrmse_ is not None
    assert np.isfinite(interp.data_nrmse_)


def test_spectral_nufft_interpolator_2d_complex_api() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 1.0, (80, 2)).astype(np.float32)
    x = points[:, 0]
    y = points[:, 1]
    values = np.sin(2 * np.pi * 2 * x) + 1j * np.cos(2 * np.pi * 3 * y)

    interp = NumpySpectralNUFFTInterpolator(
        shape=(12, 12),
        regularization_lambda=1e-2,
        n_iter=5,
        wavelet="db2",
        wavelet_level=1,
        init_method="interpolate",
    ).fit(points, values)

    grid = interp.reconstruct_grid()
    pred = interp.predict(points[:5])

    assert grid.shape == (12, 12)
    assert pred.shape == (5,)
    assert np.iscomplexobj(grid)
    assert interp.selected_lambda_ == pytest.approx(1e-2)


def test_spectral_nufft_interpolator_numpy_adjoint_init() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 1.0, 40)).astype(np.float32)
    values = (np.sin(2 * np.pi * 2 * t) + 0.3j * np.cos(2 * np.pi * 4 * t)).astype(np.complex64)

    interp = NumpySpectralNUFFTInterpolator(
        shape=(32,),
        regularization_lambda=1e-2,
        n_iter=5,
        wavelet_level=2,
        init_method="adjoint",
    ).fit(t, values)

    grid = interp.reconstruct_grid()
    pred = interp(t[:5])
    assert grid.shape == (32,)
    assert pred.shape == (5,)
    assert np.iscomplexobj(grid)
    assert interp.spectrum_ is not None


def test_spectral_nufft_interpolator_torch_tensor_io() -> None:
    torch, TorchSpectralNUFFTInterpolator = _torch_stack()
    t = torch.linspace(0.0, 0.95, 32)
    values = torch.sin(2 * torch.pi * 3 * t).to(torch.complex64)
    values = values + 0.25j * torch.cos(2 * torch.pi * 5 * t)

    interp = TorchSpectralNUFFTInterpolator(
        shape=(32,),
        regularization_lambda=1e-2,
        n_iter=5,
        wavelet_level=2,
        init_method="zeros",
    ).fit(t, values)

    grid = interp.reconstruct_grid()
    pred = interp(t[:6])

    assert torch.is_tensor(grid)
    assert torch.is_tensor(pred)
    assert grid.shape == (32,)
    assert pred.shape == (6,)
    assert grid.dtype == torch.complex64
    assert not hasattr(interp, "grid_")
    assert not hasattr(interp, "spectrum_")
    assert interp.grid_tensor_ is not None
    assert interp.spectrum_tensor_ is not None


def test_spectral_nufft_interpolator_torch_adjoint_init() -> None:
    torch, TorchSpectralNUFFTInterpolator = _torch_stack()
    t = torch.linspace(0.0, 0.95, 32)
    values = torch.sin(2 * torch.pi * 3 * t).to(torch.complex64)
    values = values + 0.25j * torch.cos(2 * torch.pi * 5 * t)

    interp = TorchSpectralNUFFTInterpolator(
        shape=(32,),
        regularization_lambda=1e-2,
        n_iter=3,
        wavelet_level=2,
        init_method="adjoint",
    ).fit(t, values)

    grid = interp.reconstruct_grid()
    pred = interp(t[:4])

    assert torch.is_tensor(grid)
    assert torch.is_tensor(pred)
    assert grid.shape == (32,)
    assert pred.shape == (4,)
    assert interp.grid_tensor_ is not None
    assert interp.spectrum_tensor_ is not None


def test_numpy_module_rejects_torch_tensor_input() -> None:
    torch, _ = _torch_stack()
    t = torch.linspace(0.0, 0.95, 16)
    values = torch.sin(2 * torch.pi * 2 * t).to(torch.complex64)

    interp = NumpySpectralNUFFTInterpolator(shape=(16,), n_iter=1, wavelet_level=1)

    with pytest.raises(TypeError, match="numpy inputs only"):
        interp.fit(t, values)


def test_torch_module_rejects_numpy_input() -> None:
    torch, TorchSpectralNUFFTInterpolator = _torch_stack()
    t = np.linspace(0.0, 0.95, 16, dtype=np.float32)
    values = np.sin(2 * np.pi * 2 * t).astype(np.complex64)

    interp = TorchSpectralNUFFTInterpolator(shape=(16,), n_iter=1, wavelet_level=1, init_method="zeros")

    with pytest.raises(TypeError, match="torch.Tensor"):
        interp.fit(t, values)


def test_torch_module_rejects_interpolate_init_method() -> None:
    _, TorchSpectralNUFFTInterpolator = _torch_stack()
    with pytest.raises(ValueError, match="only init_method='zeros' or 'adjoint'"):
        TorchSpectralNUFFTInterpolator(shape=(16,), init_method="interpolate")


def test_numpy_torch_parity_1d_smoke() -> None:
    torch, TorchSpectralNUFFTInterpolator = _torch_stack()
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 1.0, 48)).astype(np.float32)
    values = (np.sin(2 * np.pi * 3 * t) + 0.5j * np.cos(2 * np.pi * 5 * t)).astype(np.complex64)

    common = dict(
        shape=(32,),
        regularization_lambda=1e-2,
        n_iter=40,
        lr=1.5e-2,
        wavelet_level=2,
        init_method="adjoint",
        device="cpu",
        random_state=42,
    )
    np_interp = NumpySpectralNUFFTInterpolator(**common).fit(t, values)
    th_interp = TorchSpectralNUFFTInterpolator(**common).fit(torch.tensor(t), torch.tensor(values))

    assert np_interp.data_nrmse_ is not None and th_interp.data_nrmse_ is not None
    assert abs(np_interp.data_nrmse_ - th_interp.data_nrmse_) < 0.02

    grid_np = np_interp.reconstruct_grid()
    grid_th = th_interp.reconstruct_grid().detach().cpu().numpy()
    grid_delta = float(np.linalg.norm(grid_np - grid_th) / np.linalg.norm(grid_th))
    assert grid_delta < 0.05
