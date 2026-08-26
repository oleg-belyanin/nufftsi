from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

try:
    import sigpy.linop as lop

    _SIGPY_AVAILABLE = True
except ImportError:  # pragma: no cover - handled with a clear runtime error
    lop = None  # type: ignore[assignment]
    _SIGPY_AVAILABLE = False

try:
    from scipy.interpolate import PchipInterpolator, griddata

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PchipInterpolator = None  # type: ignore[assignment]
    griddata = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


InitMethod = Literal["zeros", "interpolate", "adjoint"]

__all__ = ["InitMethod", "SpectralNUFFTInterpolator", "fit_spectral_nufft_interpolator"]


def _as_shape(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(n) for n in shape)
    if len(out) not in (1, 2, 3):
        raise ValueError(f"Only 1D/2D/3D shapes are supported, got {shape}")
    if any(n <= 0 for n in out):
        raise ValueError(f"All shape dimensions must be positive, got {shape}")
    return out


def _normalize_points(points: np.ndarray, ndim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(points, dtype=np.float64)
    if ndim == 1:
        if arr.ndim == 0:
            return arr.reshape(1, 1), ()
        if arr.ndim == 1:
            return arr.reshape(-1, 1), arr.shape
        if arr.shape[-1] == 1:
            return arr.reshape(-1, 1), arr.shape[:-1]
        raise ValueError(f"For 1D, points must have shape (M,) or (..., 1), got {arr.shape}")

    if arr.ndim == 0 or arr.shape[-1] != ndim:
        raise ValueError(f"For {ndim}D, points must have shape (..., {ndim}), got {arr.shape}")
    return arr.reshape(-1, ndim), arr.shape[:-1]


def _is_torch_tensor(obj: Any) -> bool:
    cls = type(obj)
    return cls.__module__.startswith("torch") and cls.__name__ == "Tensor"


def _check_dependencies() -> None:
    if not _SIGPY_AVAILABLE:
        raise RuntimeError(
            "Install sigpy for SpectralNUFFTInterpolator (numpy backend). "
            "PyWavelets is pulled in transitively by sigpy."
        )


def _centered_spectrum_from_signal(signal: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(signal.astype(np.complex128, copy=False))) / signal.size


def _signal_from_centered_spectrum(coeff_centered: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(np.fft.ifftshift(coeff_centered)) * coeff_centered.size


def _points_to_sigpy_coord(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert unit-box points in [0, 1) to sigpy NUFFT coords in ~[-N/2, N/2].

    Empirically, coord = -points * shape matches torchkbnufft's
    ktraj = -2π * points (with scale sqrt(N) on the NUFFT output).
    """
    scale = np.asarray(shape, dtype=np.float64)
    return (-points.astype(np.float64, copy=False)) * scale


def _nufft_output_scale(shape: tuple[int, ...]) -> float:
    return float(np.sqrt(np.prod(shape)))


def _make_nufft(points: np.ndarray, shape: tuple[int, ...]) -> Any:
    _check_dependencies()
    assert lop is not None
    coord = _points_to_sigpy_coord(points, shape)
    return lop.NUFFT(ishape=shape, coord=coord)


def _forward_spectrum(coeff_centered: np.ndarray, nufft: Any, shape: tuple[int, ...]) -> np.ndarray:
    return (nufft * coeff_centered.astype(np.complex128, copy=False)) * _nufft_output_scale(shape)


def _adjoint_to_spectrum(residual: np.ndarray, nufft: Any, shape: tuple[int, ...]) -> np.ndarray:
    return (nufft.H * residual.astype(np.complex128, copy=False)) * _nufft_output_scale(shape)


def _make_wavelet(shape: tuple[int, ...], wavelet: str, level: int | None) -> Any:
    _check_dependencies()
    assert lop is not None
    axes = tuple(range(len(shape)))
    return lop.Wavelet(ishape=shape, axes=axes, wave_name=wavelet, level=level)


def _wavelet_l1_real(x: np.ndarray, wavelet_op: Any) -> float:
    coeffs = wavelet_op * np.asarray(x, dtype=np.float64)
    return float(np.sum(np.abs(coeffs)))


def _complex_wavelet_l1_spectrum(coeff_centered: np.ndarray, wavelet_op: Any) -> float:
    return _wavelet_l1_real(coeff_centered.real, wavelet_op) + _wavelet_l1_real(coeff_centered.imag, wavelet_op)


def _wavelet_l1_subgrad_real(x: np.ndarray, wavelet_op: Any) -> np.ndarray:
    coeffs = wavelet_op * np.asarray(x, dtype=np.float64)
    return np.asarray(wavelet_op.H * np.sign(coeffs), dtype=np.float64)


def _complex_griddata(points: np.ndarray, values: np.ndarray, grid_points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if not _SCIPY_AVAILABLE or griddata is None:
        raise RuntimeError("scipy.interpolate.griddata is unavailable; install scipy or use init_method='zeros'")

    real_linear = griddata(points, values.real, grid_points, method="linear")
    imag_linear = griddata(points, values.imag, grid_points, method="linear")
    real_nearest = griddata(points, values.real, grid_points, method="nearest")
    imag_nearest = griddata(points, values.imag, grid_points, method="nearest")

    real = np.where(np.isnan(real_linear), real_nearest, real_linear)
    imag = np.where(np.isnan(imag_linear), imag_nearest, imag_linear)
    return (real + 1j * imag).reshape(shape)


def _regular_grid_points(shape: tuple[int, ...]) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=-1)


def _initial_grid(points: np.ndarray, values: np.ndarray, shape: tuple[int, ...], method: InitMethod) -> np.ndarray:
    if method == "zeros":
        return np.zeros(shape, dtype=np.complex128)
    if method == "adjoint":
        raise ValueError("For adjoint initialization use the internal sigpy operator, not numpy-grid init")
    if method != "interpolate":
        raise ValueError(f"Unknown init_method={method!r}")

    if len(shape) == 1:
        if not _SCIPY_AVAILABLE or PchipInterpolator is None:
            raise RuntimeError("PchipInterpolator is unavailable; install scipy or use init_method='zeros'")
        order = np.argsort(points[:, 0])
        x = points[order, 0]
        y = values[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        grid = np.linspace(0.0, 1.0, shape[0], endpoint=False, dtype=np.float64)
        real = PchipInterpolator(unique_x, unique_y.real, extrapolate=True)(grid)
        imag = PchipInterpolator(unique_x, unique_y.imag, extrapolate=True)(grid)
        return (real + 1j * imag).astype(np.complex128)

    return _complex_griddata(points, values, _regular_grid_points(shape), shape).astype(np.complex128)


class _AdamND:
    """Minimal Adam matching torch.optim.Adam defaults for real parameter arrays."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.betas = betas
        self.eps = float(eps)
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.betas
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g64 = np.asarray(g, dtype=np.float64)
            self.m[i] *= b1
            self.m[i] += (1.0 - b1) * g64
            self.v[i] *= b2
            self.v[i] += (1.0 - b2) * (g64 * g64)
            mhat = self.m[i] / (1.0 - b1**self.t)
            vhat = self.v[i] / (1.0 - b2**self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


@dataclass(slots=True)
class SpectralNUFFTInterpolator:
    shape: Sequence[int]
    regularization_lambda: float = 1e-2
    wavelet: str = "db4"
    wavelet_level: int | None = None
    n_iter: int = 1000
    lr: float = 1.5e-2
    init_method: InitMethod = "interpolate"
    device: str = "cpu"
    random_state: int | None = 42
    lambda_candidates: Sequence[float] | None = None
    validation_fraction: float = 0.2
    validation_iter: int | None = None

    shape_: tuple[int, ...] = field(init=False)
    selected_lambda_: float = field(default=0.0, init=False)
    spectrum_: np.ndarray | None = field(default=None, init=False, repr=False)
    grid_: np.ndarray | None = field(default=None, init=False, repr=False)
    data_nrmse_: float | None = field(default=None, init=False)
    _wavelet_op: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape_ = _as_shape(self.shape)
        if self.regularization_lambda < 0:
            raise ValueError("regularization_lambda must be >= 0")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.device not in ("cpu", "cpu:0"):
            raise ValueError(
                "Numpy/sigpy backend currently supports only device='cpu'. "
                "For CUDA use nufftsi.torch."
            )
        self.selected_lambda_ = float(self.regularization_lambda)

    @property
    def ndim(self) -> int:
        return len(self.shape_)

    def _ensure_wavelet(self) -> Any:
        if self._wavelet_op is None:
            self._wavelet_op = _make_wavelet(self.shape_, self.wavelet, self.wavelet_level)
        return self._wavelet_op

    def _adjoint_initial_grid(self, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        nufft = _make_nufft(points, self.shape_)
        values = values.astype(np.complex128, copy=False).reshape(-1)
        coeff = _adjoint_to_spectrum(values, nufft, self.shape_)
        pred = _forward_spectrum(coeff, nufft, self.shape_)
        denom = max(float(np.sum(np.abs(pred) ** 2)), 1e-12)
        scale = np.vdot(pred, values) / denom
        coeff = coeff * scale
        return _signal_from_centered_spectrum(coeff).astype(np.complex128)

    def _loss_grad(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        values: np.ndarray,
        nufft: Any,
        lamda: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        signal = real + 1j * imag
        coeff = _centered_spectrum_from_signal(signal)
        pred = _forward_spectrum(coeff, nufft, self.shape_)
        residual = pred - values
        m = float(values.size)
        data_loss = float(np.mean(np.abs(residual) ** 2))

        wavelet_op = self._ensure_wavelet()
        reg_loss = _complex_wavelet_l1_spectrum(coeff, wavelet_op) / float(coeff.size)
        loss = data_loss + float(lamda) * reg_loss

        g_c = (2.0 / m) * _adjoint_to_spectrum(residual, nufft, self.shape_)
        if lamda != 0.0:
            g_c_real = _wavelet_l1_subgrad_real(coeff.real, wavelet_op) / float(coeff.size)
            g_c_imag = _wavelet_l1_subgrad_real(coeff.imag, wavelet_op) / float(coeff.size)
            g_c = g_c + float(lamda) * (g_c_real + 1j * g_c_imag)

        # Adjoint of centered_spectrum: g_x = ifftn(ifftshift(g_c))
        g_x = np.fft.ifftn(np.fft.ifftshift(g_c))
        return loss, data_loss, np.asarray(g_x.real, dtype=np.float64), np.asarray(g_x.imag, dtype=np.float64), coeff

    def _fit_with_lambda(
        self,
        points: np.ndarray,
        values: np.ndarray,
        init_grid: np.ndarray,
        lamda: float,
        n_iter: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        nufft = _make_nufft(points, self.shape_)
        values = values.astype(np.complex128, copy=False).reshape(-1)

        real = np.array(init_grid.real, dtype=np.float64, copy=True)
        imag = np.array(init_grid.imag, dtype=np.float64, copy=True)
        optimizer = _AdamND([real, imag], lr=self.lr)

        coeff = _centered_spectrum_from_signal(real + 1j * imag)
        for _ in range(n_iter):
            _, _, g_real, g_imag, coeff = self._loss_grad(real, imag, values, nufft, lamda)
            optimizer.step([g_real, g_imag])

        signal = (real + 1j * imag).astype(np.complex128)
        coeff = _centered_spectrum_from_signal(signal)
        pred = _forward_spectrum(coeff, nufft, self.shape_)
        data_nrmse = float(np.linalg.norm(pred - values) / np.linalg.norm(values))
        return signal, coeff, data_nrmse

    def _select_lambda(self, points: np.ndarray, values: np.ndarray, init_grid: np.ndarray) -> float:
        if self.lambda_candidates is None:
            return float(self.regularization_lambda)

        candidates = [float(v) for v in self.lambda_candidates]
        if not candidates:
            raise ValueError("lambda_candidates must not be empty")

        n = points.shape[0]
        n_val = max(1, int(round(n * self.validation_fraction)))
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if train_idx.size == 0:
            raise ValueError("Too few samples for validation split")

        points_train = points[train_idx]
        values_train = values[train_idx]
        points_val = points[val_idx]
        values_val = values[val_idx]
        nufft_val = _make_nufft(points_val, self.shape_)
        n_iter = self.validation_iter if self.validation_iter is not None else max(100, self.n_iter // 3)

        best_lambda = candidates[0]
        best_score = float("inf")
        for lamda in candidates:
            _, coeff, _ = self._fit_with_lambda(points_train, values_train, init_grid, lamda, n_iter=n_iter)
            pred = _forward_spectrum(coeff, nufft_val, self.shape_)
            score = float(np.linalg.norm(pred - values_val) / np.linalg.norm(values_val))
            if score < best_score:
                best_score = score
                best_lambda = lamda
        return best_lambda

    def fit(self, points: Any, values: Any) -> "SpectralNUFFTInterpolator":
        if _is_torch_tensor(points) or _is_torch_tensor(values):
            raise TypeError(
                "nufftsi.numpy accepts numpy inputs only. "
                "For torch.Tensor use nufftsi.torch."
            )

        _check_dependencies()
        pts, _ = _normalize_points(points, self.ndim)
        vals = np.asarray(values, dtype=np.complex128).reshape(-1)
        if pts.shape[0] != vals.shape[0]:
            raise ValueError(f"points and values have different lengths: {pts.shape[0]} != {vals.shape[0]}")

        if self.init_method == "adjoint":
            init_grid = self._adjoint_initial_grid(pts, vals)
        else:
            init_grid = _initial_grid(pts, vals, self.shape_, self.init_method)

        lamda = self._select_lambda(pts, vals, init_grid)
        signal, coeff, data_nrmse = self._fit_with_lambda(pts, vals, init_grid, lamda, n_iter=self.n_iter)

        self.selected_lambda_ = lamda
        self.grid_ = signal.astype(np.complex64)
        self.spectrum_ = coeff.astype(np.complex64)
        self.data_nrmse_ = data_nrmse
        return self

    def reconstruct_grid(self) -> np.ndarray:
        if self.grid_ is None:
            raise RuntimeError("Call fit(...) first")
        return self.grid_.copy()

    def predict(self, points: Any) -> np.ndarray:
        if _is_torch_tensor(points):
            raise TypeError(
                "nufftsi.numpy accepts numpy points only. "
                "For torch.Tensor use nufftsi.torch."
            )
        if self.spectrum_ is None:
            raise RuntimeError("Call fit(...) first")

        pts_np, output_shape = _normalize_points(points, self.ndim)
        nufft = _make_nufft(pts_np, self.shape_)
        pred = _forward_spectrum(self.spectrum_.astype(np.complex128), nufft, self.shape_)
        return np.asarray(pred.reshape(output_shape), dtype=np.complex64)

    def __call__(self, points: Any) -> np.ndarray:
        return self.predict(points)


def fit_spectral_nufft_interpolator(
    points: np.ndarray,
    values: np.ndarray,
    shape: Sequence[int],
    **kwargs: Any,
) -> SpectralNUFFTInterpolator:
    return SpectralNUFFTInterpolator(shape=shape, **kwargs).fit(points, values)
