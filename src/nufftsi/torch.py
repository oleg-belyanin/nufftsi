from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

try:
    import ptwt
    import torch
    import torchkbnufft as tkbn

    _TORCH_NUFFT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled with a clear runtime error
    ptwt = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    tkbn = None  # type: ignore[assignment]
    _TORCH_NUFFT_AVAILABLE = False


InitMethod = Literal["zeros", "adjoint"]

__all__ = ["InitMethod", "SpectralNUFFTInterpolator", "fit_spectral_nufft_interpolator"]


def _as_shape(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(n) for n in shape)
    if len(out) not in (1, 2, 3):
        raise ValueError(f"Only 1D/2D/3D shapes are supported, got {shape}")
    if any(n <= 0 for n in out):
        raise ValueError(f"All shape dimensions must be positive, got {shape}")
    return out


def _normalize_points_tensor(points: Any, ndim: int) -> tuple[Any, tuple[int, ...]]:
    arr = points.to(dtype=torch.float32)
    if ndim == 1:
        if arr.ndim == 0:
            return arr.reshape(1, 1), ()
        if arr.ndim == 1:
            return arr.reshape(-1, 1), tuple(arr.shape)
        if arr.shape[-1] == 1:
            return arr.reshape(-1, 1), tuple(arr.shape[:-1])
        raise ValueError(f"For 1D, points must have shape (M,) or (..., 1), got {tuple(arr.shape)}")

    if arr.ndim == 0 or arr.shape[-1] != ndim:
        raise ValueError(f"For {ndim}D, points must have shape (..., {ndim}), got {tuple(arr.shape)}")
    return arr.reshape(-1, ndim), tuple(arr.shape[:-1])


def _check_dependencies() -> None:
    if not _TORCH_NUFFT_AVAILABLE:
        raise RuntimeError("Install torch, torchkbnufft, and ptwt for SpectralNUFFTInterpolator")


def _centered_spectrum_from_signal(signal: Any) -> Any:
    dims = tuple(range(signal.ndim))
    return torch.fft.fftshift(torch.fft.fftn(signal.to(torch.complex64), dim=dims), dim=dims) / signal.numel()


def _signal_from_centered_spectrum(coeff_centered: Any) -> Any:
    dims = tuple(range(coeff_centered.ndim))
    coeff_fft_order = torch.fft.ifftshift(coeff_centered, dim=dims)
    return torch.fft.ifftn(coeff_fft_order, dim=dims) * coeff_centered.numel()


def _make_ktrajectory(points: Any) -> Any:
    return (-2.0 * torch.pi * points.T).to(torch.float32)


def _recursive_abs_sum(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return torch.sum(torch.abs(obj))
    if isinstance(obj, dict):
        total = None
        for value in obj.values():
            part = _recursive_abs_sum(value)
            total = part if total is None else total + part
        if total is None:
            raise ValueError("Empty ptwt coefficient structure")
        return total
    if isinstance(obj, (tuple, list)):
        total = None
        for value in obj:
            part = _recursive_abs_sum(value)
            total = part if total is None else total + part
        if total is None:
            raise ValueError("Empty ptwt coefficient structure")
        return total
    raise TypeError(f"Unsupported ptwt coefficient type: {type(obj)!r}")


def _wavelet_l1_real_tensor(x: Any, ndim: int, wavelet: str, level: int | None) -> Any:
    if ndim == 1:
        coeffs = ptwt.wavedec(x, wavelet, mode="zero", level=level)
    elif ndim == 2:
        coeffs = ptwt.wavedec2(x, wavelet, mode="zero", level=level)
    elif ndim == 3:
        coeffs = ptwt.wavedec3(x, wavelet, mode="zero", level=level)
    else:  # pragma: no cover - guarded by shape validation
        raise ValueError(f"Only 1D/2D/3D are supported, got {ndim}")
    return _recursive_abs_sum(coeffs)


def _complex_wavelet_l1_spectrum(coeff_centered: Any, wavelet: str, level: int | None) -> Any:
    ndim = coeff_centered.ndim
    return _wavelet_l1_real_tensor(coeff_centered.real, ndim, wavelet, level) + _wavelet_l1_real_tensor(
        coeff_centered.imag, ndim, wavelet, level
    )


def _initial_grid_tensor(shape: tuple[int, ...], device: Any) -> Any:
    return torch.zeros(shape, dtype=torch.complex64, device=device)


@dataclass(slots=True)
class SpectralNUFFTInterpolator:
    shape: Sequence[int]
    regularization_lambda: float = 1e-2
    wavelet: str = "db4"
    wavelet_level: int | None = None
    n_iter: int = 1000
    lr: float = 1.5e-2
    init_method: InitMethod = "adjoint"
    device: str = "cpu"
    random_state: int | None = 42
    lambda_candidates: Sequence[float] | None = None
    validation_fraction: float = 0.2
    validation_iter: int | None = None

    shape_: tuple[int, ...] = field(init=False)
    selected_lambda_: float = field(default=0.0, init=False)
    spectrum_tensor_: Any | None = field(default=None, init=False, repr=False)
    grid_tensor_: Any | None = field(default=None, init=False, repr=False)
    data_nrmse_: float | None = field(default=None, init=False)
    _device_obj: Any = field(default=None, init=False, repr=False)
    _nufft: Any = field(default=None, init=False, repr=False)
    _adjoint: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape_ = _as_shape(self.shape)
        if self.regularization_lambda < 0:
            raise ValueError("regularization_lambda must be >= 0")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.init_method not in ("zeros", "adjoint"):
            raise ValueError("nufftsi.torch supports only init_method='zeros' or 'adjoint'")
        self.selected_lambda_ = float(self.regularization_lambda)

    @property
    def ndim(self) -> int:
        return len(self.shape_)

    def _ensure_modules(self) -> tuple[Any, Any, Any]:
        _check_dependencies()
        if self._nufft is None or self._adjoint is None or self._device_obj is None:
            self._device_obj = torch.device(self.device)
            self._nufft = tkbn.KbNufft(im_size=self.shape_).to(self._device_obj)
            self._adjoint = tkbn.KbNufftAdjoint(im_size=self.shape_).to(self._device_obj)
        return self._device_obj, self._nufft, self._adjoint

    def _forward(self, coeff_centered: Any, ktraj: Any) -> Any:
        _, nufft, _ = self._ensure_modules()
        return nufft(coeff_centered.reshape((1, 1) + tuple(coeff_centered.shape)), ktraj, norm=None).reshape(-1)

    def _adjoint_initial_grid(self, points_t: Any, values_t: Any) -> Any:
        device, _, adjoint = self._ensure_modules()
        points_t = points_t.to(device)
        values_t = values_t.to(device=device, dtype=torch.complex64).reshape(-1)
        ktraj = _make_ktrajectory(points_t)
        coeff = adjoint(values_t.reshape(1, 1, -1), ktraj, norm=None).reshape(self.shape_)
        pred = self._forward(coeff, ktraj)
        denom = torch.clamp(torch.sum(torch.abs(pred) ** 2), min=1e-12)
        scale = torch.vdot(pred, values_t) / denom
        coeff = coeff * scale
        return _signal_from_centered_spectrum(coeff).detach()

    def _fit_with_lambda(
        self,
        points_t: Any,
        values_t: Any,
        init_grid_t: Any,
        lamda: float,
        n_iter: int,
    ) -> tuple[Any, Any, float]:
        device, _, _ = self._ensure_modules()
        ktraj = _make_ktrajectory(points_t.to(device))
        values_t = values_t.to(device=device, dtype=torch.complex64)

        real = torch.nn.Parameter(init_grid_t.real.detach().clone().to(device=device, dtype=torch.float32))
        imag = torch.nn.Parameter(init_grid_t.imag.detach().clone().to(device=device, dtype=torch.float32))
        optimizer = torch.optim.Adam([real, imag], lr=self.lr)

        for _ in range(n_iter):
            optimizer.zero_grad(set_to_none=True)
            signal = torch.complex(real, imag)
            coeff = _centered_spectrum_from_signal(signal)
            residual = self._forward(coeff, ktraj) - values_t
            data_loss = torch.mean(torch.abs(residual) ** 2)
            reg_loss = _complex_wavelet_l1_spectrum(coeff, self.wavelet, self.wavelet_level) / coeff.numel()
            loss = data_loss + float(lamda) * reg_loss
            loss.backward()
            optimizer.step()

        signal = torch.complex(real.detach(), imag.detach())
        coeff = _centered_spectrum_from_signal(signal).detach()
        pred = self._forward(coeff, ktraj)
        data_nrmse = float(torch.linalg.norm(pred - values_t) / torch.linalg.norm(values_t))
        return signal.detach(), coeff, data_nrmse

    def _select_lambda(self, points_t: Any, values_t: Any, init_grid_t: Any) -> float:
        if self.lambda_candidates is None:
            return float(self.regularization_lambda)

        candidates = [float(v) for v in self.lambda_candidates]
        if not candidates:
            raise ValueError("lambda_candidates must not be empty")

        device, _, _ = self._ensure_modules()
        n = points_t.shape[0]
        n_val = max(1, int(round(n * self.validation_fraction)))
        generator = torch.Generator(device="cpu")
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))
        perm = torch.randperm(n, generator=generator)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if train_idx.numel() == 0:
            raise ValueError("Too few samples for validation split")

        points_train = points_t[train_idx].to(device)
        values_train = values_t[train_idx].to(device=device, dtype=torch.complex64)
        points_val = points_t[val_idx].to(device)
        values_val = values_t[val_idx].to(device=device, dtype=torch.complex64)
        ktraj_val = _make_ktrajectory(points_val)
        n_iter = self.validation_iter if self.validation_iter is not None else max(100, self.n_iter // 3)

        best_lambda = candidates[0]
        best_score = float("inf")
        for lamda in candidates:
            _, coeff, _ = self._fit_with_lambda(points_train, values_train, init_grid_t, lamda, n_iter=n_iter)
            pred = self._forward(coeff.to(device), ktraj_val)
            score = float(torch.linalg.norm(pred - values_val) / torch.linalg.norm(values_val))
            if score < best_score:
                best_score = score
                best_lambda = lamda
        return best_lambda

    def fit(self, points: Any, values: Any) -> "SpectralNUFFTInterpolator":
        _check_dependencies()
        if not torch.is_tensor(points) or not torch.is_tensor(values):
            raise TypeError(
                "nufftsi.torch accepts torch.Tensor only. "
                "For numpy inputs use nufftsi.numpy."
            )

        device, _, _ = self._ensure_modules()
        points_t, _ = _normalize_points_tensor(points.to(device=device), self.ndim)
        values_t = values.to(device=device, dtype=torch.complex64).reshape(-1)
        if points_t.shape[0] != values_t.shape[0]:
            raise ValueError(f"points and values have different lengths: {points_t.shape[0]} != {values_t.shape[0]}")

        if self.init_method == "adjoint":
            init_grid_t = self._adjoint_initial_grid(points_t, values_t)
        else:
            init_grid_t = _initial_grid_tensor(self.shape_, device=device)

        lamda = self._select_lambda(points_t, values_t, init_grid_t)
        signal, coeff, data_nrmse = self._fit_with_lambda(points_t, values_t, init_grid_t, lamda, n_iter=self.n_iter)

        self.selected_lambda_ = lamda
        self.grid_tensor_ = signal.detach()
        self.spectrum_tensor_ = coeff.detach()
        self.data_nrmse_ = data_nrmse
        return self

    def reconstruct_grid(self) -> Any:
        if self.grid_tensor_ is None:
            raise RuntimeError("Call fit(...) first")
        return self.grid_tensor_.clone()

    def predict(self, points: Any) -> Any:
        _check_dependencies()
        if not torch.is_tensor(points):
            raise TypeError(
                "nufftsi.torch accepts torch.Tensor only. "
                "For numpy points use nufftsi.numpy."
            )
        if self.spectrum_tensor_ is None:
            raise RuntimeError("Call fit(...) first")

        device, _, _ = self._ensure_modules()
        pts, output_shape = _normalize_points_tensor(points.to(device=device), self.ndim)
        pred = self._forward(self.spectrum_tensor_.to(device), _make_ktrajectory(pts))
        return pred.reshape(output_shape)

    def __call__(self, points: Any) -> Any:
        return self.predict(points)


def fit_spectral_nufft_interpolator(
    points: Any,
    values: Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> SpectralNUFFTInterpolator:
    return SpectralNUFFTInterpolator(shape=shape, **kwargs).fit(points, values)
