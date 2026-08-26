"""Spectral NUFFT interpolator with numpy/sigpy and torch backends."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nufftsi")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

# Default public surface: numpy/sigpy backend (no torch required).
from nufftsi.numpy import SpectralNUFFTInterpolator, fit_spectral_nufft_interpolator

__all__ = [
    "SpectralNUFFTInterpolator",
    "fit_spectral_nufft_interpolator",
    "__version__",
]
