# Changelog

## 0.1.0 — 2026-07-09

- Initial package layout with two backends:
  - `nufftsi.numpy` — sigpy NUFFT + Wavelet, numpy Adam
  - `nufftsi.torch` — torchkbnufft + ptwt, torch Adam
- Public API: `SpectralNUFFTInterpolator`, `fit_spectral_nufft_interpolator`
- Extras: `[sigpy]`, `[torch]`, `[dev]`, `[all]`
