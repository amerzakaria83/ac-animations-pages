"""Host-side signal processing helpers (heatmap generation)."""

from __future__ import annotations

import numpy as np


def range_azimuth_map(azimuth_static: np.ndarray, num_angle_bins: int = 64,
                      mirror: bool = False) -> np.ndarray:
    """Range-azimuth magnitude from the zero-Doppler virtual antenna samples.

    ``azimuth_static`` is ``[range_bin, virtual_antenna]`` complex data as sent
    in the AZIMUTH_STATIC_HEATMAP TLV.  An FFT across the antenna axis gives
    the angle spectrum.  Returns ``[range_bin, angle_bin]`` where angle bin
    ``k`` corresponds to ``sin(theta) = 2 * (k - N/2) / N``.

    ``mirror`` flips left/right, in case the antenna ordering of a firmware
    build differs from the convention used here.
    """
    if mirror:
        azimuth_static = np.conj(azimuth_static)
    spec = np.fft.fft(azimuth_static, n=num_angle_bins, axis=1)
    return np.abs(np.fft.fftshift(spec, axes=1))


def angle_axis_deg(num_angle_bins: int = 64) -> np.ndarray:
    k = np.arange(num_angle_bins) - num_angle_bins // 2
    return np.degrees(np.arcsin(np.clip(2.0 * k / num_angle_bins, -1, 1)))


class CartesianMapper:
    """Resamples a [range, angle] polar map onto an x/y grid (nearest bin)."""

    def __init__(self, num_range_bins: int, range_resolution: float,
                 num_angle_bins: int = 64, grid: int = 160,
                 max_range: float | None = None):
        self.max_range = max_range or num_range_bins * range_resolution
        self.grid = grid
        xs = np.linspace(-self.max_range, self.max_range, grid)
        ys = np.linspace(0, self.max_range, grid // 2)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        R = np.hypot(X, Y)
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_t = np.where(R > 0, X / R, 0.0)
        r_idx = np.round(R / range_resolution).astype(int)
        a_idx = np.round(sin_t * num_angle_bins / 2 + num_angle_bins // 2).astype(int)
        a_idx = np.clip(a_idx, 0, num_angle_bins - 1)
        self.valid = r_idx < num_range_bins
        self.r_idx = np.where(self.valid, r_idx, 0)
        self.a_idx = a_idx
        self.extent = (-self.max_range, 0.0, 2 * self.max_range, self.max_range)
        # A pixel can span several range bins: max-pool so narrow peaks survive.
        self.pool = max(1, int(np.ceil(2 * self.max_range / grid / range_resolution)))

    def __call__(self, polar: np.ndarray) -> np.ndarray:
        if self.pool > 1:
            pooled = polar.copy()
            for s in range(1, self.pool // 2 + 1):
                pooled[s:] = np.maximum(pooled[s:], polar[:-s])
                pooled[:-s] = np.maximum(pooled[:-s], polar[s:])
            polar = pooled
        img = polar[self.r_idx, self.a_idx]
        return np.where(self.valid, img, 0.0)
