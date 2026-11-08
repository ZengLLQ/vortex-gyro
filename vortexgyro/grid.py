"""2D pseudo-spectral Fourier grid with 2/3 Orszag de-aliasing."""

from __future__ import annotations
import numpy as np


class SpectralGrid2D:
    """Double-periodic 2D pseudo-spectral grid for perpendicular plane (x, y)."""

    def __init__(self, nx: int = 64, ny: int = 64, lx: float = 64.0, ly: float = 64.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny

        # Real space coordinate meshes
        self.x = np.linspace(0.0, lx, nx, endpoint=False)
        self.y = np.linspace(0.0, ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Spectral wavenumbers using 2D FFT
        self.kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")

        # Perpendicular wavenumber magnitude squared: k_perp^2 = k_x^2 + k_y^2
        self.kperp2 = self.KX**2 + self.KY**2

        # 2/3 Orszag de-aliasing filter mask
        kx_max = np.max(np.abs(self.kx))
        ky_max = np.max(np.abs(self.ky))
        self.dealias_mask = (np.abs(self.KX) < (2.0 / 3.0) * kx_max) & (
            np.abs(self.KY) < (2.0 / 3.0) * ky_max
        )

    def fft2(self, field_real: np.ndarray) -> np.ndarray:
        """Forward 2D Fourier transform with de-aliasing filter applied."""
        f_hat = np.fft.fft2(field_real)
        return f_hat * self.dealias_mask

    def ifft2(self, field_hat: np.ndarray) -> np.ndarray:
        """Inverse 2D Fourier transform returning real field."""
        return np.real(np.fft.ifft2(field_hat * self.dealias_mask))

    def diff_x(self, field_hat: np.ndarray) -> np.ndarray:
        """Computes partial derivative d/dx in Fourier space."""
        return 1j * self.KX * field_hat

    def diff_y(self, field_hat: np.ndarray) -> np.ndarray:
        """Computes partial derivative d/dy in Fourier space."""
        return 1j * self.KY * field_hat

    def laplacian(self, field_hat: np.ndarray) -> np.ndarray:
        """Computes Laplacian nabla_perp^2 = -k_perp^2 in Fourier space."""
        return -self.kperp2 * field_hat
