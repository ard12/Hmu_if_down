"""PCA feature extraction and Doppler Velocity Estimation for Wi-Fi CSI."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class CSIDynamicFeatures:
    node_id: int
    dominant_velocity_mps: float
    energy_surge_ratio: float
    moving_variance: float
    is_velocity_burst: bool
    pc1_signal: np.ndarray


class CSIPCAExtractor:
    """Extracts principal components and estimates Doppler velocity profiles."""

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        carrier_freq_hz: float = 2.437e9,  # Wi-Fi Channel 6 (2.437 GHz)
        velocity_threshold_mps: float = 1.8,
        energy_surge_threshold: float = 3.0,
    ):
        self.fs = sampling_rate_hz
        self.c = 3.0e8
        self.wavelength = self.c / carrier_freq_hz  # ~0.123 m
        self.vel_threshold = velocity_threshold_mps
        self.energy_threshold = energy_surge_threshold

    def compute_pca(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Perform PCA over subcarrier columns to extract dominant spatial motion vectors.
        
        Args:
            data: Matrix of shape (T, num_subcarriers)
        Returns:
            Matrix of shape (T, n_components)
        """
        if data.shape[0] < 2:
            return np.zeros((data.shape[0], n_components))

        # Sanitize NaN/Inf values from corrupt packets before SVD
        if np.any(~np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Mean-center each subcarrier
        centered = data - np.mean(data, axis=0, keepdims=True)

        # Singular Value Decomposition (SVD)
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            pcs = centered @ vt.T[:, :n_components]
            return pcs
        except np.linalg.LinAlgError:
            return np.zeros((data.shape[0], n_components))

    def estimate_velocity(self, pc_signal: np.ndarray) -> Tuple[float, float]:
        """Estimate Doppler frequency shift and human body velocity.
        
        Returns:
            (velocity_mps, energy_surge_ratio)
        """
        if len(pc_signal) < 32:
            return 0.0, 1.0

        # Compute power spectrum using Welch's method / FFT
        freqs, psd = signal.welch(
            pc_signal,
            fs=self.fs,
            nperseg=min(len(pc_signal), 64),
            scaling="spectrum",
        )

        # Ignore DC component (< 1 Hz)
        valid_mask = (freqs >= 1.0) & (freqs <= 35.0)
        if not np.any(valid_mask):
            return 0.0, 1.0

        active_freqs = freqs[valid_mask]
        active_psd = psd[valid_mask]

        # Dominant Doppler frequency is the weighted mean or peak frequency
        peak_freq = active_freqs[np.argmax(active_psd)]

        # Human velocity: v = (lambda * f_D) / 2
        velocity_mps = (self.wavelength * peak_freq) / 2.0

        # Energy Surge: High-frequency kinetic power vs quiescent background noise
        high_band_power = np.sum(active_psd)
        low_band_power = np.sum(psd[freqs < 1.0]) + 1e-6
        surge_ratio = high_band_power / low_band_power

        return float(velocity_mps), float(surge_ratio)

    def extract(self, node_id: int, filtered_data: np.ndarray) -> CSIDynamicFeatures:
        """Full feature extraction pipeline on temporal window."""
        if filtered_data.shape[0] < 16:
            return CSIDynamicFeatures(
                node_id=node_id,
                dominant_velocity_mps=0.0,
                energy_surge_ratio=1.0,
                moving_variance=0.0,
                is_velocity_burst=False,
                pc1_signal=np.array([]),
            )

        pcs = self.compute_pca(filtered_data, n_components=2)
        pc1 = pcs[:, 0] if pcs.shape[1] > 0 else np.zeros(filtered_data.shape[0])

        vel_mps, surge_ratio = self.estimate_velocity(pc1)
        variance = float(np.var(pc1[-30:])) if len(pc1) >= 30 else float(np.var(pc1))

        is_burst = (vel_mps >= self.vel_threshold) and (surge_ratio >= self.energy_threshold)

        return CSIDynamicFeatures(
            node_id=node_id,
            dominant_velocity_mps=vel_mps,
            energy_surge_ratio=surge_ratio,
            moving_variance=variance,
            is_velocity_burst=is_burst,
            pc1_signal=pc1,
        )
