"""
Gait Cadence Extractor from CSI micro-Doppler spectrogram.
Extracts step cadence, stride regularity, and velocity envelope.
"""

from collections import deque
import math
import time
from typing import Deque, Optional, Tuple
import numpy as np


class GaitCadenceAnalyzer:
    """
    Extracts step cadence, stride regularity, and velocity envelope
    from CSI micro-Doppler spectrogram using Short-Time Fourier Transform (STFT)
    and autocorrelation.

    Cadence bands:
      Normal walk:    1.5–2.3 Hz  (90–138 steps/min)
      Slow walk:      1.0–1.5 Hz  (60–90 steps/min)
      Shuffle gait:   0.3–1.0 Hz  (< 60 steps/min)
      Standing still: < 0.3 Hz
    """

    STFT_WINDOW_S = 3.0   # 3-second analysis window
    STFT_HOP_S = 0.5      # 500ms hop

    def __init__(self, fs: float = 100.0):
        self.fs = float(fs)
        self.window_samples = int(self.fs * self.STFT_WINDOW_S)
        self.hop_samples = int(self.fs * self.STFT_HOP_S)
        # Buffer stores (amplitude: float, timestamp: float)
        self._buffer: Deque[Tuple[float, float]] = deque(maxlen=int(self.fs * 15))

    def update(self, csi_amplitude: np.ndarray, timestamp: Optional[float] = None):
        """
        Ingest new CSI amplitude samples.
        Can accept a single float scalar, 1D array, or 2D array.
        """
        ts = timestamp if timestamp is not None else time.time()

        if isinstance(csi_amplitude, (int, float)):
            self._buffer.append((float(csi_amplitude), ts))
        else:
            arr = np.asarray(csi_amplitude, dtype=np.float64)
            if arr.ndim == 0:
                self._buffer.append((float(arr), ts))
            elif arr.ndim == 1:
                dt = 1.0 / self.fs
                start_ts = ts - (len(arr) - 1) * dt
                for i, val in enumerate(arr):
                    self._buffer.append((float(val), start_ts + i * dt))
            else:
                # Average across subcarriers for 2D array (subcarriers, time)
                mean_amps = np.mean(arr, axis=0) if arr.shape[0] < arr.shape[1] else np.mean(arr, axis=1)
                dt = 1.0 / self.fs
                start_ts = ts - (len(mean_amps) - 1) * dt
                for i, val in enumerate(mean_amps):
                    self._buffer.append((float(val), start_ts + i * dt))

    def compute_stft(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum for the Doppler signal in samples.
        Returns:
          freqs: 1D array of frequency bins in Hz (0 to fs/2)
          power_db: 1D array of power in dB
        """
        n = len(samples)
        if n < 4:
            return np.array([0.0]), np.array([-100.0])

        # Remove DC component
        detrended = samples - np.mean(samples)
        # Apply Hanning window
        windowed = detrended * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        power = np.abs(fft_vals) ** 2
        power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
        return freqs, power_db

    def analyze(self, current_time: Optional[float] = None) -> dict:
        """
        Analyze recent CSI amplitude window and classify gait.

        Returns:
          {
            "cadence_hz": float,
            "gait_class": "NORMAL" | "SLOW" | "SHUFFLE" | "STATIONARY",
            "stride_regularity": float,       # 0–1 autocorrelation peak
            "velocity_envelope_peak": float,  # max Doppler velocity (m/s proxy)
            "confidence": float,
          }
        """
        now = current_time if current_time is not None else (self._buffer[-1][1] if self._buffer else time.time())

        # Select samples within window [now - STFT_WINDOW_S, now]
        window_start = now - self.STFT_WINDOW_S
        window_items = [item for item in self._buffer if window_start <= item[1] <= now]

        n_samples = len(window_items)
        sample_ratio = n_samples / max(1, self.window_samples)
        confidence = min(1.0, max(0.0, sample_ratio))

        if n_samples < 8:
            return {
                "cadence_hz": 0.0,
                "gait_class": "STATIONARY",
                "stride_regularity": 0.0,
                "velocity_envelope_peak": 0.0,
                "confidence": round(confidence, 3),
            }

        samples = np.array([item[0] for item in window_items], dtype=np.float64)

        # Check for DC signal / negligible variance
        var = float(np.var(samples))
        if var < 1e-6:
            return {
                "cadence_hz": 0.0,
                "gait_class": "STATIONARY",
                "stride_regularity": 1.0,
                "velocity_envelope_peak": 0.0,
                "confidence": round(confidence, 3),
            }

        # 1. FFT spectrum
        freqs, power_db = self.compute_stft(samples)

        # Search for dominant frequency peak in gait band [0.2, 3.5] Hz
        gait_mask = (freqs >= 0.2) & (freqs <= 3.5)
        if not np.any(gait_mask):
            dominant_freq = 0.0
        else:
            gait_freqs = freqs[gait_mask]
            gait_power = power_db[gait_mask]
            peak_idx = int(np.argmax(gait_power))
            dominant_freq = float(gait_freqs[peak_idx])

        # 2. Autocorrelation for stride regularity (unbiased estimator)
        detrended = samples - np.mean(samples)
        autocorr = np.correlate(detrended, detrended, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        n_samples = len(detrended)
        lags = np.arange(len(autocorr))
        weights = np.maximum(n_samples - lags, 1)
        autocorr_unbiased = autocorr / weights
        r0 = autocorr_unbiased[0] if autocorr_unbiased[0] > 0 else 1e-12
        autocorr_norm = autocorr_unbiased / r0

        # Find first prominent peak after zero-crossing
        stride_regularity = 0.0
        if dominant_freq > 0.1:
            expected_lag = int(self.fs / dominant_freq)
            # Search around expected lag ± 30%
            lag_min = max(1, int(expected_lag * 0.7))
            lag_max = min(len(autocorr_norm) - 1, int(expected_lag * 1.3))
            if lag_min < lag_max:
                stride_regularity = float(np.max(autocorr_norm[lag_min:lag_max + 1]))
                stride_regularity = max(0.0, min(1.0, stride_regularity))
        else:
            # For stationary / DC, regularity is 0 unless perfectly flat
            stride_regularity = 0.0


        # 3. Classify gait class
        if dominant_freq >= 1.5:
            gait_class = "NORMAL"
        elif dominant_freq >= 1.0:
            gait_class = "SLOW"
        elif dominant_freq >= 0.3:
            gait_class = "SHUFFLE"
        else:
            gait_class = "STATIONARY"

        # 4. Velocity envelope peak (Doppler shift proxy: v = f_D * lambda / 2)
        # For 2.4 GHz Wi-Fi: lambda = 0.123m -> v = f_D * 0.0615
        velocity_envelope_peak = round(float(dominant_freq * 0.123 / 2.0 * 2.0), 3)

        return {
            "cadence_hz": round(dominant_freq, 2),
            "gait_class": gait_class,
            "stride_regularity": round(stride_regularity, 3),
            "velocity_envelope_peak": max(0.05, velocity_envelope_peak),
            "confidence": round(confidence, 3),
        }
