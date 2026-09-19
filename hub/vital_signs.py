"""Post-Fall Vital Signs & Respiration micro-Doppler Estimator (Milestone 10.1).

Extracts periodic chest wall displacement (0.1 - 0.5 Hz / 6 - 30 bpm) during
post-fall quiescence using mmWave radar phase variation and CSI subcarrier dynamics.
Differentiates living fallen subjects from inanimate dropped objects or apnea events.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import signal


@dataclass
class VitalSignsResult:
    """Estimated physiological vital signs during post-fall quiescence."""

    breathing_rate_bpm: float
    chest_displacement_mm: float
    vital_detected: bool
    status: str  # "NORMAL_BREATHING", "SHALLOW_BREATHING", "TACHYPNEA", "APNEA_OR_INANIMATE"
    confidence: float

    def to_dict(self) -> Dict[str, Union[float, bool, str]]:
        return {
            "breathing_rate_bpm": round(self.breathing_rate_bpm, 1),
            "chest_displacement_mm": round(self.chest_displacement_mm, 2),
            "vital_detected": self.vital_detected,
            "status": self.status,
            "confidence": round(self.confidence, 3),
        }


class VitalSignsEstimator:
    """Estimates respiration rate and chest wall displacement from micro-Doppler signals."""

    def __init__(
        self,
        min_freq_hz: float = 0.1,   # 6 breaths/min
        max_freq_hz: float = 0.5,   # 30 breaths/min
        min_displacement_mm: float = 0.2,
        radar_wavelength_mm: float = 5.0,  # 60 GHz -> ~5 mm
    ):
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_displacement_mm = min_displacement_mm
        self.radar_wavelength_mm = radar_wavelength_mm

    def process_radar_quiescence(
        self, phase_series: np.ndarray, sample_rate_hz: float = 20.0
    ) -> VitalSignsResult:
        """Process radar phase displacement series during quiescence.

        Args:
            phase_series: 1D array of unwrapped phase values (radians) from target range-bin.
            sample_rate_hz: Radar frame sampling frequency (Hz).

        Returns:
            VitalSignsResult with respiration rate and chest displacement.
        """
        if phase_series is None or len(phase_series) < int(sample_rate_hz * 2):
            return VitalSignsResult(
                breathing_rate_bpm=0.0,
                chest_displacement_mm=0.0,
                vital_detected=False,
                status="INSUFFICIENT_DATA",
                confidence=0.0,
            )

        # Convert phase to displacement in mm: d = (lambda / 4pi) * phi
        displacement_mm = (self.radar_wavelength_mm / (4.0 * np.pi)) * phase_series
        return self._analyze_displacement_series(displacement_mm, sample_rate_hz)

    def process_csi_quiescence(
        self, csi_amplitude_series: np.ndarray, sample_rate_hz: float = 100.0
    ) -> VitalSignsResult:
        """Process CSI amplitude series (e.g. PC1 or sensitive subcarrier) during quiescence.

        Args:
            csi_amplitude_series: 1D array of CSI amplitude values.
            sample_rate_hz: CSI packet sampling frequency (Hz).

        Returns:
            VitalSignsResult with respiration rate and estimated relative displacement.
        """
        if csi_amplitude_series is None or len(csi_amplitude_series) < int(sample_rate_hz * 2):
            return VitalSignsResult(
                breathing_rate_bpm=0.0,
                chest_displacement_mm=0.0,
                vital_detected=False,
                status="INSUFFICIENT_DATA",
                confidence=0.0,
            )

        # Detrend and normalize
        normalized = csi_amplitude_series - np.mean(csi_amplitude_series)
        # Scaled pseudo-displacement
        scale = np.std(normalized) * 2.0
        return self._analyze_displacement_series(normalized * scale, sample_rate_hz)

    def _analyze_displacement_series(
        self, displacement: np.ndarray, fs: float
    ) -> VitalSignsResult:
        """Internal spectral analysis of displacement time series."""
        n_samples = len(displacement)
        if n_samples < 4:
            return VitalSignsResult(0.0, 0.0, False, "INSUFFICIENT_DATA", 0.0)

        # Detrend
        d = signal.detrend(displacement)

        # Bandpass filter within respiratory band [0.1, 0.5] Hz
        # Nyquist limit check
        nyq = 0.5 * fs
        low = max(self.min_freq_hz / nyq, 0.001)
        high = min(self.max_freq_hz / nyq, 0.99)

        if low >= high:
            return VitalSignsResult(0.0, 0.0, False, "FILTER_ERROR", 0.0)

        try:
            b, a = signal.butter(2, [low, high], btype="bandpass")
            filtered = signal.filtfilt(b, a, d)
        except Exception:
            # Fallback if filtfilt encounters edge length issues
            filtered = d

        # Calculate peak-to-peak displacement
        p2p = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))

        # FFT Power Spectrum with zero-padding for 1 BPM (1/60 Hz) frequency resolution
        nfft = max(n_samples, int(fs * 60))
        fft_vals = np.abs(np.fft.rfft(filtered, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        # Mask respiratory band
        resp_mask = (freqs >= self.min_freq_hz) & (freqs <= self.max_freq_hz)
        if not np.any(resp_mask) or np.sum(fft_vals[resp_mask]) < 1e-6:
            return VitalSignsResult(
                breathing_rate_bpm=0.0,
                chest_displacement_mm=p2p,
                vital_detected=False,
                status="APNEA_OR_INANIMATE",
                confidence=0.0,
            )

        band_freqs = freqs[resp_mask]
        band_power = fft_vals[resp_mask]

        peak_idx = np.argmax(band_power)
        peak_freq = float(band_freqs[peak_idx])

        # Respiratory band energy ratio vs total spectrum (>0.05 Hz)
        resp_energy = float(np.sum(band_power**2))
        total_energy = float(np.sum(fft_vals[freqs > 0.05]**2)) + 1e-9
        confidence = float(np.clip(resp_energy / total_energy, 0.0, 1.0))

        breathing_bpm = peak_freq * 60.0

        # Classification criteria
        is_vital = (p2p >= self.min_displacement_mm) and (confidence >= 0.30)

        if not is_vital:
            status = "APNEA_OR_INANIMATE"
        elif breathing_bpm < 10.0:
            status = "SHALLOW_BREATHING"
        elif breathing_bpm > 24.0:
            status = "TACHYPNEA"
        else:
            status = "NORMAL_BREATHING"

        return VitalSignsResult(
            breathing_rate_bpm=breathing_bpm if is_vital else 0.0,
            chest_displacement_mm=p2p,
            vital_detected=is_vital,
            status=status,
            confidence=confidence if is_vital else 0.0,
        )

    @staticmethod
    def generate_synthetic_vital_signal(
        duration_sec: float = 10.0,
        sample_rate_hz: float = 20.0,
        breathing_rate_bpm: float = 16.0,
        displacement_mm: float = 1.2,
        noise_std: float = 0.05,
    ) -> np.ndarray:
        """Synthesize a calibrated human respiration micro-Doppler phase series."""
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate_hz), endpoint=False)
        freq_hz = breathing_rate_bpm / 60.0
        # Phase corresponding to displacement: phi = (4pi / lambda) * d
        lambda_mm = 5.0
        phi_amp = (4.0 * np.pi / lambda_mm) * displacement_mm
        clean_phase = phi_amp * np.sin(2.0 * np.pi * freq_hz * t)
        noise = np.random.normal(0, noise_std, size=len(t))
        return clean_phase + noise
