import os
import random
import numpy as np

from ...config import Settings

from .phase import magnitude_to_audio

#-=-=-=-#

def intensity_to_magnitude(
	intensity: np.ndarray,
	cfg: Settings
) -> np.ndarray:
	"""
	Convert normalized image intensity into linear spectrogram magnitude.

	Parameters
	----------
	intensity (np.ndarray):
		Array containing values in the range [0, 1].

	Returns
	-------
	np.ndarray:
		Linear magnitude values suitable for STFT synthesis.

	Notes
	-----
	If `cfg.min_level_db` is None:
		intensity = 0 -> magnitude = 0
		intensity = 1 -> magnitude = 1

	Otherwise intensity is mapped onto the dB interval:
		[cfg.min_level_db, 0 dB]

	and converted back to linear amplitude.
	"""
	if cfg.min_level_db is None:
		return np.clip(intensity, 0, 1)

	db = cfg.min_level_db + np.clip(intensity, 0, 1) * (-cfg.min_level_db)
	return 10 ** (db / 20)

def _row_frequencies(
	lo: float,
	hi: float,
	n_rows: int,
	mode: str
) -> np.ndarray:
	"""
	Map n_rows source-image rows onto the [lo, hi] Hz range.

	Parameters
	----------
	mode (str):
		"log"    -> more rows devoted to the low end (matches how
		            spectrogram viewers usually draw a log-frequency axis).
		"exp"    -> the mirror of log -- more rows devoted to the high end.

		Otherwise linear.
	"""
	t = np.linspace(0, 1, n_rows)

	if mode == "log":
		lo_safe = max(lo, 1)
		hi_safe = max(hi, lo_safe * os.sys.float_info.epsilon)

		return lo_safe * (hi_safe / lo_safe) ** t

	if mode == "exp":
		lo_safe = max(lo, 1)
		hi_safe = max(hi, lo_safe * os.sys.float_info.epsilon)
		mirrored = lo_safe * (hi_safe / lo_safe) ** (1 - t)

		return lo_safe + hi_safe - mirrored

	return lo + t * (hi - lo)

def build_magnitude_spectrogram(
	intensity: np.ndarray,
	cfg: Settings,
	n_time_frames: int
) -> np.ndarray:
	"""
	Construct a full-resolution magnitude spectrogram from an image.

	The input intensity image is mapped into the frequency range
	[`frequency_min`, `frequency_max`] and inserted into a spectrogram
	containing `cfg.n_freq_bins` frequency bins.

	Frequencies outside the selected band are filled with either:

	- true silence, or
	- the configured dB floor (`min_level_db`).

	Returns
	-------
	np.ndarray
		Magnitude spectrogram with shape:
			(cfg.n_freq_bins, n_time_frames)

		suitable for phase reconstruction and ISTFT synthesis.

	Notes
	-----
	`cfg.frequency_mode` controls how image rows are distributed across
	the frequency band:

	- "linear" -> rows mapped evenly (default).
	- "log"    -> rows distributed geometrically, more resolution at low end.
	- "exp"    -> mirror of log, more resolution at the high end.
	"""
	n_bins = cfg.n_freq_bins
	freqs = np.fft.rfftfreq(cfg.n_fft, d = 1 / cfg.sample_rate)

	silence = intensity_to_magnitude(np.zeros((1, n_time_frames)), cfg)
	magnitude = np.repeat(silence, n_bins, axis = 0)

	freq_lo = cfg.resolved_freq_min
	freq_hi = cfg.resolved_freq_max

	if cfg.frequency_mode in ("log", "exp"):
		src_rows = intensity.shape[0]
		row_freqs = _row_frequencies(freq_lo, freq_hi, src_rows, cfg.frequency_mode)

		for bin_idx, f in enumerate(freqs):
			if f < freq_lo or f > freq_hi:
				continue

			row = int(np.argmin(np.abs(row_freqs - f)))
			magnitude[bin_idx, :] = intensity_to_magnitude(
				intensity[row:row + 1, :],
				cfg
			)[0]
	else:
		band_mask = (freqs >= freq_lo) & (freqs <= freq_hi)
		band_bins = np.where(band_mask)[0]

		if len(band_bins) == 0:
			raise SystemExit("Frequencies produced an empty band for the given sample_rate / n_fft.")

		src_rows = intensity.shape[0]
		row_idx = np.round(np.linspace(0, src_rows - 1, num = len(band_bins))).astype(int)

		resampled = intensity[row_idx, :]

		magnitude[band_bins, :] = intensity_to_magnitude(resampled, cfg)

	return magnitude

def render(
	magnitude: np.ndarray,
	cfg: Settings
) -> np.ndarray:
	"""
	Generate a stereo waveform from a magnitude spectrogram.

	The left and right channels are reconstructed independently
	using different random phase seeds. This produces a wider
	stereo image than duplicating a single mono reconstruction.

	Returns
	-------
	np.ndarray
		Stereo audio array with shape:
			(samples, 2)
	"""
	left = magnitude_to_audio(magnitude, cfg, random.randint(0, os.sys.maxsize))
	right = magnitude_to_audio(magnitude, cfg, random.randint(0, os.sys.maxsize))

	n = min(
		len(left),
		len(right)
	)

	return np.stack([left[:n], right[:n]], axis = -1)