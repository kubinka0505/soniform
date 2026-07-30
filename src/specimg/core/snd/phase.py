import os
import random
import numpy as np
from scipy.signal import stft as _stft, istft as _istft

from ...config import Settings

#-=-=-=-#

def _do_istft(complex_spec: np.ndarray, cfg: Settings) -> np.ndarray:
	_, audio = _istft(
		complex_spec,
		fs = cfg.sample_rate,
		window = cfg.window,
		nperseg = cfg.n_fft,
		noverlap = cfg.n_fft - cfg.hop_length
	)

	return audio

def _do_stft(
	audio: np.ndarray,
	cfg: Settings
) -> np.ndarray:
	_, _, complex_spec = _stft(
		audio,
		fs = cfg.sample_rate,
		window = cfg.window,
		nperseg = cfg.n_fft,
		noverlap = cfg.n_fft - cfg.hop_length,
	)

	return complex_spec

def magnitude_to_audio(
	magnitude: np.ndarray,
	cfg: Settings,
	seed: int | None = random.randint(0, os.sys.maxsize)
) -> np.ndarray:
	"""
	Reconstruct a waveform from a magnitude spectrogram.

	A phase estimate is generated according to `cfg.phase_mode`,
	followed by inverse STFT synthesis.

	Optional Griffin-Lim refinement iterations can
	be applied to improve phase consistency.

	Parameters
	----------
	magnitude (np.ndarray):
		Linear magnitude spectrogram.

	seed (int | None):
		Seed used when generating random phase.

	Returns
	-------
	np.ndarray
		Mono audio signal as floating-point samples.
	"""
	rng = np.random.default_rng(seed)

	if cfg.phase_mode == "zero":
		phase = np.zeros_like(magnitude)
	else:
		phase = rng.uniform(-np.pi, np.pi, size = magnitude.shape)

	spec = magnitude * np.exp(1j * phase)
	audio = _do_istft(spec, cfg)

	for _ in range(cfg.griffin_lim_iters):
		rebuilt = _do_stft(audio, cfg)

		n = min(rebuilt.shape[1], magnitude.shape[1])
		new_phase = np.angle(rebuilt[:, :n])

		spec = magnitude[:, :n] * np.exp(1j * new_phase)
		audio = _do_istft(spec, cfg)

	return audio