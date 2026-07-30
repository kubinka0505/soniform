"""
snd.py
------
Turn the shape's y(theta) waveform (one period, uniformly sampled) into a
looping audio tone at a given frequency, written as a 32-bit PCM WAV.
"""
from __future__ import annotations

import numpy as np

import wavemarks

from ..._global.render.snd import write

#-=-=-=-#

def render(
	obj,
	path_out,
	frequency: float = 220,
	sample_rate: int = 44100,
	amplitude: float = 1,
	oversample: int = 8,
) -> str:
	cycle = np.asarray(obj.y_norm, dtype = np.float64)
	n_cycle = len(cycle)

	# final output samples per period
	n_out = max(1, round(sample_rate / frequency))

	# oversampled internal waveform
	n_hi = n_out * oversample

	phase_hi = (
		np.arange(n_hi) *
		(n_cycle / n_hi)
	) % n_cycle

	hi = np.interp(
		phase_hi,
		np.arange(n_cycle),
		cycle,
		period = n_cycle,
	)

	# simple FFT low-pass
	spec = np.fft.rfft(hi)
	freqs = np.fft.rfftfreq(n_hi)

	# remove harmonics above output Nyquist
	spec[freqs > 0.5 / oversample] = 0

	hi = np.fft.irfft(spec, n_hi)

	# pick exactly one period, no delay/padding
	x_hi = np.linspace(
		0,
		n_hi,
		n_out,
		endpoint = False
	)

	signal = np.interp(
		x_hi,
		np.arange(n_hi),
		hi,
	)

	signal *= amplitude

	#-=-=-=-#
	# Write

	write(signal, path_out, sample_rate, normalize = True)

	# Add region
	mf = wavemarks.MarkerFile(path_out)

	cycle = wavemarks.Entry(
		0, len(signal),
		type = wavemarks.MarkerType.SINGLE_CYCLE
	)

	mf += cycle
	mf.save()

	return path_out