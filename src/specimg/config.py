import os
from dataclasses import dataclass

from .._global.setup import Logger

#-=-=-=-#

@dataclass
class Settings:
	# Audio geometry
	sample_rate: int   = 44100
	duration:    float = 6
	n_fft:       int   = 2048
	hop_ratio:   float = 0              # hop_length = n_fft * hop_ratio (overlap)
	window:      str  = "hann"          # any scipy.signal window name

	# Frequency placement
	frequency_min:  float = 0
	frequency_max:  float = -1
	frequency_mode: str = "linear"

	# Image handling
	invert:              bool  = False
	flip_vertical:       bool  = False
	flip_horizontal:     bool  = False  # reverse output
	gamma:               float = 1
	contrast:            float = 1
	brightness:          float = 1
	threshold:           float | None = None
	vector_render_scale: float = 32

	# Amplitude mapping
	min_level_db: float | None = None

	# Phase reconstruction
	phase_mode:        str = "random"
	griffin_lim_iters: int = 0

	# Post-processing
	fade_ms:   float = 15               # fade-in/out length in milliseconds

	#-=-=-=-#

	@property
	def hop_length(self) -> int:
		return max(1, int(self.n_fft * self.hop_ratio))

	@property
	def n_freq_bins(self) -> int:
		return self.n_fft // 2 + 1

	@property
	def resolved_freq_max(self) -> float:
		nyquist = self.sample_rate / 2

		if self.frequency_max is None or self.frequency_max < 0:
			return nyquist

		return min(self.frequency_max, nyquist)

	@property
	def resolved_freq_min(self) -> float:
		return max(0, self.frequency_min)

#-=-=-=-#

logger = Logger(os.path.basename(os.path.dirname(__file__)))