from typing import Optional, Tuple

import numpy as np

from .geometry import Shape
from ..config import Settings, Positions, Logger

#-=-=-=-#

def build_envelope(
	t: np.ndarray,
	attack: float,
	duration: float,
	release: float
) -> np.ndarray:
	"""
	Piecewise-linear "coverage fraction" envelope.

	For each sample time in `t`, describes how much of the shape's total
	arc length (measured from point index 0) should currently be traced:
	ramps 0 -> 1 over `attack`, holds at 1 for `duration`, ramps 1 -> 0
	over `release`.

	Parameters
	----------
		t (np.ndarray):
			Sample times in seconds, monotonically increasing.

		attack (float):
			Seconds to ramp from 0 to full coverage.

			0 skips the ramp.

		duration (float):
			Seconds to hold at full coverage.

		release (float):
			Seconds to ramp back down to 0.

			0 skips the ramp.

	Returns:
		Array the same shape as `t`, values clipped to [0, 1].
	"""
	frac = np.ones_like(t)

	if attack > 0:
		in_attack = t < attack
		frac[in_attack] = t[in_attack] / attack

	release_start = attack + duration

	if release > 0:
		in_release = t >= release_start
		frac[in_release] = 1 - (t[in_release] - release_start) / release

	return np.clip(frac, 0, 1)

def first_cycle_bounds(
	settings: Settings,
	sample_rate: int,
) -> Optional[Tuple[int, int]]:
	"""
	Sample-index bounds of the first full waveform cycle, in the
	final non-oversampled output sample domain.

	The rendered signal is structured as:
	- [attack samples][first cycle][rest of cycles][release samples]

	This locates the `[first cycle]` block so a single loopable period of
	the shape can be region-marked / extracted after the WAV is written
	(e.g. via wavemarks) without having to re-derive the timing there.

	Special cases, matching `synthesize`:
		- `settings.duration == 0`:
			There is no sustain region at all, so no full
			cycle ever plays at full coverage.

			Returns `None`.
		- `settings.duration < 0`:
			"single-cycle mode" - the sustain region is forced to
			exactly one period (`1 / frequency` seconds), regardless
			of the magnitude of the negative value, and that whole
			region is the first (and only) cycle.

		- `0 < settings.duration < 1 / frequency`:
			The sustain region exists but isn't long
			enough to contain one full cycle.
			
			Returns `None` rather than an unbounded/misleading region.

	Parameters
	----------
		settings:
			Render settings (`attack`, `duration`, `frequency`).

		sample_rate (int):
			Output sample rate in Hz - the final rate the sound
			file is written at, not the oversampled render rate.

	Returns
	-------
		`(start_sample, end_sample)`:
			end-exclusive
		None:
			If no full cycle exists to bound.
	"""
	if settings.duration == 0:
		return None

	cycle_length_sec = 1 / settings.frequency
	sustain_length_sec = (
		cycle_length_sec if settings.duration < 0 else settings.duration
	)

	if sustain_length_sec < cycle_length_sec:
		return None

	start_sample = round(settings.attack * sample_rate)
	end_sample = start_sample + round(cycle_length_sec * sample_rate)

	return start_sample, end_sample

#-=-=-=-#

def _resolve_effective_duration(settings: Settings) -> float:
	"""
	Resolve `settings.duration`'s special values into a real,
	non-negative sustain length in seconds.

	`0` means no sustain region at all.

	A negative value means "single-cycle mode": the sustain region is
	forced to exactly one period (`1 / frequency` seconds), regardless
	of the magnitude of the negative number.

	Parameters
	-----------
		settings:
			Render settings.

	Returns
	-------
		float:
			Sustain length in seconds.
	"""
	if settings.duration < 0:
		return 1 / settings.frequency

	return settings.duration

def build_cycle_positions(
	settings: Settings,
	sample_rate: int,
	total_samples: int,
) -> Positions:
	"""
	Compute the attack/first-cycle/release sample boundaries for a rendered waveform.

	Special cases, matching `synthesize` / `build_envelope`:
		- `settings.duration == 0`:
			No sustain region exists, so there is no full cycle at full coverage.
			`pos_start_first_cycle` and `pos_end_first_cycle` are `None`.

		- `settings.duration < 0`:
			"single-cycle mode" - the sustain region is exactly one period long,
			and that whole region is the first (and only) cycle.

		- `0 < settings.duration < 1 / frequency`:
			Sustain region exists but is shorter than one full cycle.

			`pos_start_first_cycle` / `pos_end_first_cycle` are `None`
			rather than describing a partial cycle.

	Parameters
	-----------
		settings:
			Render settings (`attack`, `duration`, `release`, `frequency`).

		sample_rate (int):
			Output sample rate in Hz - the final rate the sound is written at,
			not the oversampled render rate.

		total_samples (int):
			Actual number of samples in the rendered signal (`len(left)`),
			used for `pos_end_release` so the returned bounds always
			agree with the real array length regardless of
			independent floating-point rounding.

	Returns
	-------
		`Positions`:
			Populated positions.
	"""
	cycle_length_sec = 1 / settings.frequency
	effective_duration = _resolve_effective_duration(settings)

	pos_start_attack = 0
	pos_end_attack = round(settings.attack * sample_rate)

	sustain_start_pos = pos_end_attack
	sustain_end_pos = pos_end_attack + round(effective_duration * sample_rate)

	if effective_duration <= 0 or effective_duration < cycle_length_sec:
		pos_start_first_cycle: Optional[int] = None
		pos_end_first_cycle: Optional[int] = None
	else:
		pos_start_first_cycle = sustain_start_pos
		pos_end_first_cycle = sustain_start_pos + round(cycle_length_sec * sample_rate)

	pos_start_release = sustain_end_pos
	pos_end_release = total_samples

	return Positions(
		pos_start_attack = pos_start_attack,
		pos_end_attack = pos_end_attack,

		pos_start_first_cycle = pos_start_first_cycle,
		pos_end_first_cycle = pos_end_first_cycle,

		pos_start_release = pos_start_release,
		pos_end_release = pos_end_release,
	)

def synthesize(
	shape: Shape,
	settings: Settings,
) -> Tuple[np.ndarray, np.ndarray, Positions]:
	"""
	Render the stereo (left = X, right = Y) waveform in range [-1, 1].

	Internally rendered at `settings.sample_rate * oversampling` and then
	box-filtered back down, to reduce aliasing at high `-frequency`
	values.

	`settings.duration` has two special values, resolved here
	(not on `settings` itself):
		- `0`: no sustain region - the signal goes straight from the
		  attack peak into release, with no time spent at full coverage.
		- `< 0`: "single-cycle mode" - the sustain region is forced to
		  exactly one period (`1 / frequency` seconds) regardless of the
		  actual negative value, so the render contains exactly one full
		  cycle of the shape.

	Parameters
	----------
		shape:
			The closed, normalized point loop to trace.

		settings:
			Render settings (attack/duration/release, frequency,
			sample_rate, and oversampling parameters).

	Returns
	-------
		tuple:
			- `left`:
				left-channel (X) samples, shape `(n_samples,)`.
			- `right`:
				right-channel (Y) samples, shape `(n_samples,)`.
			- `cycle_bounds`:
				`(start_sample, end_sample)` of the first full
				cycle in the final output, or `None` if none exists
				(see `first_cycle_bounds`).
	"""
	def log_array(name: str, arr: np.ndarray) -> None:
		Logger.debug(
			"%s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f rms=%.6f nan=%s inf=%s",
			name,
			arr.shape,
			arr.dtype,
			np.min(arr),
			np.max(arr),
			np.mean(arr),
			np.sqrt(np.mean(arr ** 2)),
			np.isnan(arr).any(),
			np.isinf(arr).any(),
		)

		Logger.debug("-" * 32)

	effective_duration = _resolve_effective_duration(settings)
	effective_total_duration = settings.attack + effective_duration + settings.release

	Logger.info(
		"settings: sample_rate=%d duration=%.3fs (effective=%.3fs) frequency=%.3fHz",
		settings.sample_rate,
		settings.duration,
		effective_duration,
		settings.frequency,
	)

	cum_len = shape.cumulative_arclength()
	total_len = cum_len[-1]

	Logger.info(
		"shape: points=%d arc_length=%.6f",
		len(shape.xs),
		total_len,
	)

	oversampling = 64
	Logger.info("oversampling factor=%d", oversampling)

	hi_rate = settings.sample_rate * oversampling
	n_hi = max(oversampling, int(round(effective_total_duration * hi_rate)))

	Logger.info(
		"high-rate render: sample_rate=%d samples=%d duration=%.6fs",
		hi_rate,
		n_hi,
		n_hi / hi_rate,
	)

	t_hi = np.arange(n_hi) / float(hi_rate)
	log_array("time", t_hi)

	Logger.debug("building envelope")
	frac = build_envelope(
		t_hi,
		settings.attack,
		effective_duration,
		settings.release,
	)
	log_array("envelope", frac)

	sub_length = frac * total_len
	log_array("sub_length", sub_length)

	Logger.debug("building phase")
	phase = np.mod(t_hi * settings.frequency, 1)
	log_array("phase", phase)

	Logger.debug("mapping phase to shape")
	s = np.clip(phase * sub_length, 0, total_len)
	log_array("shape_position", s)

	Logger.debug("interpolating left channel")
	left_hi = np.interp(s, cum_len, shape.xs)
	log_array("left_hi", left_hi)

	Logger.debug("interpolating right channel")
	right_hi = np.interp(s, cum_len, shape.ys)
	log_array("right_hi", right_hi)

	Logger.debug("decimating left")
	left = _decimate(left_hi, oversampling)
	log_array("left", left)

	Logger.debug("decimating right")
	right = _decimate(right_hi, oversampling)
	log_array("right", right)

	Logger.debug("clipping output")
	left = np.clip(left, -1, 1)
	right = np.clip(right, -1, 1)

	log_array("left_final", left)
	log_array("right_final", right)

	cycle_positions = build_cycle_positions(settings, settings.sample_rate, len(left))
	Logger.info("cycle_positions=%s", cycle_positions)

	Logger.info(
		"synthesize complete: samples=%d duration=%.6fs",
		len(left),
		len(left) / settings.sample_rate,
	)

	return left, right, cycle_positions

def _decimate(signal: np.ndarray, factor: int) -> np.ndarray:
	"""
	Box-filter decimation: average every `factor` consecutive samples
	down to a single sample.

	Parameters
	----------
		signal (np.ndarray):
			1-D input array.
		factor:
		    Number of input samples averaged per output sample.
			Values `<= 1` are a no-op (input returned unchanged).

	Returns
	-------
		np.ndarray:
			Decimated array.
		    If `len(signal)` isn't a multiple of `factor`,
			the trailing `len(signal) % factor` samples
			are dropped before averaging.
	"""
	if factor <= 1:
		return signal

	n = (len(signal) // factor) * factor

	return signal[:n].reshape(-1, factor).mean(axis = 1)