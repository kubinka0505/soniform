import wave
import struct

import numpy as np

import wavemarks

#-=-=-=-#

def write_sound(
	path: str,

	left: np.ndarray,
	right: np.ndarray,

	sample_rate: int,
	boundaries: tuple
) -> None:
	n = len(left)

	interleaved = np.empty(n * 2, dtype = np.int32)

	interleaved[0::2] = np.int32(np.round(left * 2147483647))
	interleaved[0::2] = np.int32(np.round(left * 2147483647))
	interleaved[1::2] = np.int32(np.round(right * 2147483647))

	with wave.open(path, "wb") as wf:
		wf.setnchannels(2)
		wf.setsampwidth(4)
		wf.setframerate(sample_rate)
		wf.writeframes(struct.pack("<%di" % len(interleaved), *interleaved))

	if boundaries:
		mf = wavemarks.MarkerFile(path)

		start = wavemarks.Entry(
			boundaries.pos_start_attack,
			boundaries.pos_end_attack,
			name = "Attack"
		)

		first_cycle = wavemarks.Entry(
			boundaries.pos_start_first_cycle,
			boundaries.pos_end_first_cycle,
			name = "1st Cycle"
		)

		release = wavemarks.Entry(
			boundaries.pos_start_release,
			boundaries.pos_end_release,
			name = "Release"
		)

		mf += start
		mf += first_cycle
		mf += release

		mf.save()
	