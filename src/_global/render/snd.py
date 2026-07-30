import wave
import numpy as np

#-=-=-=-#

def write(
	signal: np.ndarray,
	path: str,
	sample_rate: int,
	normalize: bool = False
) -> str:
	"""
	Writes sound.
	"""
	signal = signal.astype(np.float32)

	if normalize:
		signal = np.clip(
			signal * 2147483647,
			-2147483648,
			2147483647
		).astype(np.int32)

	with wave.open(str(path), "wb") as wav:
		wav.setnchannels(1 if signal.ndim == 1 else signal.shape[1])
		wav.setsampwidth(4)
		wav.setframerate(sample_rate)
		wav.writeframes(signal.tobytes())

	return path