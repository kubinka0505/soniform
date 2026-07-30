from pathlib import Path

from ...config import Settings
from ..img.preload import image_to_intensity_array
from .construction import build_magnitude_spectrogram, render
from .post import post_process

from ...config import logger as log
from ...._global.render.snd import write

#-=-=-=-#

def convert(
	input_path: Path,
	output_path: Path,
	cfg: Settings
) -> None:
	"""
	Execute the complete image-to-audio conversion pipeline.

	Pipeline stages:
	1. Load and preprocess the image.
	2. Convert image intensity into spectrogram magnitude.
	3. Construct the full frequency-domain representation.
	4. Reconstruct audio from the spectrogram.
	5. Apply post-processing.
	6. Write the resulting WAV file.

	Parameters
	----------
	input_path (Path)
		Source image file.

	output_path (Path):
		Destination sound file.

	cfg (Settings):
		Conversion settings.
	"""
	n_time_frames = max(2, int((cfg.duration * cfg.sample_rate) / cfg.hop_length))

	input_path = Path(input_path)

	intensity = image_to_intensity_array(input_path, cfg, n_time_frames = n_time_frames)
	magnitude = build_magnitude_spectrogram(intensity, cfg, n_time_frames)
	audio = render(magnitude, cfg)
	audio = post_process(audio, cfg)
	write(audio, output_path, cfg.sample_rate, normalize = True)

	log.info(f"duration           : {cfg.duration:.3f}s")
	log.info(f"sample rate        : {cfg.sample_rate} Hz")
	log.info(f"n_fft | hop length : {cfg.n_fft} | {cfg.hop_length}")
	log.info(f"frequency band     : {cfg.resolved_freq_min} to {cfg.resolved_freq_max} Hz ({cfg.frequency_mode})")
	log.info(f"griffin-lim        : {cfg.griffin_lim_iters} iterations")