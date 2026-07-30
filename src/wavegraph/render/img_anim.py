from __future__ import annotations

from PIL import Image

from ..._global.setup import Logger
log = Logger("oscigraph")

#-=-=-=-#

def fps_to_gif_duration_ms(fps: float) -> tuple[int, float]:
	"""
	Convert a target FPS to the nearest valid GIF frame duration.

	GIF delays are stored in centiseconds, so the true achievable
	duration is always a multiple of 10 ms. Returns the duration to
	pass to Pillow, and the actual fps that duration corresponds to.

	Returns
	-------
		(duration_ms, actual_fps)
	"""
	if fps <= 0:
		raise ValueError("fps must be > 0")

	exact_ms = 1000 / fps
	centiseconds = max(1, round(exact_ms / 10)) # never allow 0 -> stuck GIF
	duration_ms = centiseconds * 10
	actual_fps = 1000 / duration_ms

	return duration_ms, actual_fps

def save_animated_image(
	frames: list[Image.Image],
	path_out: str,
	fps: float,
	loop: int = 0,
	disposal: int = 2,
	optimize: bool = True,
	transparency: int | None = 0,
	warn_threshold_pct: float = 5.0,
) -> str:
	"""
	Save frames as a GIF with a correctly computed frame duration.

	Prints a warning if the achievable fps drifts from the requested
	fps by more than `warn_threshold_pct` percent (this happens for
	fps values above ~50, since GIF can't represent them precisely).
	"""
	if not frames:
		raise ValueError("frames list is empty")

	duration_ms, actual_fps = fps_to_gif_duration_ms(fps)

	drift_pct = abs(actual_fps - fps) / fps * 100
	if drift_pct > warn_threshold_pct:
		log.warning(
			f"{fps:.2f} FPS not representable in GIF (10ms tick resolution). "
			f"Limited to {actual_fps:.2f}."
		)

	save_kwargs = dict(
		save_all = True,
		append_images = frames[1:],
		duration = duration_ms,
		loop = loop,
		disposal = disposal,
		optimize = optimize,
	)

	if transparency is not None:
		save_kwargs["transparency"] = transparency

	frames[0].save(path_out, **save_kwargs)

	return path_out

def fix_gif_duration(path_in: str, fps: float, path_out: str | None = None) -> str:
	"""
	Re-export an existing image with corrected per-frame
	duration, without needing the original frame data.

	If path_out is None, overwrites path_in.
	"""
	duration_ms, actual_fps = fps_to_gif_duration_ms(fps)

	with Image.open(path_in) as im:
		frames = []

		try:
			while True:
				frames.append(im.copy().convert("RGBA"))
				im.seek(im.tell() + 1)
		except EOFError:
			pass

	out_path = path_out or path_in
	frames[0].save(
		out_path,
		save_all = True,
		append_images = frames[1:],
		duration = duration_ms,
		loop = 0,
		disposal = 2,
		optimize = True,
		transparency = 0,
	)

	return out_path

if __name__ == "__main__":
	# quick sanity check of the fps -> duration table
	for test_fps in (12, 15, 24, 25, 30, 50, 60, 90, 120):
		ms, actual = fps_to_gif_duration_ms(test_fps)
		log.debug(f"requested={test_fps:>4} FPS -> duration={ms:>3} ms -> actual={actual:.2f} FPS")