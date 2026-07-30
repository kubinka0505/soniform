import io
import re
import numpy as np
from pathlib import Path
from resvg_py import svg_to_bytes
from PIL import Image, ImageOps, ImageEnhance

from ...config import Settings

#-=-=-=-#

def _guess_image_width(img_text: str) -> float:
	"""
	Extraction of the SVG's intrinsic width, falling back to a sane
	default if it can't be determined (percentage widths, etc).
	"""
	m = re.search(r'width\s*=\s*"([\d.]+)', img_text)

	if m:
		try:
			return float(m.group(1))
		except ValueError:
			pass

	m = re.search(r'viewBox\s*=\s*"[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+[\d.]+"', img_text)

	if m:
		try:
			return float(m.group(1))
		except ValueError:
			pass

	return 512

def _pad_image(img: Image.Image, width_ratio: float = 1, height_ratio: float = 1) -> Image.Image:
    """
	Pad image canvas to a scale ratio while centering the original content.
	"""
    if width_ratio == 1 and height_ratio == 1:
        return img

    new_w = max(img.width, int(round(img.width * width_ratio)))
    new_h = max(img.height, int(round(img.height * height_ratio)))
    
    if new_w == img.width and new_h == img.height:
        return img

    bg_color = (0, 0, 0, 0) if "A" in img.mode or img.mode == "P" else 0
    canvas = Image.new(img.mode, (new_w, new_h), bg_color)
    
    # Calculate exact offsets
    offset_x = (new_w - img.width) // 2
    offset_y = (new_h - img.height) // 2

    canvas.paste(img, (offset_x, offset_y))
    return canvas

def _load_raw_image(path: Path, cfg: Settings) -> Image.Image:
	"""
	Load the source file as-is (mode preserved, including alpha), then pad canvas.
	"""
	if path.suffix.lower() == ".svg":
		img_text = path.read_text(encoding = "UTF-8")
		base_width = _guess_image_width(img_text)

		render_width = max(64, int(base_width * cfg.vector_render_scale))

		rendered = svg_to_bytes(
			svg_string = img_text,
			width = render_width,
			shape_rendering = "geometric_precision",
			text_rendering = "geometric_precision",
			image_rendering = "optimize_quality"
		)

		img = Image.open(io.BytesIO(bytes(rendered)))
	else:
		img = Image.open(path)

	return img

def extract_intensity_layer(img: Image.Image) -> tuple[Image.Image, str]:
	"""
	Return (grayscale 'L' image, source_mode) representing drawing
	intensity, where 255 = fully "on" (loud) and 0 = fully "off" (silent).

	If the image carries real transparency, the alpha channel IS the
	intensity: transparent pixels are silent regardless of their fill
	color, and opaque pixels are loud regardless of whether that fill is
	dark or light. This is what makes transparent-background watermark
	art (a single shape on nothing) come out clean instead of producing a
	loud "background noise" from the old white-flattened brightness map.

	If the image has no meaningful transparency, intensity falls back to
	plain grayscale brightness (bright pixel = loud).
	"""
	has_alpha = img.mode in ("RGBA", "LA") or (
		img.mode == "P" and "transparency" in img.info
	)

	if has_alpha:
		rgba = img.convert("RGBA")
		alpha = rgba.getchannel("A")

		if np.asarray(alpha, dtype = np.uint8).min() < 250:
			return alpha, "alpha"

		# fully opaque despite having an alpha channel -> flatten + brightness
		bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
		flat = Image.alpha_composite(bg, rgba)

		return flat.convert("L"), "brightness"

	return img.convert("L"), "brightness"

def image_to_intensity_array(
	path: Path,
	cfg: Settings,
	n_time_frames: int
) -> np.ndarray:
	"""
	Load an image and convert it into a normalized intensity matrix.

	The image is loaded, optionally adjusted (contrast, brightness,
	inversion, thresholding, gamma correction, and flipping), then
	resized to match the requested spectrogram time resolution.

	Returns
	-------
	np.ndarray:
		Array of shape (height, n_time_frames) containing values in
		the range [0, 1].

		- Columns correspond to spectrogram time frames.
		- Rows correspond to image rows mapped to frequencies.
		- Row 0 represents the lowest frequency after preprocessing.
		- 0.0 means silent / inactive.
		- 1.0 means maximum intensity.
	"""
	raw = _load_raw_image(path, cfg)
	layer, _source = extract_intensity_layer(raw)

	if cfg.contrast != 1:
		layer = ImageEnhance.Contrast(layer).enhance(cfg.contrast)
	if cfg.brightness != 1:
		layer = ImageEnhance.Brightness(layer).enhance(cfg.brightness)
	if cfg.invert:
		layer = ImageOps.invert(layer)
	if cfg.flip_horizontal:
		layer = ImageOps.mirror(layer)
	if not cfg.flip_vertical:
		layer = ImageOps.flip(layer)

	n_freq_bins_source = layer.height
	layer = layer.crop(layer.getbbox())
	layer = _pad_image(layer, 1.05, 1)
	layer = layer.resize((n_time_frames, n_freq_bins_source), Image.BILINEAR)

	arr = np.asarray(layer, dtype = np.float64) / 255

	if cfg.threshold is not None:
		arr = (arr >= (cfg.threshold / 255)).astype(np.float64)

	if cfg.gamma != 1:
		arr = np.clip(arr, 0, 1) ** (1 / cfg.gamma)

	return arr