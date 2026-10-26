from __future__ import annotations

import random
import colorsys
import numpy as np

from PIL import Image

import matplotlib
matplotlib.set_loglevel(level = "warning")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from .img import save_animated_image
from ..core.geometry import ShapeWave

#-=-=-=-#
# Styling

SHAPE_OUTLINE_WIDTH = 1.5
SHAPE_GUIDE_WIDTH = 1.0
SHAPE_DOT_SIZE = 5.0

WAVE_TRACE_WIDTH = SHAPE_OUTLINE_WIDTH
WAVE_PROGRESS_WIDTH = SHAPE_GUIDE_WIDTH
WAVE_GUIDE_WIDTH = SHAPE_GUIDE_WIDTH
WAVE_DOT_SIZE = SHAPE_DOT_SIZE

#-=-=-=-#

plt.set_loglevel(level = "warning")
matplotlib.use("Agg")

def _build_figure(
	obj: ShapeWave,
	color: str,
	trace_full_wave: bool,
	fig_width: float,
	fig_height: float,
	dpi: int,
	left: bool,
	reverse: bool,
	shape_left: bool,
):
	"""
	Create the figure/axes once, draw everything that doesn't change
	frame-to-frame, and return handles to the mappings that do.
	"""
	xmin, xmax, ymin, ymax = obj.bounds()

	pad_x = 0.25 * (xmax - xmin + 1e-9)
	pad_y = 0.25 * (ymax - ymin + 1e-9)

	x_range_min = (xmax - xmin) + 2 * pad_x
	y_range_min = (ymax - ymin) + 2 * pad_y

	cx = (xmin + xmax) / 2
	cy = (ymin + ymax) / 2

	# Create subplots and assign based on position flag
	fig, (first_ax, second_ax) = plt.subplots(
		1, 2,
		figsize = (fig_width, fig_height),
		dpi = dpi,
		gridspec_kw = {"width_ratios": [1, 2] if left else [2, 1]},
	)

	if left:
		ax_shape, ax_wave = first_ax, second_ax
	else:
		ax_wave, ax_shape = first_ax, second_ax

	# transparent background
	fig.patch.set_alpha(0)

	ax_shape.set_facecolor((0, 0, 0, 0))
	ax_wave.set_facecolor((0, 0, 0, 0))

	dot_radius_in = (WAVE_DOT_SIZE / 2) / 72 # points -> inches
	right_margin = dot_radius_in / fig_width

	fig.subplots_adjust(
		left = 0 if left else right_margin,
		right = 1 - right_margin if left else 1,
		top = 1,
		bottom = 0,
		wspace = 0,
	)

	# left: shape
	pos = ax_shape.get_position()

	box_w_in = pos.width * fig_width
	box_h_in = pos.height * fig_height
	box_ratio = box_w_in / box_h_in # target x_range / y_range for equal pixel scale

	y_range = max(y_range_min, x_range_min / box_ratio)
	x_range = y_range * box_ratio

	x_lo, x_hi = cx - x_range / 2, cx + x_range / 2
	y_lo, y_hi = cy - y_range / 2, cy + y_range / 2

	ax_shape.plot(
		obj.x, obj.y,
		color = color,
		linewidth = SHAPE_OUTLINE_WIDTH,
	)

	shape_dot, = ax_shape.plot(
		[],
		[],
		"o",
		color = color,
		markersize = SHAPE_DOT_SIZE,
	)

	shape_guide, = ax_shape.plot(
		[],
		[],
		"--",
		color = "#888888",
		linewidth = SHAPE_GUIDE_WIDTH,
	)

	ax_shape.set_xlim(x_lo, x_hi)
	ax_shape.set_ylim(y_lo, y_hi)
	ax_shape.axis("off")

	if trace_full_wave:
		ax_wave.plot(
			obj.theta_grid,
			obj.y,
			color = color,
			linewidth = WAVE_TRACE_WIDTH,
		)

	wave_line, = ax_wave.plot(
		[],
		[],
		color = color,
		linewidth = WAVE_PROGRESS_WIDTH,
	)

	wave_dot, = ax_wave.plot([], 
		[],
		"o",
		color = color,
		markersize = WAVE_DOT_SIZE,
		clip_on = False,
	)

	wave_guide, = ax_wave.plot(
		[],
		[],
		"--",
		color = "#888888",
		linewidth = WAVE_GUIDE_WIDTH,
	)

	ax_wave.set_xlim(0, 2 * np.pi)
	ax_wave.set_ylim(y_lo, y_hi)
	ax_wave.axis("off")

	mappings = {
		"shape_dot": shape_dot,
		"shape_guide": shape_guide,
		"wave_line": wave_line,
		"wave_dot": wave_dot,
		"wave_guide": wave_guide,
	}

	# Guide line endpoint: when shape is on left, go right (x_hi); when on right, go left (x_lo)
	x_guide_end = x_hi if shape_left else x_lo
	
	return fig, mappings, x_guide_end

def _update_frame(
	obj: ShapeWave,
	progress: float,
	mappings: dict,
	x_guide_end: float,
	trace_full_wave: bool,
	shape_left: bool = True,
):
	"""
	Move the per-frame mappings to their position for this frame.

	`progress` is a position in [0, 2 * pi) along the
	generated cycle - the same phase-independent space as
	`obj.theta_grid` - not an absolute physical angle.

	The shape-panel dot/guide use `point_at_progress`, which folds
	in `obj.starting_point` automatically, so the dot correctly
	starts (and moves) from the shape's configured starting
	point rather than always from angle 0.
	"""
	if not shape_left:
		progress = (2 * np.pi - progress) % (2 * np.pi)

	px, py = obj.point_at_progress(progress)

	mappings["shape_dot"].set_data([px], [py])
	mappings["shape_guide"].set_data([px, x_guide_end], [py, py])

	if not trace_full_wave:
		mask = obj.theta_grid <= progress

		mappings["wave_line"].set_data(
			obj.theta_grid[mask],
			obj.y[mask],
		)

	cy = obj.y_at_progress(progress)

	mappings["wave_dot"].set_data([progress], [cy])

	if shape_left:
		mappings["wave_guide"].set_data([0, progress], [cy, cy])
	else:
		mappings["wave_guide"].set_data([2 * np.pi, progress], [cy, cy])

def _figure_to_frame(fig) -> np.ndarray:
	"""
	Return RGBA image data.
	"""
	fig.canvas.draw()

	return np.array(
		fig.canvas.buffer_rgba(),
		copy = True,
	)

def _random_color():
	h = random.random()

	r, g, b = colorsys.hsv_to_rgb(h, 1, 1)

	return "#{:02X}{:02X}{:02X}".format(
		round(r * 255), 
		round(g * 255),
		round(b * 255),
	)

#-=-=-=-#

def render(
	wave: ShapeWave,
	path_out: str,
	n_frames: int = 300,
	fps: int = 50,
	color: str | None = None,
	fig_width: float = 8,
	fig_height: float = 3.2,
	dpi: int = 220,
	trace_full_wave: bool = True,
	crop: bool = True,
	reverse: bool = False,
	left: bool = True,
) -> str:
	if color:
		color = "#" + color.strip("#")
	else:
		color = _random_color()

	color = to_hex(color)

	fig, mappings, x_guide_end = _build_figure(
		wave,
		color, trace_full_wave,
		fig_width, fig_height, dpi,
		left, reverse, left
	)

	# progress values, not absolute physical angles - see _update_frame.
	progress_values = np.linspace(0, 2 * np.pi, n_frames, endpoint = False)

	if reverse:
		progress_values = progress_values[::-1]

	frames = []

	try:
		for progress in progress_values:
			_update_frame(wave, progress, mappings, x_guide_end, trace_full_wave, left)

			frames.append(
				Image.fromarray(_figure_to_frame(fig), mode = "RGBA")
			)
	finally:
		plt.close(fig)

	if crop:
		global_bbox = None

		for frame in frames:
			bbox = frame.getchannel("A").getbbox()

			if bbox is None:
				continue

			if global_bbox is None:
				global_bbox = list(bbox)
			else:
				global_bbox[0] = min(global_bbox[0], bbox[0]) # left
				global_bbox[1] = min(global_bbox[1], bbox[1]) # top
				global_bbox[2] = max(global_bbox[2], bbox[2]) # right
				global_bbox[3] = max(global_bbox[3], bbox[3]) # bottom

		if global_bbox:
			global_bbox = tuple(global_bbox)
			frames = [frame.crop(global_bbox) for frame in frames]

	save_animated_image(frames, path_out, fps)

	return path_out