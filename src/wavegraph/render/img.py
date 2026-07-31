from __future__ import annotations

import random
import colorsys
import numpy as np

from PIL import Image

import matplotlib
matplotlib.set_loglevel(level = "warning")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from .img_anim import save_animated_image
from ..core.geometry import ShapeWave

#-=-=-=-#
# Styling

SHAPE_OUTLINE_WIDTH = 1.5
SHAPE_GUIDE_WIDTH = 1.0
SHAPE_DOT_SIZE = 5

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
	dpi: int
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


	fig, (ax_shape, ax_wave) = plt.subplots(
		1, 2,
		figsize = (fig_width, fig_height),
		dpi = dpi,
		gridspec_kw = {"width_ratios": [1, 2]},
	)

	# transparent background
	fig.patch.set_alpha(0)

	ax_shape.set_facecolor((0, 0, 0, 0))
	ax_wave.set_facecolor((0, 0, 0, 0))

	dot_radius_in = (WAVE_DOT_SIZE / 2) / 72 # points -> inches
	right_margin = dot_radius_in / fig_width

	fig.subplots_adjust(
		left = 0,
		right = 1 - right_margin,
		top = 1,
		bottom = 0,
		wspace = 0,
	)

	# left: shape
	pos = ax_shape.get_position()

	box_w_in = pos.width * fig_width
	box_h_in = pos.height * fig_height
	box_ratio = box_w_in / box_h_in  # target x_range / y_range for equal pixel scale

	# There's no divisor that makes a single fixed y_range simultaneously
	# (a) equal-aspect for this box and (b) big enough on x to fit the
	# shape - if the box is narrower than the shape needs, one of those
	# has to give. The old fallback gave up on (a), fitting x to the
	# shape and leaving x-units-per-inch != y-units-per-inch (the
	# "squish"). Instead, grow y_range (and x_range with it, in lockstep
	# via box_ratio) until it's big enough for *both* constraints at
	# once. This y_range is then reused for ax_wave below, so the two
	# panels always share the same vertical scale and the guide line
	# still lines up - any extra padding beyond pad_x/pad_y is the
	# minimum geometrically required to keep the shape both equal-aspect
	# and fully visible in this box, not a leftover magic-number fudge.
	y_range = max(y_range_min, x_range_min / box_ratio)
	x_range = y_range * box_ratio

	x_lo, x_hi = cx - x_range / 2, cx + x_range / 2
	y_lo, y_hi = cy - y_range / 2, cy + y_range / 2

	# Still no ax_shape.set_aspect("equal") - x_range/y_range above is
	# already derived from the box's real physical width/height, so
	# equal aspect is already baked into the numbers. set_aspect would
	# re-shrink/reposition the box on top of this and desync the guide
	# line again (see earlier note history on this function).
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

	# right: wave -- shares y_lo/y_hi with the shape panel above, not a
	# separately-computed pad_y range, so the two stay in sync even when
	# y_range had to grow to fit the shape.
	#
	# obj.theta_grid is plotted here (rather than an absolute physical
	# angle) because it's guaranteed monotonically increasing over
	# [0, 2*pi) regardless of `starting_point` - see ShapeWave's
	# docstring. Plotting a starting-point-rotated angle here instead
	# would reintroduce a backward jump partway through the array
	# (wherever the rotation wraps past 2*pi), which renders as a
	# spurious straight line cutting across this panel.
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

	wave_dot, = ax_wave.plot(
		[],
		[],
		"o",
		color = color,
		markersize = WAVE_DOT_SIZE,
		# `progress` sits exactly on the axes' left edge (x = 0) at the
		# start of the cycle, and matplotlib clips markers to the axes
		# by default - so without this, the dot would render as half a
		# circle right at the seam. Only this artist needs it; the
		# trace/guide lines should stay clipped normally.
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

	return fig, mappings, x_hi

def _update_frame(
	obj: ShapeWave,
	progress: float,
	mappings: dict,
	x_guide_end: float,
	trace_full_wave: bool,
):
	"""
	Move the per-frame mappings to their position for this frame.

	`progress` is a position in [0, 2*pi) along the generated cycle -
	the same phase-independent space as `obj.theta_grid` - not an
	absolute physical angle. The shape-panel dot/guide use
	`point_at_progress`, which folds in `obj.starting_point`
	automatically, so the dot correctly starts (and moves) from the
	shape's configured starting point rather than always from angle 0.
	"""
	px, py = obj.point_at_progress(progress)

	mappings["shape_dot"].set_data(
		[px],
		[py],
	)

	mappings["shape_guide"].set_data(
		[px, x_guide_end],
		[py, py],
	)

	if not trace_full_wave:
		mask = obj.theta_grid <= progress

		mappings["wave_line"].set_data(
			obj.theta_grid[mask],
			obj.y[mask],
		)

	cy = obj.y_at_progress(progress)

	mappings["wave_dot"].set_data(
		[progress],
		[cy],
	)

	mappings["wave_guide"].set_data(
		[0, progress],
		[cy, cy],
	)

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
	crop: bool = True
) -> str:
	if color:
		color = "#" + color.strip("#")
	else:
		color = _random_color()

	color = to_hex(color)

	fig, mappings, x_guide_end = _build_figure(
		wave,
		color,
		trace_full_wave,
		fig_width,
		fig_height,
		dpi,
	)

	# progress values, not absolute physical angles - see _update_frame.
	progress_values = np.linspace(
		0,
		2 * np.pi,
		n_frames,
		endpoint = False,
	)

	frames = []

	try:
		for progress in progress_values:
			_update_frame(
				wave,
				progress,
				mappings,
				x_guide_end,
				trace_full_wave,
			)

			frames.append(
				Image.fromarray(
					_figure_to_frame(fig),
					mode = "RGBA",
				)
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

		if global_bbox is not None:
			global_bbox = tuple(global_bbox)

			frames = [
				frame.crop(global_bbox)
				for frame in frames
			]

	save_animated_image(frames, path_out, fps)

	return path_out