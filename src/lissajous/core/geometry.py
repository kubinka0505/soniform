from __future__ import annotations

import numpy as np

from svgelements import SVG, Path as SVGPath, Shape as SVGShape

from ..config import Logger as log

#-=-=-=-#

class Shape:
	"""A closed 2D point loop, ready to be traced by the oscillator."""

	def __init__(self, xs: np.ndarray, ys: np.ndarray):
		# xs/ys are "closed": xs[-1] == xs[0], ys[-1] == ys[0].
		self.xs = xs
		self.ys = ys

	#-=-=-=-#
	# construction

	@classmethod
	def from_shape(cls, path: str, flatten_samples: int, shape_points: int):
		svg = SVG.parse(path)
		shapes = [e for e in svg.elements() if isinstance(e, (SVGPath, SVGShape))]

		if not shapes:
			raise ValueError(f'No paths/shapes found in "{path}".')

		if len(shapes) > 1:
			log.warning(
				f"{len(shapes)} sub-paths/shapes found in the shape; using the first one.",
				UserWarning
			)

		shape_path = shapes[0]
		if not isinstance(shape_path, SVGPath):
			shape_path = SVGPath(shape_path) # convert e.g. circle/rect/polygon to a Path

		# Densely sample the (possibly curved) path parametrically.
		ts = np.linspace(0, 1, flatten_samples, endpoint = True)
		raw_pts = [shape_path.point(t) for t in ts] # Point objects, not complex numbers
		xs = np.array([p.x for p in raw_pts])
		ys = np.array([p.y for p in raw_pts])

		# svgelements has no isclosed(); check first/last sampled point instead.
		if not np.isclose(xs[0], xs[-1]) or not np.isclose(ys[0], ys[-1]):
			log.warning(
				"Selected path does not appear to be closed; it will be force-closed by connecting its end back to its start.",
				UserWarning
			)

			xs = np.append(xs, xs[0])
			ys = np.append(ys, ys[0])

		shape = cls(xs, ys)
		shape._resample_uniform_arclength(shape_points)

		return shape

	# transforms

	def normalize(self, margin: float = 0.9):
		"""
		Center at the origin, flip Y into math/audio orientation, scale to fit.
		"""
		cx = (self.xs.max() + self.xs.min()) / 2
		cy = (self.ys.max() + self.ys.min()) / 2

		self.xs = self.xs - cx
		self.ys = -(self.ys - cy) # Y grows downward
		self._fit_to_margin(margin)

		return self

	def rotate(self, degrees: float):
		"""
		Rotate the shape counter-clockwise around the origin, then re-fit.
		"""
		if not degrees % 360:
			return self

		theta = np.radians(degrees)
		cos_t, sin_t = np.cos(theta), np.sin(theta)

		xs, ys = self.xs, self.ys

		self.xs = xs * cos_t - ys * sin_t
		self.ys = xs * sin_t + ys * cos_t

		self._fit_to_margin(self._last_margin)

		return self

	def roll_start(self, index: int):
		"""
		Rotate the *point order* (not geometry) so tracing begins at `index`.
		"""
		n = len(self.xs) - 1 # last point duplicates the first (closed loop)

		if n <= 0:
			return self

		index %= n

		if not index:
			return self

		open_xs, open_ys = self.xs[:-1], self.ys[:-1]

		open_xs = np.roll(open_xs, -index)
		open_ys = np.roll(open_ys, -index)

		self.xs = np.append(open_xs, open_xs[0])
		self.ys = np.append(open_ys, open_ys[0])

		return self

	def reverse(self):
		self.xs = self.xs[::-1].copy()
		self.ys = self.ys[::-1].copy()
		return self

	# internals

	def _fit_to_margin(self, margin: float) -> None:
		self._last_margin = margin
		extent = max(self.xs.max() - self.xs.min(), self.ys.max() - self.ys.min())

		if not extent:
			raise ValueError("Degenerate shape with zero width and height.")

		scale = (2 * margin) / extent
		self.xs = self.xs * scale
		self.ys = self.ys * scale

	def _resample_uniform_arclength(self, n_points: int) -> None:
		cum_len = self.cumulative_arclength()
		total_len = cum_len[-1]

		if total_len <= 0:
			raise ValueError("Degenerate shape: zero total path length.")

		target = np.linspace(0, total_len, n_points, endpoint = True)

		self.xs = np.interp(target, cum_len, self.xs)
		self.ys = np.interp(target, cum_len, self.ys)

	def cumulative_arclength(self) -> np.ndarray:
		dx, dy = np.diff(self.xs), np.diff(self.ys)
		seg_len = np.hypot(dx, dy)

		return np.concatenate(([0], np.cumsum(seg_len)))