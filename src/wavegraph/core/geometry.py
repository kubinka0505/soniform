from __future__ import annotations

import math
import numpy as np
from svgelements import SVG, Path, Shape

from ..config import logger as log

#-=-=-=-#

def _iter_shapes(file: SVG):
	for element in file.elements():
		if isinstance(element, Shape) and not isinstance(element, SVG):
			yield element

def _has_fill(shape: Shape) -> bool:
	"""
	True for an actual filled shape (an enclosed area); False for a
	"border" - a stroke-only outline with fill = "none" (or no fill at all).
	"""
	fill = getattr(shape, "fill", None)
	value = getattr(fill, "value", fill)

	return value is not None

def _close_path(path: Path) -> None:
	"""
	Bridge the gap between a path's last point and its first point with a
	straight line, in place, if it isn't already closed. This is only
	meant for "border" elements (see _has_fill) that describe an outline
	rather than a filled area - the rest of the pipeline needs a closed
	loop to walk all the way around.
	"""
	start = path.first_point
	end = path.current_point

	if start is None or end is None:
		return

	if math.hypot(start.x - end.x, start.y - end.y) > 1e-6:
		path.line(start)

def load_first_shape(file_path: str) -> Path:
	"""
	Load an image file and return its single drawable shape as a Path,
	with any transforms baked in.

	Rules:
	- More than one filled shape (or, lacking any filled shape, more than
	  one border) is an error - only one shape per file is supported.
	- If both a filled shape and a border are present, the filled shape
	  wins and the border is ignored.
	- If there's only a border (no filled shape), it's converted into a
	  shape by closing it (see _close_path).
	- An SVG with no elements at all, or none that are drawable shapes,
	  is an error.
	"""
	img = SVG.parse(file_path)

	if not any(not isinstance(e, SVG) for e in img.elements()):
		raise ValueError(f'Shape file "{file_path}" is empty.')

	all_shapes = list(_iter_shapes(img))
	shapes = [s for s in all_shapes if _has_fill(s)]
	borders = [s for s in all_shapes if not _has_fill(s)]

	if len(shapes) > 1:
		raise ValueError(f'Expected exactly 1 shape in "{file_path}", found {len(shapes)}.')

	if shapes:
		element = shapes[0]
		is_border = False
	elif len(borders) > 1:
		raise ValueError(
			f'Expected exactly 1 shape in "{file_path}", found no filled shape '
			f"and {len(borders)} borders (unfilled outlines) - ambiguous."
		)
	elif borders:
		element = borders[0]
		is_border = True
	else:
		raise ValueError(f'No drawable shapes (path/rect/circle/polygon/...) found in "{file_path}".')

	path = Path(element)
	path.reify() # bake transforms into the path's own coordinates

	# svgelements already evaluates curved segments (cubic/quadratic
	# bezier, arcs) parametrically inside Path.point(t), so no manual
	# curve-flattening is needed here - only borders need help, since an
	# open outline has no "boundary" to walk all the way around.
	if is_border:
		_close_path(path)

	return path

def sample_boundary(path: Path, n_samples: int = 4000) -> np.ndarray:
	"""
	Densely sample the path boundary. Returns an (N, 2) array of (x, y).
	"""
	ts = np.linspace(0.0, 1.0, n_samples, endpoint = False)
	pts = []

	log.info(f"Found {len(ts)} points")
	for t in ts:
		p = path.point(t)
		#log.debug(f"[{len(ts)}] X {p.x:.12f} | Y {p.y:.12f}")
		pts.append((p.x, p.y))

	return np.array(pts, dtype = float)

def polygon_centroid(points: np.ndarray) -> np.ndarray:
	"""
	Area-weighted centroid (shoelace formula).

	Falls back to the mean of points if the polygon is degenerate (zero area, e.g. an open path).
	"""
	x = points[:, 0]
	y = points[:, 1]

	x2 = np.roll(x, -1)
	y2 = np.roll(y, -1)

	cross = x * y2 - x2 * y
	area = cross.sum() / 2.0

	if abs(area) < 1e-9:
		return points.mean(axis = 0)

	cx = ((x + x2) * cross).sum() / (6.0 * area)
	cy = ((y + y2) * cross).sum() / (6.0 * area)

	return np.array([cx, cy])

#-=-=-=-#

class ShapeWave:
	"""
	Re-parametrizes a shape boundary by polar angle around its centroid,
	exposing r(theta), y(theta), and the (x, y) boundary point at theta.
	"""
	def __init__(self, file_path: str, n_samples: int = 4000, n_theta: int = 2048):
		self.path = load_first_shape(file_path)
		raw = sample_boundary(self.path, n_samples)
		self.centroid = polygon_centroid(raw)

		dx = raw[:, 0] - self.centroid[0]
		# flip y here because image y grows downward; we want a conventional
		# math-style upward-positive y for the waveform / plot.
		dy = -(raw[:, 1] - self.centroid[1])
		theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

		order = np.argsort(theta)
		theta_sorted = theta[order]
		x_sorted = raw[order, 0]
		y_sorted = -(raw[order, 1] - self.centroid[1]) # centered, flipped y
		r_sorted = np.hypot(dx[order], dy[order])

		# de-duplicate identical angles (can happen at seams) to keep
		# np.interp well-defined
		theta_sorted, uniq_idx = np.unique(theta_sorted, return_index = True)
		x_sorted = x_sorted[uniq_idx]
		y_sorted = y_sorted[uniq_idx]
		r_sorted = r_sorted[uniq_idx]

		# extend arrays periodically so interpolation wraps cleanly at 0 / 2pi
		self._theta_ext = np.concatenate([theta_sorted - 2 * np.pi, theta_sorted, theta_sorted + 2 * np.pi])
		self._x_ext = np.tile(x_sorted, 3)
		self._y_ext = np.tile(y_sorted, 3)
		self._r_ext = np.tile(r_sorted, 3)

		self.theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint = False)
		self.r = np.interp(self.theta_grid, self._theta_ext, self._r_ext)
		self.y = np.interp(self.theta_grid, self._theta_ext, self._y_ext)
		self.x = np.interp(self.theta_grid, self._theta_ext, self._x_ext)

		# normalize y to roughly [-1, 1] for audio / display convenience
		span = max(abs(self.y.max()), abs(self.y.min()), 1e-9)
		self.y_norm = self.y / span

	def point_at(self, theta: float):
		"""
		(x, y) boundary point (centroid-relative, y-up) at angle theta.
		"""
		theta = np.mod(theta, 2 * np.pi)

		x = np.interp(theta, self._theta_ext, self._x_ext)
		y = np.interp(theta, self._theta_ext, self._y_ext)

		return x, y

	def y_at(self, theta):
		theta = np.mod(theta, 2 * np.pi)

		return np.interp(theta, self._theta_ext, self._y_ext)

	def bounds(self):
		"""
		(xmin, xmax, ymin, ymax) of the shape, centroid-relative & y-up.
		"""
		return self._x_ext.min(), self._x_ext.max(), self._y_ext.min(), self._y_ext.max()