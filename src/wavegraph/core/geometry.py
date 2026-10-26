from __future__ import annotations

import math
import numpy as np
from svgelements import SVG, Path, Shape

from ..config import logger as log

#-=-=-=-#

def _iter_shapes(file: SVG):
	"""
	Yield every drawable shape element contained in a parsed SVG document.

	Parameters
	----------
		file:
			A parsed `svgelements.SVG` document (result of `SVG.parse(...)`)

	Yields
	------
	Shape:
		Each element in the document that is a drawable shape
		(path, rect, circle, polygon, ...), excluding the root/nested
		`SVG` container elements themselves.
	"""
	for element in file.elements():
		if isinstance(element, Shape) and not isinstance(element, SVG):
			yield element

def _has_fill(shape: Shape) -> bool:
	"""
	Determine whether a shape is filled or is a stroke-only outline.

	Parameters
	----------
	shape:
		The shape to inspect.

	Returns
	-------
	bool:
		True if the shape has an actual fill (an enclosed area);
		False if it's a "border" - a stroke-only outline with
		`fill = "none"` (or no fill attribute at all).
	"""
	fill = getattr(shape, "fill", None)
	value = getattr(fill, "value", fill)

	return value is not None

def _close_path(path: Path) -> None:
	"""
	Close an open path in place by bridging its last point to its first.

	Parameters
	----------
	path:
		The path to close, modified in place.
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

	Parameters
	----------
	file_path:
		Path to the SVG file to load.

	Returns
	-------
	Path:
		The single resolved shape, with transforms baked into its
		own coordinates and closed if it originated from a border outline.

	Raises
	------
	ValueError:
		If the file is empty, contains no drawable shapes,
		or contains more than one candidate shape/border.
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

	if is_border:
		_close_path(path)

	return path

def sample_boundary(path: Path, n_samples: int = 4000) -> np.ndarray:
	"""
	Densely sample the path boundary at evenly-spaced parametric steps.

	Parameters
	----------
	path:
		The (closed) path to sample.
	n_samples:
		Number of points to sample along the path's parametric range `[0, 1]`.
		Higher values produce a smoother boundary approximation at the cost of more computation.

	Returns
	-------
	np.ndarray:
		An `(n_samples, 2)` array of `(x, y)` coordinates, in the
		path's native (image) coordinate space, where y grows downward.
	"""
	ts = np.linspace(0, 1, n_samples, endpoint = False)
	pts = []

	log.info(f"Found {len(ts)} points")
	for t in ts:
		p = path.point(t)
		#log.debug(f"[{len(ts)}] X {p.x:.12f} | Y {p.y:.12f}")
		pts.append((p.x, p.y))

	return np.array(pts, dtype = float)

def polygon_centroid(points: np.ndarray) -> np.ndarray:
	"""
	Compute the area-weighted centroid of a closed polygon (shoelace formula).

	Parameters
	----------
		points: An `(N, 2)` array of `(x, y)` boundary points, in order
			around the polygon.

	Returns
	-------
		np.ndarray: A 2-element array `[cx, cy]` giving the centroid.
			Falls back to the arithmetic mean of `points` if the polygon
			is degenerate (zero enclosed area, e.g. an open path).
	"""
	x = points[:, 0]
	y = points[:, 1]

	x2 = np.roll(x, -1)
	y2 = np.roll(y, -1)

	cross = x * y2 - x2 * y
	area = cross.sum() / 2

	if abs(area) < 1e-9:
		return points.mean(axis = 0)

	cx = ((x + x2) * cross).sum() / (6 * area)
	cy = ((y + y2) * cross).sum() / (6 * area)

	return np.array([cx, cy])

#-=-=-=-#

class ShapeWave:
	def __init__(self, file_path: str, n_samples: int = 4000, starting_point: float = 0, n_theta: int = 2048):
		"""
		Load a shape and build its angle-parametrized boundary representation.

		Parameters
		----------
			file_path: Path to the SVG file containing the shape.

			n_samples: Number of raw boundary points to sample from the
				path before converting to polar form (see `sample_boundary`).
				Higher values give a more accurate boundary at the cost
				of more computation.

			starting_point: Fraction, in `[0, 1)`, of the way around
				the shape's boundary sampling where the generated cycle
				begins. `0` starts at angle `0` (the positive x-axis,
				before rotation); any other value shifts where in the
				shape's boundary the resulting waveform/frame sequence
				"starts", without changing the shape's geometry. `0.25`
				starts a quarter of the way around, `0.5` starts
				halfway around, and so on. Useful for aligning the
				start of the generated audio/animation to a particular
				point on the boundary.

			n_theta: Number of points in the uniform angular grid
				(`theta_grid`), i.e. the resolution of the resulting
				waveform/animation in one full revolution.
		"""
		self.file_path = file_path
		self.path = load_first_shape(file_path)
		raw = sample_boundary(self.path, n_samples)
		self.centroid = polygon_centroid(raw)

		dx = raw[:, 0] - self.centroid[0]

		dy = -(raw[:, 1] - self.centroid[1])

		theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

		self._boundary_theta_walk = theta

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

		self.starting_point = starting_point
		self._phase_rad = np.mod(starting_point, 1) * 2 * np.pi

		self.theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint = False)
		sample_theta = self.theta_grid + self._phase_rad

		self.r = np.interp(sample_theta, self._theta_ext, self._r_ext)
		self.y = np.interp(sample_theta, self._theta_ext, self._y_ext)
		self.x = np.interp(sample_theta, self._theta_ext, self._x_ext)

		# normalize y to roughly [-1, 1] for audio / display convenience
		span = max(abs(self.y.max()), abs(self.y.min()), 1e-9)
		self.y_norm = self.y / span

	def verify(self, tolerance: float = 1e-2) -> None:
		"""
		Check that the shape is star-shaped with respect to its
		centroid, i.e. that the angle-based re-parametrization this
		class relies on (`theta -> r/x/y`) is actually well-defined.

		Parameters
		----------
		tolerance:
			Maximum allowed backward angular step, in
			radians, between consecutive raw boundary samples
			before it's treated as a genuine direction reversal
			rather than sampling noise.

			Larger values are more forgiving of small concave
			wiggles; smaller values catch subtler problems but may
			also flag noisy sampling of an otherwise-fine shape.

		Raises
		------
		ValueError:
			If the boundary is not star-shaped with respect to its centroid.
		"""
		theta = self._boundary_theta_walk

		diffs = np.diff(theta)
		diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

		# also check the step that closes the loop (last sample back to first)
		closing = (theta[0] - theta[-1] + np.pi) % (2 * np.pi) - np.pi
		diffs = np.concatenate([diffs, [closing]])

		direction = np.sign(np.median(diffs))
		if direction == 0:
			direction = 1

		winding = diffs.sum() / (2 * np.pi)
		reversed_at = np.where(diffs * direction < -tolerance)[0]

		problems = []

		if reversed_at.size:
			frac = reversed_at[0] / len(theta)
			problems.append(
				f"the angle around the centroid reverses direction at "
				f"roughly {frac:%} of the way around the boundary"
			)

		if abs(abs(winding) - 1) > 0.5:
			problems.append(
				f"the boundary winds around the centroid {winding:.2f} "
				"times (expected exactly approx. 1)"
			)

		if problems:
			raise ValueError(
				"Shape is not correctly shaped with respect "
				"to its centroid, so its angle-based trace would be "
				f"unreliable or chaotic: {'; '.join(problems)}."
			)

	def point_at_progress(self, progress: float):
		"""
		Boundary point at a position along the generated cycle, rather
		than an absolute physical angle.

		Parameters
		----------
		progress:
			Position in `[0, 2*pi)` along the generated
			cycle (any real value; wrapped automatically).

		Returns
		-------
		tuple[float, float]:
			The `(x, y)` boundary point,
			centroid-relative and y-up.
		"""
		return self.point_at(progress + self._phase_rad)

	def y_at_progress(self, progress: float):
		"""
		Like `point_at_progress`, but returns only the y-coordinate.

		Parameters
		----------
		progress:
			Position in `[0, 2*pi)` along the generated
			cycle (any real value; wrapped automatically).

		Returns
		-------
		float:
			The y-coordinate of the boundary at that progress
			point, accounting for `starting_point`.
		"""
		return self.y_at(progress + self._phase_rad)

	def point_at(self, theta: float):
		"""
		Look up the boundary point at an absolute angle.

		Note this uses the raw (unrotated) boundary sampling, so it is
		unaffected by `starting_point` - `theta` here is an absolute
		angle around the shape's centroid, not a position along
		`theta_grid`.

		Parameters
		----------
		theta:
			Angle in radians (any real value; wrapped into
			`[0, 2*pi)` automatically).

		Returns
		-------
		tuple[float, float]:
			The `(x, y)` boundary point at `theta`,
			centroid-relative and y-up.
		"""
		theta = np.mod(theta, 2 * np.pi)

		x = np.interp(theta, self._theta_ext, self._x_ext)
		y = np.interp(theta, self._theta_ext, self._y_ext)

		return x, y

	def y_at(self, theta):
		"""
		Look up the (centroid-relative, y-up) boundary y-coordinate at an
		absolute angle.

		Like `point_at`, this uses the raw (unrotated) boundary sampling
		and is unaffected by `starting_point`.

		Parameters
		----------
		theta:
			Angle in radians (any real value; wrapped into `[0, 2*pi)` automatically).

		Returns
		-------
		float:
			The y-coordinate of the boundary at `theta`.
		"""
		theta = np.mod(theta, 2 * np.pi)

		return np.interp(theta, self._theta_ext, self._y_ext)

	def bounds(self):
		"""
		Get the bounding box of the shape's boundary samples.

		Returns
		-------
		tuple[float, float, float, float]:
			`(xmin, xmax, ymin, ymax)` of the shape, centroid-relative and y-up.
		"""
		return self._x_ext.min(), self._x_ext.max(), self._y_ext.min(), self._y_ext.max()