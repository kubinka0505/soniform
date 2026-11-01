from __future__ import annotations

import numpy as np

from svgelements import (
	SVG,
	Path as SVGPath,
	Shape as SVGShape,
	Move, Close
)

from ..config import logger as log

#-=-=-=-#

class Shape:
	"""A closed 2D point loop, ready to be traced by the oscillator."""

	def __init__(self, xs: np.ndarray, ys: np.ndarray, open_info: dict | None = None):
		# xs/ys are "closed": xs[-1] == xs[0], ys[-1] == ys[0].
		self.xs = xs
		self.ys = ys
		self._open_info = open_info or {
			"is_open": False,
			"subpaths": 1,
			"subpaths_closed": 1,
			"self_intersections": 0,
			"reason": "built directly as a closed loop",
		}

	#-=-=-=-#
	# construction

	@classmethod
	def from_shape(cls, path: str, flatten_samples: int, shape_points: int):
		svg = SVG.parse(path)
		shapes = [e for e in svg.elements() if isinstance(e, (SVGPath, SVGShape))]

		if not shapes:
			raise ValueError(f'No paths/shapes found in "{path}".')

		if len(shapes) > 1:
			log.warning(f"{len(shapes)} sub-paths/shapes found in the shape; using the first one.")

		shape_path = shapes[0]
		if not isinstance(shape_path, SVGPath):
			# convert e.g. circle/rect/polygon to a Path
			shape_path = SVGPath(shape_path)

		open_info = _analyze_open(shape_path, flatten_samples)

		# densely sample the (possibly curved) path parametrically
		ts = np.linspace(0, 1, flatten_samples, endpoint = True)
		raw_pts = [shape_path.point(t) for t in ts] # point objects, not complex numbers
		xs = np.array([p.x for p in raw_pts])
		ys = np.array([p.y for p in raw_pts])

		# svgelements has no isclosed(); check first/last sampled point instead
		if not np.isclose(xs[0], xs[-1]) or not np.isclose(ys[0], ys[-1]):
			log.warning("Selected path does not appear to be closed; it will be force-closed by connecting its first to last point")

			xs = np.append(xs, xs[0])
			ys = np.append(ys, ys[0])

		shape = cls(xs, ys, open_info)
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

	# helpers

	@staticmethod
	def _resolve_point_start(value: str, shape, logger, parser) -> int:
		"""
		Parse starting point as "x, y" coordinates.
		Find and return the index of the closest point in the shape.
		"""
		parts = value.split(",")
		
		try:
			target_x = float(parts[0].strip())
			target_y = float(parts[1].strip()) if len(parts) > 1 else 0
		except (ValueError, IndexError):
			parser.error(f'Invalid starting point format: {value}. Use "x" or "x, y".')

		# Find closest point in shape by distance
		distances = np.sqrt((shape.xs - target_x) ** 2 + (shape.ys - target_y) ** 2)
		closest_idx = int(np.argmin(distances))
		
		min_distance = distances[closest_idx]

		found_x = float(shape.xs[closest_idx])
		found_y = float(shape.ys[closest_idx])
		
		idx0_x = float(shape.xs[0])
		idx0_y = float(shape.ys[0])

		logger.info(
			f"Starting point: target ({target_x:.6f}, {target_y:.6f}) -> "
			f"closest point index {closest_idx} at ({found_x:.6f}, {found_y:.6f}) distance={min_distance:.6f}"
		)
		logger.info(
			f"Shape[0]=({idx0_x:.6f}, {idx0_y:.6f}), found point is +{closest_idx} offset"
		)
		
		# nearby points for debugging
		nearby_indices = [max(0, closest_idx - 2), closest_idx - 1, closest_idx, closest_idx + 1, min(len(shape.xs) - 1, closest_idx + 2)]
		nearby = ", ".join([f"[{i}]=({shape.xs[i]:.3f},{shape.ys[i]:.3f})" for i in nearby_indices if 0 <= i < len(shape.xs)])
		logger.debug(f"Nearby points: {nearby}")

		return closest_idx

	# properties
	@property
	def open(self) -> bool:
		return self._open_info["is_open"]

	@property
	def subpaths(self) -> int:
		return self._open_info["subpaths"]

	@property
	def subpaths_closed(self) -> int:
		return self._open_info["subpaths_closed"]

	@property
	def self_intersections(self) -> int:
		return self._open_info["self_intersections"]

	@property
	def reason(self) -> str:
		return self._open_info["reason"]

	@property
	def info(self) -> dict:
		"""All of the above as one dict."""
		return self._open_info

def _count_crossings(starts: np.ndarray, ends: np.ndarray, ids: np.ndarray) -> int:
	"""
	Count proper crossings between line segments.

	Adjacent segments of the same subpath are skipped.
	"""
	count = 0
	m = len(starts)

	for i in range(m):
		j = np.arange(i + 1, m)
		j = j[~((ids[j] == ids[i]) & (j - i <= 1))]
		if not len(j):
			continue

		p, r = starts[i], ends[i] - starts[i]
		q, s = starts[j], ends[j] - starts[j]
		qp = q - p

		denom = r[0] * s[:, 1] - r[1] * s[:, 0]
		with np.errstate(divide = "ignore", invalid = "ignore"):
			t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
			u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom

		hit = (denom != 0) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
		count += int(hit.sum())

	return count

def _analyze_open(path: SVGPath, samples: int) -> dict:
	"""Decide whether a path is open (encloses no space) and report the evidence."""
	subpaths = list(path.as_subpaths())
	subpaths_closed = 0
	seg_starts, seg_ends, seg_ids = [], [], []

	for k, sub in enumerate(subpaths):
		# drawable segments only; Move/Close carry no geometry to sample
		pieces = [seg for seg in sub if not isinstance(seg, (Move, Close))]

		if not pieces:
			continue

		per_seg = max(2, samples // len(pieces))
		ts = np.linspace(0, 1, per_seg, endpoint = True)
		pts = np.array([[p.x, p.y] for seg in pieces for p in (seg.point(t) for t in ts)])

		# a subpath is closed if it ends with Z or its end meets its start
		has_close = any(isinstance(seg, Close) for seg in sub)
		if has_close or np.allclose(pts[0], pts[-1]):
			subpaths_closed += 1

		seg_starts.append(pts[:-1])
		seg_ends.append(pts[1:])
		seg_ids.append(np.full(len(pts) - 1, k))

	if not seg_starts:
		return {
			"is_open": True,
			"subpaths": len(subpaths),
			"subpaths_closed": 0,
			"self_intersections": 0,
			"reason": "no drawable segments",
		}

	crossings = _count_crossings(
		np.vstack(seg_starts),
		np.vstack(seg_ends),
		np.concatenate(seg_ids)
	)

	is_open = not all((subpaths_closed, crossings))

	if subpaths_closed:
		reason = f"{subpaths_closed} closed subpath(s) enclose a region"
	elif crossings:
		reason = f"{crossings} self-intersection(s) enclose a region"
	else:
		reason = "no closed subpaths and no self-intersections"

	return {
		"is_open": is_open,
		"subpaths": len(subpaths),
		"subpaths_closed": subpaths_closed,
		"self_intersections": crossings,
		"reason": reason,
	}