import os
from dataclasses import dataclass

from .._global.setup import Logger

from typing import Optional

#-=-=-=-#

@dataclass
class Settings:
	path_input: str
	path_output: str

	attack: float
	duration: float
	release: float
	frequency: float
	sample_rate: int
	reverse: bool

	degrees: float
	starting_point: int

	flatten_samples: int
	shape_points: int
	margin: float

@dataclass(slots = True)
class Positions:
	"""
	Sample-index boundaries (in the final, non-oversampled output)
	of the structural regions of a rendered waveform:
	"""
	pos_start_attack:      int
	pos_end_attack:        int

	pos_start_first_cycle: Optional[int]
	pos_end_first_cycle:   Optional[int]

	pos_start_release:     int
	pos_end_release:       int

#-=-=-=-#

logger = Logger(os.path.basename(os.path.dirname(__file__)))