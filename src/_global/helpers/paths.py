import os
from pathlib import Path

#-=-=-=-#

def _parse_path_input(path: str) -> str:
	path = os.path.expanduser(path)
	path = os.path.expandvars(path)
	path = str(Path(path).resolve())

	return path