from typing import Any
from collections.abc import Hashable

#-=-=-=-#

def build_format_map(
	source: dict[str, Any],
	aliases: dict[str, list[Hashable]],
) -> dict[Hashable, Any]:
	"""
	source:
		{
			"framerate": 60,
			"frequency": "1,234",
		}

	aliases:
		{
			"framerate": ["fps", "framerate"],
			"frequency": ["freq", "frequency"],
		}
	"""
	result = {}

	for key, value in source.items():
		result[key] = value

		for alias in aliases.get(key, ()):
			result[alias] = value

	return result