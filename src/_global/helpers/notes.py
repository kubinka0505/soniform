import re
import math

#-=-=-=-#

class NoteParser:
	"""
	Note <-> frequency converter.

	Accepts
	-------
		C4, Db4, D#4, Bb4, A#4

	Encodes (using flats)
	---------------------
		440 -> A4
		466.16 -> Bb4
	"""
	_NAMES = (
		"C", "Db", "D", "Eb",
		"E", "F", "Gb", "G",
		"Ab", "A", "Bb", "B",
	)

	_SEMITONES = {
		"C": 0,

		"C#": 1,
		"Db": 1,

		"D": 2,

		"D#": 3,
		"Eb": 3,

		"E": 4,

		"F": 5,

		"F#": 6,
		"Gb": 6,

		"G": 7,

		"G#": 8,
		"Ab": 8,

		"A": 9,

		"A#": 10,
		"Bb": 10,

		"B": 11,
	}


	@classmethod
	def decode(cls, note: str) -> float:
		"""
		Convert note name -> frequency.

		Examples
		--------
			A4  -> 440
			Bb4 -> 466.16
			A#4 -> 466.16
		"""
		match = re.fullmatch(
			r"([A-G])([#b]?)(\d+)",
			note.strip(),
		)

		if not match:
			raise ValueError(
				f"Invalid note: {note!r}"
			)

		name, accidental, octave = match.groups()

		key = name + accidental

		if key not in cls._SEMITONES:
			raise ValueError(
				f"Invalid note: {note!r}"
			)

		midi = (
			(int(octave) + 1) * 12
			+ cls._SEMITONES[key]
		)

		return 440.0 * (
			2 ** ((midi - 69) / 12)
		)


	@classmethod
	def encode(cls, frequency: float) -> str:
		"""
		Convert frequency -> nearest flat note.

		Examples
		--------
			440 -> A4
			466.16 -> Bb4
		"""
		if frequency <= 0:
			raise ValueError(
				"Frequency must be positive"
			)

		midi = round(
			69 + 12 * math.log2(
				frequency / 440
			)
		)

		octave = midi // 12 - 1

		return (
			f"{cls._NAMES[midi % 12]}"
			f"{octave}"
		)


	@classmethod
	def midi(cls, note: str) -> int:
		"""
		Convert note name -> MIDI number.
		"""
		match = re.fullmatch(
			r"([A-G])([#b]?)(\d+)",
			note.strip(),
		)

		if not match:
			raise ValueError(
				f"Invalid note: {note!r}"
			)

		name, accidental, octave = match.groups()

		return (
			(int(octave) + 1) * 12
			+ cls._SEMITONES[name + accidental]
		)


	@classmethod
	def from_midi(cls, midi: int) -> str:
		"""
		Convert MIDI number -> flat note name.
		"""
		octave = midi // 12 - 1

		return (
			f"{cls._NAMES[midi % 12]}"
			f"{octave}"
		)