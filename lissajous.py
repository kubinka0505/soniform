#!/usr/bin/env python3
"""
Converts a closed vector shape into a stereo sound file that traces the shape
on an X-Y oscilloscope / vectorscope display (left channel = X, right channel = Y).

The output has three phases:
	[attack] -> [duration] -> [release]

	* attack   : traced loop grows from silence up to the full shape, while still cycling at `-frequency` Hz.
	* duration : full shape is traced repeatedly at `-frequency` Hz.
	* release  : mirror of attack, loop shrinks back down to silence.
"""
import os
import argparse

from src.lissajous.config import logger as log

#-=-=-=-#

__title__   = os.path.splitext(os.path.basename(__file__))[0]
__author__  = "kubinka0505"
__credits__ = __author__
__date__    = "28th June 2026"

#-=-=-=-#

def main(argv: list[str] | None = None):
	from src._global.helpers.numbers import _parse_number

	parser = argparse.ArgumentParser(
		description = __doc__,
		formatter_class = argparse.ArgumentDefaultsHelpFormatter,
		add_help = False
	)

	required = parser.add_argument_group("Required arguments")
	optional = parser.add_argument_group("Optional arguments")
	switch   = parser.add_argument_group("Switch arguments")

	#-=-=-=-#
	# Required

	required.add_argument(
		"-i", "--path-input",
		type = str,
		metavar = str,
		required = True,
		help = "Path to the input shape file"
	)

	#-=-=-=-#
	# Optional

	optional.add_argument(
		"-o", "--path-output",
		type = str,
		metavar = str,
		default = os.path.normpath("outputs/{prog_name}/{file_name}/{file_name}_{sample_rate}_{frequency}.wav"),
		help = "Output sound path"
	)

	optional.add_argument(
		"-a", "--attack",
		type = _parse_number,
		metavar = float,
		default = 0.5,
		help = "Seconds for waveform to grow from silence to full shape"
	)

	optional.add_argument(
		"-d", "--duration",
		type = _parse_number,
		metavar = float,
		default = 2,
		help = "Seconds for the full static waveform is generated"
	)

	optional.add_argument(
		"-r", "--release",
		type = _parse_number,
		metavar = float,
		default = 0.5,
		help = "Seconds for the waveform to shrink back down to silence"
	)

	optional.add_argument(
		"-freq", "--frequency",
		type = _parse_number,
		metavar = float,
		default = 440 * (2 ** (3 / 12)) / 4, # C4
		help = "Loop frequency in Hz - how many times per second the shape is traced")

	optional.add_argument(
		"-sr", "--sample-rate",
		type = _parse_number,
		metavar = int,
		default = 192E3,
		help = "Output sample rate in Hz"
	)


	optional.add_argument(
		"-deg", "--degrees",
		type = _parse_number,
		metavar = float,
		default = -45,
		help = "Rotate the shape counter-clockwise by this many degrees before tracing"
	)

	optional.add_argument(
		"-ss", "--starting-point",
		type = _parse_number,
		metavar = int,
		default = 0,
		help = "Index (0 to `points - 1`) into the flattened/resampled shape marking the point tracing begins from"
	)

	optional.add_argument(
		"-s", "--samples",
		dest = "flatten_samples",
		type = _parse_number,
		metavar = int,
		default = 4000,
		help = "Internal vector curve-flattening resolution"
	)

	optional.add_argument(
		"-p", "--points",
		dest = "shape_points",
		type = _parse_number,
		metavar = int,
		default = 2000,
		help = "Number of uniformly arc-length-spaced points representing the shape Also the valid range for starting point"
	)

	optional.add_argument(
		"-m", "--margin",
		type = _parse_number,
		metavar = float,
		default = 0.5,
		help = "Amplitude margin, from 0 to 1 so the shape is scaled to fit within them"
	)

	#-=-=-=-#
	# Switch

	switch.add_argument(
		"-rev", "--reverse",
		action = "store_true",
		help = "Form / collapse the shape from silence instead of its start point"
	)

	switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Display this message and exit"
	)

	#-=-=-=-#

	args = parser.parse_args(argv)

	from src.lissajous.config import Settings
	from src.lissajous.core.geometry import Shape
	from src.lissajous.core.synthesis import synthesize
	from src.lissajous.render import write_sound

	for name in (
		"attack", "duration", "release",
		"frequency", "sample_rate"
	):
		if getattr(args, name) < 0:
			parser.error(f"--{name} must be >= 0")

	# Path making

	base = os.path.splitext(os.path.basename(args.path_input))[0]
	path_out = args.path_output.format(
		prog_name = __title__,
		file_name = base,
		sample_rate = int(args.sample_rate),
		frequency = str(round(args.frequency, 3)).replace(".", ",")
	) or f"{base}.gif"

	if args.path_output and path_out != args.path_output:
		os.makedirs(os.path.dirname(path_out), exist_ok = True)

	#-=-=-=-#

	cfg = Settings(
		path_input = args.path_input,
		path_output = path_out,
		attack = args.attack,
		duration = args.duration,
		release = args.release,
		frequency = args.frequency,
		sample_rate = args.sample_rate,
		reverse = args.reverse,
		degrees = args.degrees,
		starting_point = args.starting_point,
		flatten_samples = args.flatten_samples,
		shape_points = args.shape_points,
		margin = args.margin,
	)

	log.info(f'Loading shape from "{cfg.path_input}"...')

	shape = Shape.from_shape(cfg.path_input, cfg.flatten_samples, cfg.shape_points)
	shape.normalize(margin = cfg.margin)

	if cfg.starting_point:
		shape.roll_start(cfg.starting_point)
	if cfg.degrees:
		shape.rotate(cfg.degrees)
	if cfg.reverse:
		shape.reverse()

	left, right, boundaries = synthesize(shape, cfg)

	write_sound(
		cfg.path_output,
		left, right,
		cfg.sample_rate,
		boundaries
	)

#-=-=-=-#

if __name__ == "__main__":
	main()