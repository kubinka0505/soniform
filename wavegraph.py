#!/usr/bin/env python3
"""
Turns an SVG shape into a wave rolling shape animated image and a matching
looping sound tone, in the style of circle/square/hexagon -> wave diagrams.

Only the first drawable shape in the graphic is used
(path, rect, circle, ellipse, polygon, polyline, or line).

The shape should be a simple, roughly star-shaped closed outline
(i.e. every boundary point is visible from the centroid)

This covers circles, squares, regular/irregular polygons, blobs, stars, etc.

Highly concave or self-intersecting shapes may produce a jumbled waveform
since the angle-to-radius mapping stops being one-to-one.
"""
import os
import argparse

from src._global.helpers.notes import NoteParser
from src._global.helpers.string import build_format_map

from src.wavegraph.config import logger as log

#-=-=-=-#

__title__   = os.path.splitext(os.path.basename(__file__))[0]
__author__  = "kubinka0505"
__credits__ = __author__
__date__    = "27th June 2026"

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
		help = "Path to the input vector graphics file",
		required = True
	)

	#-=-=-=-#
	# Optional

	optional.add_argument(
		"-oi", "--path-output-image",
		type = str,
		metavar = str,
		default = os.path.normpath("outputs/{prog_name}/{file_name}/{file_name}_{samples}.gif"),
		help = "Output image path."
	)

	optional.add_argument(
		"-os", "--path-output-sound",
		type = str,
		metavar = str,
		default = os.path.normpath("outputs/{prog_name}/{file_name}/{file_name}_{sample_rate}_{frequency}.wav"),
		help = "Output sound path."
	)

	optional.add_argument(
		"-c", "--color",
		type = str,
		metavar = str,
		default = None,
		help = "Hex color for the plotted curve/shape. Random if None."
	)

	optional.add_argument(
		"-f", "--frames",
		type = _parse_number,
		metavar = int,
		default = 300,
		help = "Number of animation frames"
	)

	optional.add_argument(
		"-fps", "--frame-rate",
		type = int,
		metavar = int,
		default = 50,
		help = "Image frame rate"
	)

	optional.add_argument(
		"-s", "--samples",
		type = _parse_number,
		metavar = int,
		default = 1500,
		help = "Boundary sample density"
	)

	optional.add_argument(
		"-t", "--theta-resolution",
		type = _parse_number,
		metavar = int,
		default = 2048,
		help = "Angular resolution of the waveform"
	)

	optional.add_argument(
		"-dpi", "--dpi",
		type = _parse_number,
		metavar = int,
		default = 220,
		help = "Output image DPI - higher means better resolution"
	)

	optional.add_argument(
		"-freq", "--frequency",
		type = _parse_number,
		metavar = float,
		default = NoteParser.decode("C4"),
		help = "Sound tone frequency in Hz"
	)

	optional.add_argument(
		"-p", "--phase",
		type = _parse_number,
		metavar = float,
		default = 0,
		help = "Graph tracing start phase, from 0 to 360"
	)

	optional.add_argument(
		"-sr", "--sample-rate",
		type = _parse_number,
		metavar = int,
		default = 192E3,
		help = "Sound sample rate"
	)

	#-=-=-=-#
	# Switch

	switch.add_argument(
		"-nc", "--no-crop",
		action = "store_true",
		help = "Do not trim transparency around output image"
	)

	switch.add_argument(
		"-nt", "--no-trace-full-wave",
		action = "store_false",
		help = "Show full wave from first frame instead of progressively drawing it"
	)

	switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Display this message and exit"
	)

	#-=-=-=-#
	# Parse

	args = parser.parse_args()

	args.phase = (args.phase % 360) / 360

	if args.color:
		args.color = "#" + args.color.strip("#")

	from src.wavegraph.core.geometry import ShapeWave
	from src.wavegraph.render.img import render as render_img
	from src.wavegraph.render.snd import render as render_snd

	# Path making
	base = os.path.splitext(os.path.basename(args.path_input))[0]

	values = {
		k: v
		for k, v in vars(args).items()
		if isinstance(v, (str, int, float))
	}

	values.update({
		"prog_name": __title__,
		"file_name": base,

		"frequency": f"{values['frequency']:.3f}".replace(".", ","),
		"sample_rate": int(args.sample_rate),
	})

	aliases = {
		"prog_name": ["prog"],

		"color": ["col", "clr"],

		"frames": ["f"],
		"framerate": ["fps", "frame_rate"],

		"samples": ["s"],

		"theta_resolution": ["theta", "theta_res"],

		"frequency": ["freq"],
		"phase": ["phs"],

		"sample_rate": ["sr", "samplerate"],
	}

	fmt = build_format_map(values, aliases)

	path_img = args.path_output_image.format(**fmt)
	path_snd = args.path_output_sound.format(**fmt)

	if args.path_output_sound and path_img != args.path_output_sound:
		os.makedirs(os.path.dirname(path_img), exist_ok = True)

	if args.path_output_image and path_img != args.path_output_image:
		os.makedirs(os.path.dirname(path_img), exist_ok = True)

	#-=-=-=-#

	log.info(f"Loading shape from {args.path_input}...")
	wave = ShapeWave(
		args.path_input,
		n_samples = int(args.samples * (args.dpi / 10)),
		starting_point = args.phase,
		n_theta = args.theta_resolution
	)
	wave.verify()

	log.info(f"Rendering sound -> {path_snd} ({args.frequency} Hz)")
	path_snd = render_snd(
		wave,
		path_snd,
		frequency = args.frequency,
		sample_rate = args.sample_rate,
	)

	log.info(f"Rendering image -> {path_img} ({args.frames} frames @ {args.frame_rate} FPS)")
	path_img = render_img(
		wave,
		path_img,
		n_frames = args.frames,
		fps = args.frame_rate,
		dpi = args.dpi,
		color = args.color,
		trace_full_wave = args.no_trace_full_wave,
		crop = not args.no_crop
	)

	return path_snd, path_img

if __name__ == "__main__":
	main()