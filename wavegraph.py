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

from src._global.setup import Logger
log = Logger("oscigraph")

#-=-=-=-#

__title__   = os.path.splitext(os.path.basename(__file__))[0]
__author__  = "kubinka0505"
__credits__ = __author__
__date__    = "27th June 2026"

#-=-=-=-#

def main(argv: list[str] | None = None):
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
		"-oi", "--output-image",
		type = str,
		metavar = str,
		default = os.path.normpath("outputs/{prog_name}/{file_name}/{file_name}_{samples}.gif"),
		help = "Output image path."
	)

	optional.add_argument(
		"-os", "--output-sound",
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
		type = int,
		metavar = int,
		default = 300,
		help = "Number of animation frames"
	)

	optional.add_argument(
		"-fps", "--framerate",
		type = int,
		metavar = int,
		default = 50,
		help = "Image framerate"
	)

	optional.add_argument(
		"-s", "--samples",
		type = int,
		metavar = int,
		default = 1500,
		help = "Boundary sample density"
	)

	optional.add_argument(
		"-tr", "--theta-resolution",
		type = int,
		metavar = int,
		default = 2048,
		help = "Angular resolution of the waveform"
	)

	optional.add_argument(
		"-dpi", "--dpi",
		type = int,
		metavar = int,
		default = 220,
		help = "Output image DPI - higher means better resolution"
	)

	optional.add_argument(
		"-freq", "--frequency",
		type = float,
		metavar = float,
		default = 440 * (2 ** (3 / 12)) / 4, # C4
		help = "Sound tone frequency in Hz"
	)

	optional.add_argument(
		"-sr", "--sample-rate",
		type = int,
		metavar = int,
		default = 192E3,
		help = "Sound sample rate"
	)

	#-=-=-=-#
	# Switch

	switch.add_argument(
		"-ncrp", "--no-crop",
		action = "store_true",
		help = "Do not trim transparency around output image"
	)

	switch.add_argument(
		"-trc", "--trace-full-wave",
		action = "store_true",
		help = "Progressively draw the wave instead of showing it fully from frame 1"
	)

	switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Display this message and exit"
	)

	#-=-=-=-#
	# Parse

	args = parser.parse_args()

	from src.wavegraph.core.geometry import ShapeWave
	from src.wavegraph.render.img import render as render_img
	from src.wavegraph.render.snd import render as render_snd

	# Path making

	base = os.path.splitext(os.path.basename(args.path_input))[0]
	img_path = args.output_image.format(
		prog_name = __title__,
		file_name = base,
		samples = args.samples
	) or f"{base}.gif"

	snd_path = args.output_sound.format(
		prog_name = __title__,
		file_name = base,
		sample_rate = int(args.sample_rate),
		frequency = str(round(args.frequency, 3)).replace(".", ",")
	) or f"{base}.wav"

	if args.output_sound and img_path != args.output_sound:
		os.makedirs(os.path.dirname(img_path), exist_ok = True)

	if args.output_image and img_path != args.output_image:
		os.makedirs(os.path.dirname(img_path), exist_ok = True)

	#-=-=-=-#

	log.info(f"Loading shape from {args.path_input}...")
	wave = ShapeWave(
		args.path_input,
		n_samples = int(args.samples * (args.dpi / 10)),
		n_theta = args.theta_resolution
	)

	log.info(f"Rendering sound -> {snd_path} ({args.frequency} Hz)")
	snd_path = render_snd(
		wave,
		snd_path,
		frequency = args.frequency,
		sample_rate = args.sample_rate,
	)

	log.info(f"Rendering image -> {img_path} ({args.frames} frames @ {args.framerate} FPS)")
	img_path = render_img(
		wave,
		img_path,
		n_frames = args.frames,
		fps = args.framerate,
		dpi = args.dpi,
		color = args.color,
		trace_full_wave = args.trace_full_wave,
		crop = not args.no_crop
	)

if __name__ == "__main__":
	os.sys.exit(main())