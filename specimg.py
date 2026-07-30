#!/usr/bin/env python3
"""
Encodes an image into the magnitude spectrogram of an audio file.
"""
import os
import argparse

#-=-=-=-#

__title__   = os.path.splitext(os.path.basename(__file__))[0]
__author__  = "kubinka0505"
__credits__ = __author__
__date__    = "29th June 2026"

#-=-=-=-#
# Configuration

def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description = __doc__,
		formatter_class = argparse.ArgumentDefaultsHelpFormatter,
		add_help = False
	)

	#-=-=-=-#
	# Required
	required = parser.add_argument_group("Required arguments")

	required.add_argument(
		"-i", "--path-input",
		type = str,
		metavar = str,
		required = True,
		help = "Input image. Supports SVG and formats supported by PIL"
	)

	#-=-=-=-#
	# Processing
	processing = parser.add_argument_group("Processing arguments")

	processing.add_argument(
		"-fmin", "--frequency-min",
		type = float,
		metavar = float,
		default = 15000,
		help = "Bottom frequency the shape is mapped onto spectrogram"
	)

	processing.add_argument(
		"-fmax", "--frequency-max",
		type = float,
		metavar = float,
		default = 16000,
		help = "Up frequency the shape is mapped onto spectrogram. -1 stands for `sample_rate`"
	)

	# delete?
	processing.add_argument(
		"-peak", "--target-peak-db",
		type = float,
		metavar = float,
		default = 0
	)

	processing.add_argument(
		"-ml", "--min-level-db",
		type = float,
		metavar = float,
		default = None,
		help = "dB floor for silent/out-of-band content. Default is true digital silence (no hiss)."
	)

	processing.add_argument(
		"-g", "--griffin-lim",
		type = int,
		metavar = int,
		default = 0,
		dest = "griffin_lim_iters",
		help = "Number of Griffin-Lim refinement iterations"
	)

	processing.add_argument(
		"-fm", "--frequency-mode",
		type = str,
		choices = ["linear", "log", "exp"],
		default = "linear",
		help = "Frequency mapping mode"
	)

	processing.add_argument(
		"-pm", "--phase-mode",
		type = str,
		default = "random",
		choices = ["random", "zero"],
		help = "Phase reconstruction mode"
	)

	#-=-=-=-#
	# Optional
	optional = parser.add_argument_group("Optional arguments")

	optional.add_argument(
		"-o", "--path-output",
		type = str,
		metavar = str,
		default = os.path.normpath("outputs/{prog_name}/{file_name}/{file_name}_{sample_rate}_{frequency_min}-{frequency_max}.wav"),
		help = "Output sound path"
	)

	optional.add_argument(
		"-sr", "--sample-rate",
		type = int,
		metavar = int,
		default = 44100,
		help = "Sound sample rate"
	)

	optional.add_argument(
		"-d", "--duration",
		type = float,
		metavar = float,
		default = 1,
		help = "Sound duration, in seconds"
	)

	optional.add_argument(
		"-fft", "--n-fft",
		type = int,
		metavar = int,
		default = 2048,
		help = "Controls frequency resolution"
	)

	optional.add_argument(
		"-hl", "--hop-ratio",
		type = float,
		metavar = float,
		default = 0,
		help = "hop_length = n_fft * hop_ratio"
	)

	optional.add_argument(
		"-w", "--window",
		type = str,
		metavar = str,
		default = "hann",
		help = "Window type: hann, hamming, blackman, ..."
	)

	optional.add_argument(
		"-fms", "--fade-ms",
		type = float,
		metavar = float,
		default = 15
	)

	#-=-=-=-#
	# Adjustments
	adjustments = parser.add_argument_group("Image preprocessing arguments")

	adjustments.add_argument(
		"-ig", "--gamma",
		type = float,
		metavar = float,
		default = 1,
		help = "Image gamma"
	)

	adjustments.add_argument(
		"-ic", "--contrast",
		type = float,
		metavar = float,
		default = 1,
		help = "Image contrast"
	)

	adjustments.add_argument(
		"-ib", "--brightness",
		type = float,
		metavar = float,
		default = 1,
		help = "Image brightness"
	)

	#-=-=-=-#
	# Vector
	conversion = parser.add_argument_group("Image conversion arguments")

	conversion.add_argument(
		"-vt", "--threshold",
		type = float,
		metavar = float,
		default = None,
		help="Image binarization level"
	)

	conversion.add_argument(
		"-vs", "--vector-render-scale",
		type = float,
		metavar = float,
		default = 64,
		help = "Supersampling multiplier for image rasterization"
	)

	#-=-=-=-#
	# Switch
	switch = parser.add_argument_group("Switch arguments")

	switch.add_argument(
		"-inv", "--invert",
		action = "store_true",
		help = "Invert colors in spectrogram"
	)

	switch.add_argument(
		"-fx", "--flip-horizontal",
		action = "store_true",
		help = "Flip spectrogram in X axis"
	)

	switch.add_argument(
		"-fy", "--flip-vertical",
		action = "store_true",
		help = "Flip spectrogram in Y axis"
	)

	switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Display this message and exit"
	)

	#-=-=-=-#

	args = parser.parse_args()

	from src.specimg.config import Settings
	from src.specimg.core.snd.export import convert

	if args.frequency_max < 0:
		args.frequency_max = args.sample_rate

	# Path making

	base = os.path.splitext(os.path.basename(args.path_input))[0]
	path_out = args.path_output.format(
		prog_name = __title__,
		file_name = base,
		sample_rate = args.sample_rate,
		frequency_min = str(round(args.frequency_min, 3)).replace(".", ","),
		frequency_max = str(round(args.frequency_max, 3)).replace(".", ",")
	) or f"{base}.gif"

	if args.path_output and path_out != args.path_output:
		os.makedirs(os.path.dirname(path_out), exist_ok = True)

	#-=-=-=-#

	cfg = Settings(
		sample_rate = args.sample_rate,
		duration = args.duration,
		n_fft = args.n_fft,
		hop_ratio = args.hop_ratio,
		window = args.window,
		frequency_min = args.frequency_min,
		frequency_max = args.frequency_max,
		frequency_mode = args.frequency_mode,
		invert = args.invert,
		flip_vertical = args.flip_vertical,
		flip_horizontal = args.flip_horizontal,
		gamma = args.gamma,
		contrast = args.contrast,
		brightness = args.brightness,
		threshold = args.threshold,
		vector_render_scale = args.vector_render_scale,
		min_level_db = args.min_level_db,
		phase_mode = args.phase_mode,
		griffin_lim_iters = args.griffin_lim_iters,
		fade_ms = args.fade_ms
	)

	if not os.path.exists(args.path_input):
		parser.error(f"Input file not found: {args.path_input}")

	convert(args.path_input, path_out, cfg)

if __name__ == "__main__":
	main()