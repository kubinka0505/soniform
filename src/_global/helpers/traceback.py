import os
import traceback
from contextlib import contextmanager

#-=-=-=-#

@contextmanager
def _parser_traceback(parser):
	try:
		yield
	except Exception as err:
		# Traverse to the deepest part of the traceback where the error was raised
		tb = err.__traceback__
		while tb.tb_next:
			tb = tb.tb_next

		lineno = tb.tb_lineno
		filename = tb.tb_frame.f_code.co_filename
		filename = os.path.relpath(filename, os.getcwd())

		# Format the output with line number info
		os.sys.stderr.write(f"\n{parser.prog} | {filename} @ line {lineno}:\n{err.__class__.__name__}: {err}\n\a")
		os.sys.exit(2)