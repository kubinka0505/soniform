import logging

#-=-=-=-#

class Logger:
	def __init__(self, name: str):
		self._logger = logging.getLogger(name)
		self._logger.setLevel(logging.DEBUG)

	def debug(self, msg: str = "", *args, **kwargs):
		self._logger.debug(msg, *args, **kwargs)

	def info(self, msg: str = "", *args, **kwargs):
		self._logger.info(msg, *args, **kwargs)

	def warning(self, msg: str = "", *args, **kwargs):
		self._logger.warning(msg, *args, **kwargs)

	def error(self, msg: str = "", *args, **kwargs):
		self._logger.error(msg, *args, **kwargs)

	def exception(self, msg: str = "", *args, **kwargs):
		self._logger.exception(msg, *args, **kwargs)

logging.basicConfig(
	level = logging.DEBUG,
	format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)