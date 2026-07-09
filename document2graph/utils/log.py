import sys
import coloredlogs
import logging
from pprint import pprint

logger = logging.getLogger()
coloredlogs.install(level='INFO', logger=logger, isatty=True)


class Log:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def log(self, s):
        self.logger.info(s)
        sys.stdout.flush()

    def plog(self, s):
        plog(s)

    def info(self, s):
        self.logger.info(s)
        sys.stdout.flush()

    def debug(self, s):
        pprint(s)
        sys.stdout.flush()

    def error(self, s):
        self.logger.error(s)
        sys.stdout.flush()

    def warning(self, s):
        self.logger.warning(s)
        sys.stdout.flush()


def plog(s):
    pprint(s)
    sys.stdout.flush()
