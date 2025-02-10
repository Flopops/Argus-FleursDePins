import sys
from io import StringIO
from contextlib import contextmanager
import logging.config


logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "info": {
                "format": "%(levelname)s-%(asctime)s|%(message)s",
                "datefmt": "%Y%m%d-%H:%M:%S",
            }
        },
        "handlers": {
            "info_stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "info",
                "level": logging.INFO,
            },
            "info_rotating_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "info",
                "level": logging.INFO,
                "filename": "info.log",
                "mode": "a",
            },
        },
        "loggers": {
            "": {
                "level": logging.INFO,
                "handlers": [
                    "info_stream_handler",
                    "info_rotating_file_handler",
                ],  # only change here
                "propagate": False,
            }
        },
    }
)
LOGGER = logging.getLogger()
# LOGGER.setLevel(logging.INFO)


@contextmanager
def capture_print():
    old_stdout = sys.stdout  # Save the current stdout
    captured_output = StringIO()  # Create a stringIO object to capture output
    sys.stdout = captured_output  # Redirect stdout to the stringIO object
    try:
        yield captured_output  # Yield control with the captured output
    finally:
        sys.stdout = (
            old_stdout  # Restore original stdout after exiting the block
        )
