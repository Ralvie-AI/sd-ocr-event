import sys 
import re
import os
import subprocess
import logging
import argparse
from functools import wraps
from typing import Callable, Optional
from datetime import datetime, timezone, timedelta, time
from time import sleep as time_sleep
from logging.handlers import RotatingFileHandler
import platformdirs

from typing import Tuple, Optional

from PIL import Image, ImageDraw
import cv2
import numpy as np

# Constants for DWM to get the real window size (minus shadows)
DWMWA_EXTENDED_FRAME_BOUNDS = 9
BLACK_RATIO_THRESHOLD = 0.05  # 5%
BLACK_PIXEL_THRESHOLD = 10
BOX_THICKNESS = 4
BOX_COLOR = (255, 0, 0)  # Red in RGB

GetDirFunc = Callable[[Optional[str]], str]

logger = logging.getLogger(__name__)

def ensure_path_exists(path: str) -> None:
    """
     Ensure path exists if not create it. This is useful for creating directories in case they don't exist before we're going to use them.

     @param path - Path to check for existence. It will be created if it doesn't exist

     @return True if path exists
    """
    # Create a directory if it doesn t exist.
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_path_exists(path: str) -> None:
    """
     Ensure path exists if not create it. This is useful for creating directories in case they don't exist before we're going to use them.

     @param path - Path to check for existence. It will be created if it doesn't exist

     @return True if path exists
    """
    # Create a directory if it doesn t exist.
    if not os.path.exists(path):
        os.makedirs(path)


def _ensure_returned_path_exists(f: GetDirFunc) -> GetDirFunc:
    """
     Decorator to ensure returned path exists. This is useful for functions that need to be wrapped in a get_dir function.

     @param f - function that takes a subpath and returns a path

     @return wrapped function that returns the path that was passed to the function and ensures it exists in the path_
    """
    @wraps(f)
    def wrapper(subpath: Optional[str] = None) -> str:
        """
         Wrapper for : func : ` waflib. Tools. check_path ` that ensures the path exists.

         @param subpath - Path to check for existence. If None path is assumed to be a directory.

         @return Path to the file or directory that was checked for existence. This is a convenience function that wraps the function
        """
        path = f(subpath)
        ensure_path_exists(path)
        return path

    return wrapper


# filename: "0a07029c9a901fe0819abf69dca12c0d_2026-01-14T00-55-52.905552Z.png"
# '2026-01-14 00:55:52.905552'
def get_image_name_to_utc(filename : str) -> str:
    ts_part = re.sub(r"^[^_]+_|\.png$", "", filename)
    dt_utc = datetime.strptime(ts_part, "%Y-%m-%dT%H-%M-%S.%fZ").replace(tzinfo=timezone.utc)

    result = dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")

    return result 


def add_second_to_utc(date_time, seconds):
    # 1. Define your starting timestamp string
    timestamp_str = date_time

    # 2. Parse the string into a datetime object
    # .fromisoformat() handles the timezone (+00:00) automatically
    dt = datetime.fromisoformat(timestamp_str)

    # 3. Add 9.095 seconds using timedelta
    new_dt = dt + timedelta(seconds=seconds)

    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    added_duration_timestamp = new_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return timestamp, added_duration_timestamp


def parse_time(value: str) -> time:
    try:
        hour, minute = map(int, value.split(":"))
        return time(hour, minute)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: '{value}'. Use HH:MM (e.g., 09:00)"
        )

def parse_days(value):
    try:
        return [int(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Days must be comma-separated integers (e.g. 0,1,2,3,4)")
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def stop_process_by_exe(exe_name, time_sleep_time=0.2):
    logger.info(f"killing start cmd_name {exe_name}")
    subprocess.run(f"taskkill /F /IM {exe_name}", shell=True)
    time_sleep(time_sleep_time)  # wait 200ms for process cleanup

def setup_logging(
    name: str,
    testing=False,
    verbose=False,
    log_stderr=True,
    log_file=False,
):  # pragma: no cover
    """
     Setup logging for SD components. This is a wrapper around : func : ` logging. getLogger ` to allow us to set up a logging handler for each SD component.
     
     @param name - The name of the logger. Used for logging messages to the console
     @param testing - Whether or not we are testing
     @param verbose - Whether or not to log to stderr ( debug )
     @param log_stderr - Whether or not to log to stderr
     @param log_file - Whether or not to log to file (
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.handlers = []

    # run with LOG_LEVEL=DEBUG to customize log level across all SD components
    log_level = os.environ.get("LOG_LEVEL")
    # Set the logging level to the current logging level.
    if log_level:
        # Set the logging level as specified in env var
        if hasattr(logging, log_level.upper()):
            root_logger.setLevel(getattr(logging, log_level.upper()))
        else:
            root_logger.warning(
                f"No logging level called {log_level} (as specified in env var)"
            )

    # Add a handler for stderr output.
    if log_stderr:
        root_logger.addHandler(_create_stderr_handler())
    # Add a handler for the file handler.
    if log_file:
        root_logger.addHandler(_create_file_handler(name, testing=testing))

    def excepthook(type_, value, traceback):
        """
         Catch exceptions and log them to root_logger. This is a wrapper around sys. excepthook which logs the exception if log_stderr is set to False.
         
         @param type_ - The type of exception raised. Should be one of : exc : ` sys. exc_info `
         @param value - The value of the exception
         @param traceback
        """
        root_logger.exception("Unhandled exception", exc_info=(type_, value, traceback))
        # call the default excepthook if log_stderr isn't true
        # (otherwise it'll just get duplicated)
        # If log_stderr is set to true then sys. excepthook__ type_ value traceback is logged and the traceback is not logged.
        if not log_stderr:
            sys.__excepthook__(type_, value, traceback)

    sys.excepthook = excepthook


def _create_file_handler(
    name, testing=False, log_json=False
) -> logging.Handler:
    log_dir = get_log_dir(name)

    global log_file_path

    file_ext = ".log.json" if log_json else ".log"

    # Daily log file instead of per-run timestamp
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_name = f"{name}_{'testing_' if testing else ''}{date_str}{file_ext}"

    log_file_path = os.path.join(log_dir, log_name)

    # Append mode + rotation
    fh = RotatingFileHandler(
        log_file_path,
        mode="a",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )

    fh.setFormatter(_create_human_formatter())

    return fh

def _create_stderr_handler() -> logging.Handler:  # pragma: no cover
    """
     Create a handler that writes to stderr. This is useful for debugging and to ensure that stderr is printed to the console in a human readable format.
     
     
     @return A logging. Handler to use for outputting to stderr ( or logging. StreamHandler ). Note that the handler does not have a formatter
    """
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(_create_human_formatter())

    return stderr_handler

def _create_human_formatter() -> logging.Formatter:  # pragma: no cover
    """
     Create a formatter that prints to the console. This is useful for debugging the log messages that don't fit into the console.
     
     
     @return A : class : ` logging. Formatter ` with the same format as the one returned by : func : ` asctime `
    """
    return logging.Formatter(
        "%(asctime)s [%(levelname)-5s]: %(message)s  (%(name)s:%(lineno)s)",
        "%Y-%m-%d %H:%M:%S",
    )


@_ensure_returned_path_exists
def get_log_dir(module_name: Optional[str] = None) -> str:  # pragma: no cover
    """
     Get the path to Sundial's log directory. If module_name is specified it will be appended to the log directory to form a fully qualified path

     @param module_name - name of module to append to the log directory

     @return full path to log directory or None if not found ( in which case we're in an untrusted
    """
    # on Linux/Unix, platformdirs changed to using XDG_STATE_HOME instead of XDG_DATA_HOME for log_dir in v2.6
    # we want to keep using XDG_DATA_HOME for backwards compatibility
    # https://github.com/Sundial/sd-core/pull/122#issuecomment-1768020335
    # Return the path to the log directory for the current user s log files.
    if sys.platform.startswith("linux"):
        log_dir = platformdirs.user_cache_path("Sundial") / "log"
    else:
        log_dir = platformdirs.user_log_dir("Sundial")
    return os.path.join(log_dir, module_name) if module_name else log_dir


# ─────────────────────────────────────────────
#  Black-background crop
# ─────────────────────────────────────────────

def crop_black_background(
    image_path: str,
    output_path: Optional[str] = None,
    threshold: int = BLACK_PIXEL_THRESHOLD,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray <= threshold)
    black_ratio = black_pixels / gray.size

    if black_ratio <= BLACK_RATIO_THRESHOLD:
        logger.info("No significant black background — skipping crop.")
        return

    logger.info(f"Black background detected ({black_ratio:.1%}) — cropping.")

    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) == 0:
        logger.warning("Image is entirely black — skipping.")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[y_min : y_max + 1, x_min : x_max + 1]
    result = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    if output_path:
        result.save(output_path)
        logger.info(f"Cropped image saved: {output_path}")
        if os.path.exists(image_path):
            os.remove(image_path)
