import os 
import sys 
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def start_exe(exec_cmd, timeout_sec=15):
    logger.info(f"Starting module {exec_cmd}")
    if not isinstance(exec_cmd, list):
        exec_cmd = [exec_cmd]

    logger.debug("Running: {}".format(exec_cmd))

    # Don't display a console window on Windows
    # See: https://github.com/ActivityWatch/activitywatch/issues/212
    startupinfo = None
    if sys.platform in ("win32", "cygwin"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        # Use the 'with' statement to ensure underlying handles are cleaned up even if exceptions occur
        with subprocess.Popen(
                exec_cmd,
                universal_newlines=True,
                startupinfo=startupinfo
        ) as proc:
            try:
                # Block and wait, with a timeout mechanism to prevent the process accumulation
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                # If the exe hangs, force kill it to prevent processes from piling up!
                logger.error(f"Task execution timed out ({timeout_sec}s)! Force cleaning up...")
                proc.kill()
                proc.wait()
    except Exception as e:
        logger.error(f"Unexpected error occurred while starting the process: {e}")



if __name__ == "__main__":
    exe_dir = script_dir = Path(__file__).resolve().parent.parent
    script_dir = Path(__file__).resolve().parent
    ocr_exe = os.path.join(exe_dir, "dist", "sd-ocr-event", "sd-ocr-event.exe")
    if os.path.exists(ocr_exe):
        print("ocr_exe", ocr_exe)

    img_file = os.path.join(script_dir, "test.png")

    if os.path.exists(img_file):
        print("img_file", img_file)

    cmd = [ocr_exe, '--server_url', 'http://localhost:7600/screenshot/update_ocr_text', '--image_path',
      img_file, '--screenshot_id', '1']
    start_exe(cmd)