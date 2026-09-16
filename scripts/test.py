import os 
import sys 
import logging
import subprocess
from pathlib import Path
from sd_ocr_event.ocr_event import ActiveEventWindowOCRText

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
    img_file = os.path.join(script_dir, "test.png")
    cor_event = ActiveEventWindowOCRText(
        server_url="", 
        user_id="", 
        image_path=img_file,
        event_id="",
        timestamp="",
        duration=""
        ).run_ocr_test()

    # cmd_path = r"D\sd-ocr-event.exe"
    # user_id = 00000000000000000
    # cmd = [cmd_path, '--event_id', '4', '--user_id', str(user_id), '--image_path', '']    
    # start_exe(cmd)
    