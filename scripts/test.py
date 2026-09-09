import subprocess
import os 
import sys
import time
import logging 


EXE_NAME = 'sd-ocr-event'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
EXE_PATH = os.path.join(
                    os.path.expanduser("~"),
                    "Desktop", "activitywatch", EXE_NAME, "dist", EXE_NAME, EXE_NAME)


def start_exe(command_list):
    try:
        logger.info(f"Starting {EXE_NAME}...")
        logger.debug("Command: %s", " ".join(command_list))

        proc = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        logger.debug(f"Started pid={proc.pid}")

        time.sleep(2)

        if proc.poll() is not None:
            stdout, stderr = proc.communicate()

            logger.error(
                f"{EXE_NAME} exited immediately. "
                f"returncode={proc.returncode}"
            )

            if stdout:
                logger.error(stdout.decode(errors="ignore"))

            if stderr:
                logger.error(stderr.decode(errors="ignore"))
        else:
            logger.debug(f"{EXE_NAME} is still running")

    except Exception:
        logger.exception(f"Failed to start {EXE_NAME}")


if __name__ == "__main__":

    logger.info(f"exe is file => {os.path.isfile(EXE_PATH)}")

    command_list = [
        EXE_PATH,
        "--server_url", "",
        "--event_id", "955","964",
        "--image_path", "","",
    ]
    logger.info(f"command_list => {str(command_list)}")
    start_exe(command_list)