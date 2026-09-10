import os 
import sys 
import logging
import subprocess
from pathlib import Path
from sd_ocr_event.ocr_event import ActiveEventWindowOCRText

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    exe_dir = script_dir = Path(__file__).resolve().parent.parent
    script_dir = Path(__file__).resolve().parent
    img_file = os.path.join(script_dir, "test.png")
    cor_event = ActiveEventWindowOCRText(
        server_url="", 
        event_id="",
        user_id="", 
        image_path=img_file,
    ).run_ocr_test()
    
    