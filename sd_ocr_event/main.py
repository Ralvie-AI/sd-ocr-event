import argparse
import logging 

from sd_ocr_event.utils import setup_logging
from sd_ocr_event.ocr_event import ActiveWindowOCRText

logger = logging.getLogger(__name__)
    
def main():

    parser = argparse.ArgumentParser(description="Imate to Text")
    parser.add_argument("--server_url", required=True, help="URL to update ocr text")
    parser.add_argument("--image_path", required=True, help="User ID for identification")
    parser.add_argument("--screenshot_id", type=int, default=0, help="Screenshot ID")

    args = parser.parse_args()

    # Set up logging
    setup_logging("sd-ocr-event", log_file=True)

    ActiveWindowOCRText(
        server_url=args.server_url,
        image_path=args.image_path,
        screenshot_id=args.screenshot_id
    ).run_ocr()    


if __name__ == '__main__':
    main()