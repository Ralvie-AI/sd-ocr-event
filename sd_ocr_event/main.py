import argparse
import logging 

from sd_ocr_event.utils import setup_logging
from sd_ocr_event.ocr_event import ActiveWindowOCRText

logger = logging.getLogger(__name__)
    
def main():

    parser = argparse.ArgumentParser(description="Imate to Text")
    parser.add_argument("--server_url", required=True, help="URL to update ocr text")
    parser.add_argument("--image_path", nargs="+", required=True, help="Images for ocr")
    parser.add_argument("--event_id", type=int, nargs="+", default=[0], help="Event IDs")

    args = parser.parse_args()

    # Set up logging
    setup_logging("sd-ocr-event", log_file=True)

    ActiveWindowOCRText(
        server_url=args.server_url,
        image_path=args.image_path,
        event_id=args.event_id
    ).run_ocr()    


if __name__ == '__main__':
    main()