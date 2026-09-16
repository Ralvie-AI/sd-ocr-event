import argparse
import logging 

from sd_ocr_event.utils import setup_logging
from sd_ocr_event.ocr_event import ActiveEventWindowOCRText

logger = logging.getLogger(__name__)
    
def main():

    parser = argparse.ArgumentParser(description="Imate to Text")
    parser.add_argument("--server_url", required=False, help="URL to update ocr text")    
    parser.add_argument("--user_id", required=str, help="User Id")    
    parser.add_argument("--image_path", type=str, default="", help="Image path")
    parser.add_argument("--event_id", type=int, default=0, help="Event ID")    
    parser.add_argument("--timestamp", type=str, default="", help="Time Stamp")
    parser.add_argument("--duration", type=str, default="", help="Duration")

    args = parser.parse_args()

    # Set up logging
    setup_logging("sd-ocr-event", log_file=True)

    ActiveEventWindowOCRText(
        server_url=args.server_url,        
        user_id=args.user_id,
        image_path= args.image_path,
        event_id=args.event_id,
        timestamp=args.timestamp,
        duration= args.duration                
    ).create_event_ocr()

if __name__ == '__main__':
    main()
