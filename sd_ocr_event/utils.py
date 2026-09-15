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
from sd_ocr_event.const import EVENT_SCREENSHOT_FOLDER_USER, EVENT_SCREENSHOT_FOLDER
from pathlib import Path
from glob import glob
import shutil
from sd_core.const import STAGING
from sd_core.log import setup_logging

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
        logger.debug("No significant black background — skipping crop.")
        return

    logger.debug(f"Black background detected ({black_ratio:.1%}) — cropping.")

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
        logger.debug(f"Cropped image saved: {output_path}")
        if os.path.exists(image_path):
            os.remove(image_path)

def get_image_name_to_utc_dt(filename: str) -> datetime:
    import os
    import re
    from datetime import datetime, timezone

    filename = os.path.basename(filename)

    #แก้ Regex ตรงนี้: ใส่ (\.\d+)? เพื่อบอกว่า "ทศนิยมวินาที จะมีหรือไม่มีก็ได้"
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(\.\d+)?Z", filename)

    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    ts_part = match.group(0)

    #เพิ่มการตรวจสอบ: ถ้าชื่อไฟล์ไม่มีจุดทศนิยม ให้แกะฟอร์แมตแบบไม่มี .%f
    if "." not in ts_part:
        return datetime.strptime(
            ts_part,
            "%Y-%m-%dT%H-%M-%SZ"
        ).replace(tzinfo=timezone.utc)
    else:
        return datetime.strptime(
            ts_part,
            "%Y-%m-%dT%H-%M-%S.%fZ"
        ).replace(tzinfo=timezone.utc)

def get_image(start_time: datetime, end_time: datetime, user_id: str, event_id: int):

    setup_logging("sd-ocr-event", log_file=True)
    screenshot_folder_user = EVENT_SCREENSHOT_FOLDER_USER.format(user_id=user_id)
    filename_list = glob(os.path.join(screenshot_folder_user, "*.png"))

    if STAGING == 1:
        path_for_debug = Path(EVENT_SCREENSHOT_FOLDER) / 'DEBUG' / f'eventID_{event_id}'
        path_for_debug.mkdir(parents=True, exist_ok=True)

        for file in filename_list:
            if file.endswith('_active.png'):
                file_name = Path(file).name
                shutil.copy(file, os.path.join(path_for_debug, f'{file_name}'))

    logger.debug(f'total filename_list => {len(filename_list)}')
    if filename_list:
        try: 
            filtered_files = [f for f in filename_list if not f.endswith("_active.png")]
            logger.debug(f'[filename_list: {len(filtered_files)}] - {filtered_files}')
            image_time_list = sorted(
                    image_time
                    for image_time in filtered_files
                    if start_time <= get_image_name_to_utc_dt(image_time) <= end_time
            )
            logger.debug(f'[image_time_list: {len(image_time_list)}] - {image_time_list}')
            if image_time_list:
                screenshot_path = _move_image_file(image_time_list[-1])
                screenshot_time = get_image_name_to_utc_dt(screenshot_path)
                logger.debug(f'[SCREENSHOT_PATH]: {screenshot_path}')
                logger.debug(f'[SCREENSHOT_TIME]: {screenshot_time}')

                for tmp_file_data in filename_list:
                    os.remove(tmp_file_data)  

                return screenshot_path, screenshot_time
            else:
                for tmp_file_data in filename_list:
                    os.remove(tmp_file_data)  
                logger.debug(f'image_time_list NOT FOUND')
                return None, None
            
        except Exception as e:
            for tmp_file_data in filename_list:
                os.remove(tmp_file_data) 
            logger.debug(f"[OCRText] {e}")
            raise RuntimeError(e)
    else:
        logger.debug(f'filename_list NOT FOUND')
        return None, None

def _get_readable_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
        
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def _move_image_file(tmp_file):
        # logger.info(f"tmp_file => {tmp_file}")
        full_screen_img = Path(tmp_file).name
        tmp_ocr, ocr_ext = os.path.splitext(full_screen_img)
        ocr_img = tmp_ocr + "_active.png"
        # screenshot_path = os.path.join(EVENT_SCREENSHOT_FOLDER, full_screen_img)
        screenshot_ocr_path = os.path.join(EVENT_SCREENSHOT_FOLDER, ocr_img)
            
        tmp_ocr_full_path, ocr_tmp_ext = os.path.splitext(tmp_file)
        ocr_tmp_file = tmp_ocr_full_path + "_active.png"

        try:
        #shutil.copy2(tmp_file, screenshot_path)
            shutil.copy2(ocr_tmp_file, screenshot_ocr_path)
        except Exception as e:
            logger.debug(f'ocr_tmp_file => {ocr_tmp_file}')
            logger.debug(f'screenshot_ocr_path => {screenshot_ocr_path}')
            logger.exception(e)
    
        #crop_black_background(screenshot_path, screenshot_path)
    
        # if os.path.getsize(screenshot_path) > 1024 * 1024:
        #     file_size = _get_readable_file_size(screenshot_path)
        #     logger.info(f"File size => {file_size}")
        #     _aggressive_compress_png(screenshot_path, screenshot_path)
    
        return screenshot_ocr_path

def _aggressive_compress_png(input_path, output_path):                   
                
        with Image.open(input_path) as img:
            # 1. Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
                    
            # 2. Resize the image (PNGs at 4K or 1080p are rarely under 500kb)
            # We will scale it down to a max width of 1280px to save space
            width, height = img.size
            if width > 1280:
                ratio = 1280 / width
                new_size = (1280, int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized to {new_size[0]}x{new_size[1]}")
    
            # 3. Apply Quantization (The most important step for PNG size)
            # We reduce the image to a 256-color palette         
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    
            os.remove(input_path)
                
            # 4. Save with optimization
            img.save(output_path, "PNG", optimize=True)