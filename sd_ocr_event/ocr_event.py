
import numpy as np
import cv2
import json
import platform
import importlib.util
import gc
import os
import time
import logging
import requests
import shutil

from PIL import Image
from pathlib import Path
from glob import glob

from sd_ocr_event.const import EVENT_SCREENSHOT_FOLDER_USER, EVENT_SCREENSHOT_FOLDER
from sd_ocr_event.utils import crop_black_background

logger = logging.getLogger(__name__)


class ActiveWindowOCRText:
    def __init__(self, server_url, event_id, image_path, warmup=False) -> None:
        super().__init__()
        self._reader_cache = None
        self.server_url = server_url
        self.event_id = event_id
        self.image_path = image_path

        if warmup:
            self._warmup()

    def _warmup(self):
        try:       
            #logging.getlogger("RapidOCR").setLevel(logging.ERROR)

            reader = self.get_cached_reader()
            warmup_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.putText(warmup_img, "Warmup", (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
            _ = reader(warmup_img)
            del warmup_img
            #logger.info("[OCRText] Warmup completed")
        except Exception:
            #logger.exception("[OCRText] Warmup failed")
            raise


    def use_mps(self) -> bool:
        """Detect apple silicon"""
        try:
            if platform.machine() == "arm64":
                #logger.info(f"[OCRText] Detected Apple Silicon")
                return True
            else:
                return False
        except Exception:
            return False

    def has_intel_cpu(self) -> bool:
        """Rough check if CPU is Intel."""
        try:
            cpu_info = (platform.processor() or platform.machine() or "").lower()
            if "intel" in cpu_info:
                #logger.info(f"[OCRText] Detected Intel CPU: {cpu_info}")
                return True
            else:
                return False
        except Exception:
            return False

    def get_cached_reader(self):
        """
        Return a cached RapidOCR reader, chosen based on hardware.
        Apple Silicon -> Torch + MPS
        Intel -> OpenVINO
        Fallback -> ONNX Runtime 
        """
        if self._reader_cache is not None:
            return self._reader_cache

        #logger.info("[OCRText] Initializing RapidOCR reader")

        try:
            from rapidocr import RapidOCR, EngineType, OCRVersion
        except Exception as e:
            #logger.exception(f"[OCRText] Failed to import RapidOCR: {e}")
            raise RuntimeError(f"No suitable RapidOCR backend found. {e}")

        # # --- Apple Silicon (Torch) ---
        if self.use_mps():
            #logger.info("[OCRText] Apple Silicon detected, checking Torch + MPS support")
            if importlib.util.find_spec("torch") is None:
                print('Torch not installed')
                #logger.warning("[OCRText] Torch not installed, cannot use MPS backend")
            else:
                try:
                    import torch
                    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                        self._reader_cache = RapidOCR(params={
                            "Det.engine_type": EngineType.TORCH,
                            "Rec.engine_type": EngineType.TORCH,
                            "Cls.engine_type": EngineType.TORCH,
                            "Global.use_cls": False,
                            "EngineConfig.torch.use_mps": True,
                            "Cls.cls_batch_num": 16,
                            "Rec.rec_batch_num": 16,

                            "Rec.ocr_version": OCRVersion.PPOCRV5 ,
                        })
                        logger.info("[OCR] Backend: TORCH (MPS - Apple Silicon GPU)")
                        return self._reader_cache
                    else:
                        print('Torch MPS backend is NOT available')
                        # logger.warning(
                        #     "[OCRText] Torch installed but MPS backend is NOT available "
                        #     "(likely Intel Mac or unsupported macOS version)"
                        # )
                except Exception as e:
                    print(e)

        # --- Intel-based MacBook from 2006 to 2021 (OpenVINO) ---
        if self.has_intel_cpu():
            if importlib.util.find_spec("openvino") is not None:
                try:
                    self._reader_cache = RapidOCR(params={
                        "Det.engine_type": EngineType.OPENVINO,
                        "Rec.engine_type": EngineType.OPENVINO,
                        "Global.use_cls": False,
                        "Det.device_name": "AUTO",
                        "Cls.device_name": "AUTO",
                        "Rec.device_name": "AUTO",

                        "Rec.ocr_version": OCRVersion.PPOCRV5 ,
                    })
                    logger.info("[OCR] Backend: OPENVINO (Intel CPU)")
                    return self._reader_cache
                except Exception as e:
                    print(e)

        # --- Others (ONNX Runtime) ---
        try:
            self._reader_cache = RapidOCR(params={
                "Global.use_cls": False,
                "Rec.ocr_version": OCRVersion.PPOCRV5 ,
                })
            #logger.info("[OCRText] Loaded Engine: ONNX Runtime")
            # logger.info("[OCR] Backend: ONNX Runtime (CPU)")
            return self._reader_cache
        except Exception as e:
            logger.exception(f"[OCRText] ONNX Runtime backend failed to load: {e}")
            raise RuntimeError("All RapidOCR backends failed to initialize.")

    def _send_ocr_result(self, json_output):
        """Send OCR results to server with error handling."""
        try:
            payload = {
                'screenshot_id': self.screenshot_id,
                'ocr_text': json.dumps(json_output)
            }
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_e:
            logger.error(f"Error during API request: {req_e}")
        except Exception as e:
            logger.error(f"Error sending OCR result: {e}")

    def run_ocr(self, min_conf=0.9, save_box_info=False, save_conf_info=False):

        t_init = time.perf_counter()

        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image")

        reader = self.get_cached_reader()

        try:
            output = reader(img)
        except Exception:
            raise

        if not output:
            logger.info("[OCRText] No text detected")
            json_output = {"data": [{"text": "No text detected"}]} 
            self._send_ocr_result(json_output)        
        else:

            t_ocr_total = time.perf_counter() - t_init
            logger.info(f"[OCRText] run_ocr time: {t_ocr_total:.2f}s")
            #logger.info(f"[TIMING] ocr_execution: {time.perf_counter() - _t_ocr_mode:.3f}s")

            json_output = {
                "data": []
            }

            for box, text, conf in zip(output.boxes, output.txts, output.scores):               
                if conf < min_conf:
                    continue
                json_data = {"text": text}
                # if save_conf_info:
                #     json_data["confidence"] = float(conf)
                # if save_box_info:
                #     json_data["box"] = [[float(p[0]), float(p[1])] for p in box]
                json_output['data'].append(json_data)


            # with open('data.json', 'w', encoding='utf-8') as f:
            #     json.dump(json_output, f, ensure_ascii=False)

            self._send_ocr_result(json_output)


    def get_image_path_and_event_id(self):
            screenshot_folder_user = EVENT_SCREENSHOT_FOLDER_USER.format(user_id=self.user_id)
            filename_list = glob(os.path.join(screenshot_folder_user, "*.png"))
    
            filtered_files = [f for f in filename_list if not f.endswith("_ocr.png")]
    
            filename_list_tmp = sorted(filtered_files, reverse=False)            
          
            if not os.path.isdir(EVENT_SCREENSHOT_FOLDER):
                os.makedirs(EVENT_SCREENSHOT_FOLDER)
             
    
            screenshot_path = self._move_image_file(filename_list_tmp[-1])         

            for tmp_file_data in filename_list:
                os.remove(tmp_file_data)              
           

    def _move_image_file(self, tmp_file):
            # logger.info(f"tmp_file => {tmp_file}")
            full_screen_img = Path(tmp_file).name
            tmp_ocr, ocr_ext = os.path.splitext(full_screen_img)
            ocr_img = tmp_ocr + "_ocr.png"
            screenshot_path = os.path.join(EVENT_SCREENSHOT_FOLDER, full_screen_img)
            screenshot_ocr_path = os.path.join(EVENT_SCREENSHOT_FOLDER, ocr_img)
            
            tmp_ocr_full_path, ocr_tmp_ext = os.path.splitext(tmp_file)
            ocr_tmp_file = tmp_ocr_full_path + "_ocr.png"
    
            shutil.copy2(tmp_file, screenshot_path)
            shutil.copy2(ocr_tmp_file, screenshot_ocr_path)
    
            crop_black_background(screenshot_path, screenshot_path)
    
            if os.path.getsize(screenshot_path) > 1024 * 1024:
                file_size = self.get_readable_file_size(screenshot_path)
                logger.info(f"File size => {file_size}")
                self._aggressive_compress_png(screenshot_path, screenshot_path)
    
            return screenshot_path

    def _aggressive_compress_png(self, input_path, output_path):                   
                
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

# if __name__ == "__main__":
#     ocr = ActiveWindowOCRText(warmup=True)
#     ocr.run_ocr(
#     img_path="/Users/armatura/Desktop/activitywatch/sd-ocr-event/scripts/test.png"
# )
