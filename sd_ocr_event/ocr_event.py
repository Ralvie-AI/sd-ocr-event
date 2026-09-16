import json
import platform
import os
import time
import re
import logging
import shutil
import importlib.util
import urllib3
from glob import glob
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import cv2
import pyopencl as cl
import requests
from PIL import Image

from sd_ocr_event.const import EVENT_SCREENSHOT_FOLDER_USER, EVENT_SCREENSHOT_FOLDER
from sd_ocr_event.utils import crop_black_background

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


logger = logging.getLogger(__name__)

class ActiveEventWindowOCRText:
    def __init__(self, server_url, user_id, image_path,
                 event_id, timestamp, duration,
                 warmup=False) -> None:
        super().__init__()
        self._reader_cache = None
        self.server_url = "http://localhost:7600/screenshot/event/screenshots"        
        self.user_id = user_id        
        self.image_path = image_path
        self.event_id = event_id
        self.timestamp = timestamp
        self.duration = float(duration) if duration != "" else ""
        self.screenshot_id = 0
        self.image_org = ""

        if warmup:
            self._warmup()

    def _warmup(self):
        try:
            reader = self.get_cached_reader()
            warmup_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.putText(warmup_img, "Warmup", (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
            _ = reader(warmup_img)
            del warmup_img
        except Exception:
            logger.exception("[OCRText] Warmup failed")
            raise
            

    def _get_gpu_names(self):
        names = []
        for platform in cl.get_platforms():
            for device in platform.get_devices(device_type=cl.device_type.GPU):
                if device.available:
                    names.append(device.name.lower())
        return names


    def use_directml(self) -> bool:
        """ Detect if system has GPU for DirectML acceleration. 
        - NVIDIA GPUs are allowed. 
        - Intel GPUs are not used for DirectML (prefer CPU). 
        - AMD GPUs are allowed unless they match a deny-list of older/weak architectures."""

        try:    
            active_gpus = []  
            try:
                active_gpus = self._get_gpu_names()
            except Exception:
                pass

            if not active_gpus:
                logger.warning("[OCRText] No GPU name detected")
                return False

            logger.info(f"[OCRText] Detected active GPU: {active_gpus}")

            def has_any(text: str, keywords: list[str]) -> bool:
                return any(k in text for k in keywords)

            def has_any_regex(text: str, patterns: list[str]) -> bool:
                return any(re.search(p, text) for p in patterns)
            
            for gpu_name in active_gpus:
                gpu_name = gpu_name.lower()

                # NVIDIA (always allowed)
                if has_any(gpu_name, ["nvidia", "geforce", "quadro", "rtx", "gtx"]):
                    logger.info(f"[OCRText] Detected NVIDIA - DirectML")
                    return True

                # --- Intel GPUs ---
                if has_any(gpu_name, ["intel arc", "arc a"]):
                    logger.info("[OCRText] Intel Arc detected - DirectML")
                    return True

                if has_any(gpu_name, ["iris xe"]):
                    logger.info("[OCRText] Intel Iris Xe detected")
                    continue

                if has_any(gpu_name, ["intel", "uhd", "hd graphics", "iris"]):
                    logger.info("[OCRText] Intel iGPU detected")
                    continue
        
                # AMD / ATI GPUs
                if has_any(gpu_name, ["amd", "radeon", "ati"]):
                    logger.info(f"[OCRText] Detected AMD")

                    # DENY: Vega integrated graphics (APUs) # Catches "vega 8", "vega8", "vega 8 mobile", etc.
                    vega_integrated_ids = ["3", "6", "7", "8", "10", "11"]
                    for vid in vega_integrated_ids:
                        if re.search(rf"vega\s*{vid}\b", gpu_name):
                            logger.info(f"[OCRText] DENY: Vega integrated graphics (APUs)")
                            continue

                    # DENY: Vega 10 discrete (RX Vega 56/64/Frontier) # But allow Radeon VII (Vega 20).
                    if "radeon vii" not in gpu_name:
                        if (re.search(r"vega\s*56\b", gpu_name) or
                            re.search(r"vega\s*64\b", gpu_name) or
                            "vega frontier" in gpu_name):
                            logger.info(f"[OCRText] DENY: Vega 10 discrete")
                            continue

                    # DENY: any remaining "vega" that isn't Radeon VII
                    if "vega" in gpu_name and "radeon vii" not in gpu_name:
                        logger.info(f"[OCRText] DENY: vega that isn't Radeon VII")
                        continue

                    # DENY: older discrete GCN generations # 3.1 RX 400/500 Polaris
                    polaris_patterns = [
                        r"rx\s*460\b", r"rx\s*470\b", r"rx\s*480\b",
                        r"rx\s*550\b", r"rx\s*560\b", r"rx\s*570\b", r"rx\s*580\b", r"rx\s*590\b",
                    ]
                    if has_any_regex(gpu_name, polaris_patterns):
                        logger.info(f"[OCRText] DENY: older discrete GCN")
                        continue

                    # 3.2 R9 / R7 / R5 
                    r9_patterns = [r"r9\s*295", r"r9\s*290", r"r9\s*280", r"r9\s*270"]
                    r7_patterns = [r"r7\s*370", r"r7\s*360", r"r7\s*350", r"r7\s*340",
                                r"r7\s*260", r"r7\s*250", r"r7\s*240"]
                    r5_patterns = [r"r5\s*340", r"r5\s*330", r"r5\s*240", r"r5\s*230"]

                    if has_any_regex(gpu_name, r9_patterns + r7_patterns + r5_patterns):
                        logger.info(f"[OCRText] DENY: R9 / R7 / R5")
                        continue

                    # DENY: old HD series (HD 4000–7000)
                    if has_any(gpu_name, ["hd 4", "hd 5", "hd 6"]):
                        logger.info(f"[OCRText] DENY: old HD series (HD 4000–7000)")
                        continue

                    # DENY: older Radeon Pro / FirePro workstation cards
                    if has_any(
                        gpu_name,
                        ["radeon pro wx", "radeon pro w5", "radeon pro w4", "firepro w", "firepro s"],
                    ):
                        logger.info(f"[OCRText] DENY: older Radeon Pro / FirePro")
                        continue

                    # DENY: very old APU series
                    if has_any(
                        gpu_name,
                        [" a4-", " a6-", " a8-", " a10-", " a12-",
                        "bristol ridge", "kaveri", "carrizo", "stoney ridge"],
                    ):
                        logger.info(f"[OCRText] DENY: very old APU")
                        continue

                    # DENY: weak generic integrated "Radeon Graphics"
                    if "radeon graphics" in gpu_name:
                        # Allow only if it has RDNA iGPU identifiers (modern APUs)
                        rdna_igpu_markers = [
                            "880m", "870m", "860m",  # RDNA3.5
                            "780m", "760m", "740m",  # RDNA3
                            "680m", "660m", "650m", "630m", "610m",  # RDNA2
                        ]
                        if not has_any(gpu_name, rdna_igpu_markers):
                            logger.info(f"[OCRText] DENY: weak generic integrated Radeon Graphics")
                            continue

                    logger.info(f"[OCRText] Detected AMD GPU")
                    return True

            logger.info(f"[OCRText] No compatible GPU found for directML")
            return False

        except Exception:
            logger.exception("[OCRText] use_directml() failed")
            return False

    def has_intel_cpu(self) -> bool:
        """Rough check if CPU is Intel."""
        try:
            cpu_info = (platform.processor() or platform.machine() or "").lower()
            if "intel" in cpu_info:
                logger.info(f"[OCRText] Detected Intel CPU: {cpu_info}")
                return True
            else:
                return False
        except Exception:
            return False

    def get_cached_reader(self):
        """ Return a cached RapidOCR reader, chosen based on hardware. -NVIDIA/AMD GPU->DirectML with ONNX Runtime -Intel->OpenVINO -Others->ONNX Runtime"""
        if self._reader_cache is not None:
            return self._reader_cache
        
        try:
            from rapidocr import EngineType, OCRVersion, RapidOCR
        except Exception as e:
            #logger.exception(f"[OCRText] Failed to import RapidOCR: {e}")
            raise RuntimeError(f"No suitable RapidOCR backend found. {e}")
        
        # # # --- GPU (DirectML: NVIDIA / AMD) ---'
        if self.use_directml():
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"[OCRText] Available providers: {providers}")
            if "DmlExecutionProvider" in providers:
                try:
                    self._reader_cache = RapidOCR(params={"EngineConfig.onnxruntime.use_dml": True,"Global.use_cls": False,
                                                          "Rec.ocr_version": OCRVersion.PPOCRV5,
                                                          })
                    logger.info(f"[OCRText] Loaded Engine: ONNX Runtime DirectML (GPU)")
                    return self._reader_cache
                
                except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError,
                                    TimeoutError) as e:
                    self._reader_cache = RapidOCR(params={"EngineConfig.onnxruntime.use_dml": True,"Global.use_cls": False,})
                    logger.info(f"[OCRText] Loaded Engine: ONNX Runtime DirectML (GPU)")
                    return self._reader_cache                    
                except Exception as e:
                    logger.warning(f"[OCRText] GPU detected, but DirectML failed to load: {e}")
            else:
                logger.warning(f"[OCRText] Cannot find DmlExecutionProvider")

        # #  # --- Intel (OpenVINO) ---
        if self.has_intel_cpu():
            if importlib.util.find_spec("openvino") is not None:
                try:
                    self._reader_cache = RapidOCR(params={
                        "Det.engine_type": EngineType.OPENVINO, "Cls.engine_type": EngineType.OPENVINO, "Rec.engine_type": EngineType.OPENVINO,
                        "Global.use_cls": False,"Det.device_name": "AUTO", "Cls.device_name": "AUTO","Rec.device_name": "AUTO",
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        })
                    logger.info("[OCRText] Loaded Engine: OpenVINO (Intel CPU)")
                    return self._reader_cache
                except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError,
                                    TimeoutError) as e:
                    self._reader_cache = RapidOCR(params={
                        "Det.engine_type": EngineType.OPENVINO, "Cls.engine_type": EngineType.OPENVINO, "Rec.engine_type": EngineType.OPENVINO,
                        "Global.use_cls": False,"Det.device_name": "AUTO", "Cls.device_name": "AUTO","Rec.device_name": "AUTO",                       
                    })
                    logger.info("[OCRText] Loaded Engine: OpenVINO (Intel CPU)")
                    return self._reader_cache
                
                except Exception as e:
                    logger.warning(f"[OCRText] Intel CPU detected, but OpenVINO failed to load: {e}")        

        try:
            self._reader_cache = RapidOCR(params={"Global.use_cls": False,
                                                  "Rec.ocr_version": OCRVersion.PPOCRV5,
                                                  })
            logger.info("[OCRText] Loaded Engine: ONNX Runtime")
            return self._reader_cache
        except (requests.exceptions.RequestException,
                urllib3.exceptions.HTTPError,
                TimeoutError) as e:
            self._reader_cache = RapidOCR(params={"Global.use_cls": False,})
            logger.info("[OCRText] Loaded Engine: ONNX Runtime")
            return self._reader_cache
        except Exception as e:
            logger.exception(f"[OCRText] ONNX Runtime backend failed to load: {e}")
            raise RuntimeError(f"No suitable RapidOCR backend found. {e}")            

    def _send_ocr_result(self, json_output):
        """Send OCR results to server with error handling."""
        try:
            payload = {
                'screenshot_id': self.screenshot_id,
                'ocr_text': json.dumps(json_output)
            }
            server_url = "http://localhost:7600/screenshot/update_ocr_text"
            response = requests.post(server_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_e:
            logger.error(f"Error during API request: {req_e}")
        except Exception as e:
            logger.error(f"Error sending OCR result: {e}")
        
    def run_ocr(self, min_conf=0.9, save_box_info=False, save_conf_info=False):

        # Main OCR execution function
        t_init = time.perf_counter()

        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image")

        reader = self.get_cached_reader() # get RapidOCR reader
        output = None
        try:
            output = reader(img)            
        except Exception:
            logger.exception("[OCRText] reader(img) failed during fullscreen_ocr")
            raise

        if not output:
            logger.info("[OCRText] No text detected")
            json_output = {"data": [{"text": "No text detected"}]} 
            self._send_ocr_result(json_output)        
        else:

            t_ocr_total = time.perf_counter() - t_init
            logger.info(f"[OCRText] run_ocr time: {t_ocr_total:.2f}s")

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

        filenames = [self.image_org, self.image_path]
        for filename in filenames:                
            try:
                os.remove(filename)
            except OSError as e:
                logger.error("Failed to remove %s : %s", filename, e)
        

    def run_ocr_test(self, min_conf=0.9, save_box_info=False, save_conf_info=False):
    
            # Main OCR execution function
            t_init = time.perf_counter()
    
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to load image")
    
            reader = self.get_cached_reader() # get RapidOCR reader
            output = None
            try:
                output = reader(img)            
            except Exception:
                logger.exception("[OCRText] reader(img) failed during fullscreen_ocr")
                raise
    
            if not output:
                logger.info("[OCRText] No text detected")
                json_output = {"data": [{"text": "No text detected"}]} 
                self._send_ocr_result(json_output)        
            else:
    
                t_ocr_total = time.perf_counter() - t_init
                logger.info(f"[OCRText] run_ocr time: {t_ocr_total:.2f}s")
    
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
    
    def get_files_in_range(self, directory, start_time, duration):
        start_time = datetime.fromisoformat(start_time)
        end_time = start_time + timedelta(seconds=duration)

        result = []

        for file_path in Path(directory).glob("*.png"):
            match = re.search(
                r"_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+Z)\.png$",
                file_path.name,
            )

            if not match:
                continue

            timestamp = match.group(1)

            # Convert:
            # 2026-09-11T07-52-56.935429Z
            # to:
            # 2026-09-11T07:52:56.935429+00:00
            timestamp = timestamp.replace("Z", "+00:00")
            timestamp = re.sub(
                r"T(\d{2})-(\d{2})-(\d{2})",
                r"T\1:\2:\3",
                timestamp,
            )

            file_time = datetime.fromisoformat(timestamp)

            if start_time <= file_time <= end_time:
                result.append(str(file_path))

        return result

    def create_event_ocr(self):
        screenshot_folder_user = EVENT_SCREENSHOT_FOLDER_USER.format(user_id=self.user_id)
        filename_list = self.get_files_in_range(screenshot_folder_user, self.timestamp, self.duration)        

        filtered_files = [f for f in filename_list if not f.endswith("_ocr.png")]               
        
        if not os.path.isdir(EVENT_SCREENSHOT_FOLDER):
            os.makedirs(EVENT_SCREENSHOT_FOLDER)

        logger.info(f"filtered_files => {filtered_files}")
        if len(filtered_files) > 0:
            
            screenshot_path, screenshot_ocr_path = self.move_image_file(filtered_files[-1])

            for tmp_file_data in filename_list:
                os.remove(tmp_file_data)

            response = None
            try:
                capture_time =  datetime.now(timezone.utc)

                payload = {
                    'file_location': screenshot_ocr_path,    
                    'created_at': capture_time.isoformat(),
                    'event_id': self.event_id,
                    'is_ocr_text_enabled': True,
                    'is_event_screenshot': True
                }
                logger.info(f"payload => {payload}")
                logger.info(f"server url => {self.server_url}")
                response = requests.post(self.server_url, json=payload)
                response.raise_for_status() # Raise an exception for bad status codes

            except requests.exceptions.RequestException as req_e:
                logger.error(f"Error during API request: {req_e}")
            except Exception as e:
                logger.error(f"Error in scheduled job: {e}")

            self.image_path = screenshot_ocr_path
            self.image_org = screenshot_path
            self.screenshot_id = response.json()['screenshot_id']
            self.run_ocr()
        else:
            logger.info("There is no file found.")

    def move_image_file(self, tmp_file):

        full_screen_img = Path(tmp_file).name
        tmp_ocr, ocr_ext = os.path.splitext(full_screen_img)
        ocr_img = tmp_ocr + "_ocr.png"
        screenshot_path = os.path.join(EVENT_SCREENSHOT_FOLDER, full_screen_img)
        screenshot_ocr_path = os.path.join(EVENT_SCREENSHOT_FOLDER, ocr_img)
        
        tmp_ocr_full_path, ocr_tmp_ext = os.path.splitext(tmp_file)
        ocr_tmp_file = tmp_ocr_full_path + "_ocr.png"

        shutil.copy2(tmp_file, screenshot_path)
        shutil.copy2(ocr_tmp_file, screenshot_ocr_path)

        # crop_black_background(screenshot_path, screenshot_path)

        # if os.path.getsize(screenshot_path) > 1024 * 1024:
        #     file_size = self.get_readable_file_size(screenshot_path)
        #     logger.info(f"File size => {file_size}")
        #     self.aggressive_compress_png(screenshot_path, screenshot_path)

        return screenshot_path, screenshot_ocr_path

    def aggressive_compress_png(self, input_path, output_path):                   
                
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


if __name__ == "__main__":
    server_url = ""
    user_id = "840c895a938ec8cc9f455b89eba91465"
    image_path = "no-text.png"

    # ba91465_2026-09-15T07-46-54.255866Z_ocr.png
    # ActiveEventWindowOCRText(server_url, screenshot_id, image_path, warmup=True).run_ocr()
    t = ActiveEventWindowOCRText(server_url, user_id, image_path, warmup=True)
    t.create_event_ocr("2026-09-15 07:50:08.702000+00:00", 81.758)
    # ocr = ActiveWindowOCRText(server_url, screenshot_id, warmup=True)
    # # ocr.run_ocr(img_path=r"C:\Users\User\Pictures\ss_test.PNG")    
    # result = ocr.run_ocr(img_path=r"ch.png")    

    # print(result)
    