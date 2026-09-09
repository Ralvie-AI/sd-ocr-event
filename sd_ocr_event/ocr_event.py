
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

from sd_ocr_event.const import EVENT_SCREENSHOT_FOLDER_USER, EVENT_SCREENSHOT_FOLDER
from sd_ocr_event.utils import crop_black_background, get_image_name_to_utc_dt, get_image
from datetime import datetime
logger = logging.getLogger(__name__)


class ActiveWindowOCRText:
    def __init__(self, server_url: str, event_id: list[int], image_path: list[str], warmup=False) -> None:
        super().__init__()
        self._reader_cache = None
        self.server_url = server_url if server_url else "http://localhost:7600/ocr_event/"
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

    def _send_ocr_result(self, json_output, index):
        """Send OCR results to server with error handling."""
        try:
            payload = {
                'event_id': self.event_id[index],
                'ocr_text': json.dumps(json_output)
            }
            response = requests.post(self.server_url+'/ocr_event_extraction', json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_e:
            logger.error(f"Error during API request: {req_e}")
        except Exception as e:
            logger.error(f"Error sending OCR result: {e}")

    def run_ocr(self, min_conf=0.9, save_box_info=False, save_conf_info=False):

        logger.debug(len(self.event_id))
        for i in range(len(self.event_id)):

            t_init = time.perf_counter()

            img = cv2.imread(self.image_path[i], cv2.IMREAD_COLOR)
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
                self._send_ocr_result(json_output, i)        
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

                self._send_ocr_result(json_output, i)

# if __name__ == "__main__":
#     ocr = ActiveWindowOCRText(
#         server_url="",
#         event_id="955" "964",
#         image_path="" "",
#         warmup=True)
#     ocr.run_ocr()
