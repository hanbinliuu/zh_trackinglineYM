# -- coding: utf-8 --x
import os.path
import time
from datetime import datetime

import cv2
from loguru import logger

from Camera.camera import CameraWrapper
from TrackingAlgo2 import TrackingAlgo

if __name__ == "__main__":
    # 起点需要手动设定
    TrackingAlgo = TrackingAlgo()

    # 设置日志
    LOG_DIR = os.path.expanduser("./logs")
    current_date = datetime.now().strftime("%Y-%m-%d")
    LOG_FILE = os.path.join(LOG_DIR, f"file_{current_date}.log")
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    logger.add(LOG_FILE, rotation="200MB", retention="10 days", level="DEBUG")

    cam = CameraWrapper()
    cam.enable_device(0)
    count = 0
    st_time = time.time()
    while True:
        try:
            start = time.time()

            logger.info(f"拍照前: {time.time()}")

            image = cam.get_image()

            # 增加拍图后时间log
            logger.info(f"拍图后时间: {time.time()}")

            TrackingAlgo.image = image
            # TrackingAlgo.distance_calc()
            # cv2.imshow('box', TrackingAlgo.image)
            # cv2.waitKey()

            TrackingAlgo.tracking_operation('192.168.6.181:5000')
            # cv2.imshow('box', image)
            # cv2.waitKey(100)  # 等待3秒钟

            if TrackingAlgo.global_iteration1 < 2:
                TrackingAlgo.global_iteration1 += 1
            if TrackingAlgo.global_iteration2 < 2:
                TrackingAlgo.global_iteration2 += 1
            end = time.time()
            logger.info(f"每帧处理时间: {end - start}")

        except Exception as e:
            logger.error(e)
            cam.close_device()

    ed_time = time.time()
    print(1 / ((ed_time - st_time) / count))
    # 关闭设备
    cam.close_device()
