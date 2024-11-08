import time

import grpc
from protos import zhonghe_tracking_pb2, zhonghe_tracking_pb2_grpc

import cv2
import numpy as np


def mat2bytes(image: np.ndarray) -> bytes:
    return np.array(cv2.imencode('.jpg', image)[1]).tobytes()


def stream_video(request_flag,
                 timestamp,
                 image,
                 vision_move_speed,
                 vision_move_to_location,
                 global_x,
                 global_y,
                 global_distance,
                 part_distance,
                 width_target,
                 angle_yaw,
                 width_yaw,
                 channel_n_port):
    with grpc.insecure_channel(channel_n_port) as channel:
        stub = zhonghe_tracking_pb2_grpc.ZhongheTrackingStub(channel)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_bytes = mat2bytes(image)
        requests = zhonghe_tracking_pb2.TrackingRequest(request_flag=request_flag,
                                                        timestamp=timestamp,
                                                        image=img_bytes,
                                                        vision_move_speed=vision_move_speed,
                                                        vision_move_to_location=vision_move_to_location,
                                                        global_x=global_x,
                                                        global_y=global_y,
                                                        global_distance=global_distance,
                                                        part_distance=part_distance,
                                                        width_target=width_target,
                                                        angle_yaw=angle_yaw,
                                                        width_yaw=width_yaw)

        try:
            response = stub.TrackingOperation(requests)
            # print('success!')
        except:
            print('error!')
