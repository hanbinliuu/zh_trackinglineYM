import math
import time
from loguru import logger

import grpc
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from protos import zhonghe_tracking_pb2_grpc, zhonghe_tracking_pb2
import math
import yaml
import threading


def remove_outliers(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = 3 * std_dev
    filtered_data = [value for value in data if abs(value - mean) <= threshold]

    return filtered_data


def rotation_warnings(heading_data, last_heading_data):
    """ :param heading_data: 当前方位角 float
        :param last_heading_data: 上一个方位角 float """

    # 计算当前方位角和上一个方位角的差值
    angle_change = heading_data - last_heading_data

    # 将差值限制在 [-180, 180] 范围内
    angle_change %= 360
    if angle_change > 180:
        angle_change -= 360

    # 如果差值超出正负90度范围，则发出警告并返回 True
    if abs(angle_change) > 90:
        logger.warning('小车发生转弯！！！')
        return True

    # 如果差值在正负90度范围内，则表示未发生转弯，返回 False
    return False


# def position_calculate(start_point, heading_data, last_odometer_data, origin_point):
#     """ 计算单步位置信息 """
#
#     global delta_x, delta_y
#     if len(last_odometer_data) == 1:
#         diff_ododata = 0
#     else:
#         diff_ododata = last_odometer_data[-1] - last_odometer_data[-2]
#
#     # 小车前进
#     if 0 <= heading_data < 90 and diff_ododata >= 0:
#         delta_x = -diff_ododata * math.cos(math.radians(heading_data))
#         delta_y = diff_ododata * math.sin(math.radians(heading_data))
#
#     if 90 <= heading_data < 180 and diff_ododata >= 0:
#         delta_x = diff_ododata * math.cos(math.radians(180 - heading_data))
#         delta_y = diff_ododata * math.sin(math.radians(180 - heading_data))
#
#     if 180 <= heading_data < 270 and diff_ododata >= 0:
#         delta_x = diff_ododata * math.cos(math.radians(heading_data - 180))
#         delta_y = -diff_ododata * math.sin(math.radians(heading_data - 180))
#
#     if 270 <= heading_data < 360 and diff_ododata >= 0:
#         delta_x = -diff_ododata * math.cos(math.radians(360 - heading_data))
#         delta_y = -diff_ododata * math.sin(math.radians(360 - heading_data))
#
#     # 小车回退
#     if (0 <= heading_data < 90) and diff_ododata < 0:
#         delta_x = -diff_ododata * math.cos(math.radians(heading_data))
#         delta_y = diff_ododata * math.sin(math.radians(heading_data))
#
#     if 90 <= heading_data < 180 and diff_ododata < 0:
#         delta_x = diff_ododata * math.cos(math.radians(180 - heading_data))
#         delta_y = diff_ododata * math.sin(math.radians(180 - heading_data))
#
#     if 180 <= heading_data < 270 and diff_ododata < 0:
#         delta_x = diff_ododata * math.cos(math.radians(heading_data - 180))
#         delta_y = -diff_ododata * math.sin(math.radians(heading_data - 180))
#
#     if 270 <= heading_data < 360 and diff_ododata < 0:
#         delta_x = -diff_ododata * math.cos(math.radians(360 - heading_data))
#         delta_y = -diff_ododata * math.sin(math.radians(360 - heading_data))
#
#     # 更新位置
#     position = [start_point[0] + delta_x, start_point[1] + delta_y]
#     # 计算distance
#     distance = distance_between_points(position[0], position[1], origin_point[0], origin_point[1])
#
#     print('此时里程计数据:', last_odometer_data[-1], '上一时刻里程计数据:', last_odometer_data[-2], '里程差:',
#           diff_ododata)
#     print('小车起点:', [round(x, 2) for x in start_point], '小车终点:', [round(x, 2) for x in position],
#           'part_distance:', round(distance, 2))
#     return distance, position


def quadrant_judgement(heading_data):
    if 0 < heading_data < 90:
        return 2
    if 90 < heading_data < 180:
        return 1
    if 180 < heading_data < 270:
        return 4
    if 270 < heading_data < 360:
        return 3

    if heading_data == 0 or heading_data == 90 or heading_data == 180 or heading_data == 270:
        return 5


def position_calculate2(start_point, heading, last_heading, last_odometer_list, origin_point):
    """ last_heading: 上一个角度
        heading: 当前角度 """

    global delta_x, delta_y
    if len(last_odometer_list) == 1:
        diff_ododata = 0
    else:
        diff_ododata = last_odometer_list[-1] - last_odometer_list[-2]

    if (quadrant_judgement(last_heading) == 1 or quadrant_judgement(last_heading) == 2 or
            quadrant_judgement(last_heading) == 3 or quadrant_judgement(last_heading) == 4):
        if quadrant_judgement(heading) == 1:
            # x变大， y变大
            delta_x = diff_ododata * math.cos(math.radians(180 - heading))
            delta_y = diff_ododata * math.sin(math.radians(180 - heading))
        if quadrant_judgement(heading) == 2:
            # x变小， y变大
            delta_x = -diff_ododata * math.cos(math.radians(heading))
            delta_y = diff_ododata * math.sin(math.radians(heading))
        if quadrant_judgement(heading) == 3:
            # x变小， y变小
            delta_x = -diff_ododata * math.cos(math.radians(360 - heading))
            delta_y = -diff_ododata * math.sin(math.radians(360 - heading))
        if quadrant_judgement(heading) == 4:
            # x变大， y变小
            delta_x = diff_ododata * math.sin(math.radians(270 - heading))
            delta_y = -diff_ododata * math.cos(math.radians(270 - heading))

        if quadrant_judgement(last_heading) == 5:
            if heading == 0:
                delta_x = -diff_ododata
            if heading == 90:
                delta_y = diff_ododata
            if heading == 180:
                delta_x = diff_ododata
            if heading == 270:
                delta_y = -diff_ododata
            else:
                if quadrant_judgement(heading) == 1:
                    # x变大， y变大
                    delta_x = diff_ododata * math.cos(math.radians(180 - heading))
                    delta_y = diff_ododata * math.sin(math.radians(180 - heading))
                if quadrant_judgement(heading) == 2:
                    # x变小， y变大
                    delta_x = -diff_ododata * math.cos(math.radians(heading))
                    delta_y = diff_ododata * math.sin(math.radians(heading))
                if quadrant_judgement(heading) == 3:
                    # x变小， y变小
                    delta_x = -diff_ododata * math.cos(math.radians(360 - heading))
                    delta_y = -diff_ododata * math.sin(math.radians(360 - heading))
                if quadrant_judgement(heading) == 4:
                    # x变大， y变小
                    delta_x = diff_ododata * math.sin(math.radians(270 - heading))
                    delta_y = -diff_ododata * math.cos(math.radians(270 - heading))

        # 更新位置
        position = [start_point[0] + delta_x, start_point[1] + delta_y]
        # 计算distance
        part_distance = distance_between_points(position[0], position[1], origin_point[0], origin_point[1])
        global_distance = distance_between_points(position[0], position[1], 0, 0)

        # print('此时里程计数据:', round(last_odometer_list[-1], 2), '上一时刻里程计数据:',
        #       round(last_odometer_list[-2], 2), '里程差:',
        #       diff_ododata)

        logger.info('里程差:{} | 小车起点:{} | 小车终点:{} | part_distance:{}'.format(diff_ododata,
                                                                                      [round(x, 2) for x in
                                                                                       start_point],
                                                                                      [round(x, 2) for x in position],
                                                                                      round(part_distance, 2)))
        # print('小车起点:', [round(x, 2) for x in start_point], '小车终点:', [round(x, 2) for x in position],
        #       'part_distance:', round(part_distance, 2))

        return part_distance, global_distance, position


def distance_between_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def map_angle(angle):
    if 0 < angle < 180:
        return angle
    else:
        return (angle + 180) % 180


def rectangle_center(x1, y1, x2, y2, x3, y3, x4, y4):
    """ 中心点位 """
    midpoint_ab = ((x1 + x2) / 2, (y1 + y2) / 2)
    midpoint_cd = ((x3 + x4) / 2, (y3 + y4) / 2)
    center_x = (midpoint_ab[0] + midpoint_cd[0]) / 2
    center_y = (midpoint_ab[1] + midpoint_cd[1]) / 2
    return center_x, center_y


def calculate_slope_and_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def midpoint(point1, point2):
    """
    计算两点的中点坐标
    """
    x = (point1[0] + point2[0]) / 2
    y = (point1[1] + point2[1]) / 2
    return (x, y)


class TrackingAlgo:

    def __init__(self) -> None:

        with open('/home/yimu/zhonghe/algorithm/zh_trackingposition/config/config.yaml', 'r') as config_file:
            # with open('config/config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.data_df = None
        self.request_flag = 2
        self.response_flag = 0
        self.reset_flag = 0  # 十字路口重置标志位

        self.default_image = cv2.imread(config["default_image_path"])
        self.image = self.default_image  # 当前帧的图像输出流
        self.frequency = config['frequency']  # 目前的请求频率(根据相机帧数)
        self.lens2head_distance = config['lens2head_distance']  # 镜头到焊头的实际距离（mm）

        # 当前视觉引导的行架控制
        self.vision_move_speed = 0  # 使用速度控制，速度参数输出，行进方向向左偏移为-，向右为+
        self.vision_move_to_location = config['vision_move_to_location']  # 使用位置控制，位置参数输出
        self.vision_move_compensate = config['vision_move_compensate']  # 行架补偿参数
        self.pulse_distance_coefficient = config['pulse_distance_coefficient']  # 脉冲距离转化系数

        self.current_move_speed = 0  # 当前行架行进速度
        self.current_location = 0  # 当前行架位置
        self.v_to_h = config['v_to_h']  # 镜头中心到探测头中心的实际距离(mm)
        self.deformation_x = config['deformation_x']  # 实际距离(mm)/像素(pixelim'sho'w)的换算系数
        self.target_offset = 0  # 实际距离偏移值(mm)，以焊缝中心为标准，焊缝检测循迹时的偏移量，行进方向向左偏移为-，向右为+。
        self.bias = 0  # 偏移像素值

        # 当前陀螺仪，里程计引导的小车定位
        self.global_x = 0  # 在地图中小车的全局坐标x
        self.global_y = 0  # 在 地图中小车的全局坐标y
        self.global_distance = 0  # 在地图中小车的已行动距离
        self.part_distance = 0  # 从焊缝开始计算，小车在单条焊缝检测循迹中的已行动距离

        self.width_target = 0  # 图像中焊缝宽度
        self.angle_yaw = 0  # 角度偏航(弧度)
        self.width_yaw = 0  # 单轴运动平台偏航，行进方向向左偏移为-，向右为+.

        self.odometer = [0, 0]  # 里程计数据list
        self.heading = None  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.border_percent = config['border_percent']  # 边界系数
        self.pixel_distance = 0  # 中心偏移的像素值
        self.real_movement_distance = None  # 真实移动的距离 cm

        self.yolo_model = YOLO(config['yolo'])  # 加载yolo模型
        self.device = 'cpu'
        self.yolo_results = None

        # 定义起点，原点坐标, origin_point 为焊缝起点坐标， start_point 需要每次更新
        self.origin_point = [self.global_x, self.global_y]
        self.start_point = [self.global_x, self.global_y]

        self.each_last_odometer = [0, 0]  # 上一时刻的里程计数据
        self.last_heading = []
        self.global_iteration1 = 0
        self.global_iteration2 = 0

        self.response_flag_lis = []
        self.vision_move_to_location_lis = []

        # 保存实验数据
        self.data_list = []
        self.gryo_x = []
        self.gryo_y = []
        self.gryo_z = []

    @staticmethod
    def mat2bytes(image: np.ndarray) -> bytes:
        return np.array(cv2.imencode('.jpg', image)[1]).tobytes()
        return np.array(cv2.imencode('.jpg', image)[1]).tobytes()

    @staticmethod
    def cal_centers_obb(results):
        pred = results[0].obb
        boxes = np.array(pred.xyxyxyxy.cpu(), dtype=np.int).squeeze()
        if len(boxes) != 0:
            black_index_mid = rectangle_center(boxes[0][0], boxes[0][1],
                                               boxes[1][0], boxes[1][1],
                                               boxes[2][0], boxes[2][1],
                                               boxes[3][0], boxes[3][1], )
            return black_index_mid, boxes
        else:
            return [0, 0], boxes

    @staticmethod
    def check_class(results, cls):

        """ check检测项 """
        # todo 可以把yolo模型放在一起, 0 是line，1是cross
        pred = results[0].obb
        if pred.cls.tolist().count(cls) > 0:
            return True
        else:
            return False

    def process_heading(self, heading, global_iteration):

        self.heading = remove_outliers(heading)
        mean_heading = np.mean(np.array(self.heading))
        if global_iteration == 0:
            self.last_heading.extend([mean_heading, mean_heading])
        else:
            self.last_heading.append(mean_heading)
            if global_iteration > 0 and rotation_warnings(self.last_heading[-1], self.last_heading[-2]):
                logger.warning('小车发生转弯！！！')

        self.last_heading = [x for x in self.last_heading if not math.isnan(x)]

        return self.last_heading

    def reading_data(self, channel_n_port):
        """ 读取数据线程 """
        with grpc.insecure_channel(channel_n_port) as channel:
            stub = zhonghe_tracking_pb2_grpc.ZhongheTrackingStub(channel)
            img_bytes = self.mat2bytes(self.image)
            requests = zhonghe_tracking_pb2.TrackingRequest(request_flag=self.request_flag,
                                                            reset_flag=self.reset_flag,
                                                            timestamp=time.time(),
                                                            image=img_bytes,
                                                            vision_move_speed=self.vision_move_speed,
                                                            vision_move_to_location=self.vision_move_to_location,
                                                            global_x=self.global_x,
                                                            global_y=self.global_y,
                                                            global_distance=self.global_distance,
                                                            part_distance=self.part_distance,
                                                            width_target=self.width_target,
                                                            angle_yaw=self.angle_yaw,
                                                            width_yaw=self.width_yaw)

            response = stub.TrackingOperation(requests)

            self.data_list.append({
                'image': self.image,
                'response_flag': response.response_flag,
                'vision_move_to_location': response.vision_move_to_location,
                'reset_flag': response.reset_flag,
                'odometer': response.odometer,
                'heading': response.heading,
                'global_x': response.global_x,
                'global_y': response.global_y})

    def truss_move_and_position(self):

        """ 行架控制和定位线程 """
        selected_dict = self.data_list[0]
        # 读取数据
        self.image = selected_dict['image']
        self.response_flag = selected_dict['response_flag']
        self.reset_flag = selected_dict['reset_flag']
        self.vision_move_to_location = selected_dict['vision_move_to_location']
        self.heading = selected_dict['heading']
        self.odometer = selected_dict['odometer']
        self.global_x = selected_dict['global_x']
        self.global_y = selected_dict['global_y']

        logger.info('里程计读数:{}'.format(np.mean(np.array(self.odometer))))
        logger.info('response_flag: {}'.format(self.response_flag))
        logger.info('reset_flag: {}'.format(self.reset_flag))

        # 清数字
        if len(self.each_last_odometer) > 10:
            self.each_last_odometer = self.each_last_odometer[-5:]
        if len(self.last_heading) > 10:
            self.last_heading = self.last_heading[-5:]
        if len(self.response_flag_lis) > 10:
            self.response_flag_lis = self.response_flag_lis[-5:]
        if len(self.vision_move_to_location_lis) > 10:
            self.vision_move_to_location_lis = self.vision_move_to_location_lis[-5:]

        self.yolo_results = self.yolo_model.predict(self.image, save=False, imgsz=640, conf=0.5, device=self.device)

        # check 陀螺仪和里程计是否有数据
        if len(self.heading) == 0:
            logger.critical('陀螺仪没数据！！！')
        if len(self.odometer) == 0:
            logger.critical('里程计没数据！！！')
        # todo 如果检测到十字路口，重置，flag 1十字路口，0没十字路口
        if self.check_class(self.yolo_results, 1):
            self.reset_flag = 1
        else:
            self.reset_flag = 0

        # 处理航向角数据， self.last_heading是一个列表
        self.last_heading = self.process_heading(self.heading, self.global_iteration1)
        self.angle_yaw = self.last_heading[-1] - self.last_heading[-2]

        # todo response flag发给算法，更新焊缝起点
        if self.reset_flag == 1 or self.response_flag == 1:
            self.origin_point = [self.global_x, self.global_y]
            self.start_point = [self.global_x, self.global_y]
            self.vision_move_to_location = 0

        if self.response_flag == 2:
            # 线被识别到，里程计没数据只做行架控制
            if self.check_class(self.yolo_results, 0) and len(self.heading) > 0 and \
                    len(self.odometer) == 0:
                logger.info('只接收到航向角时间: {}'.format(time.time()))
                print('桁架位置:', self.vision_move_to_location)
                self.truss_move()

            # 线被识别到，都有数据，做位置更新和行架控制
            elif self.check_class(self.yolo_results, 0) and len(self.heading) > 0 and \
                    len(self.odometer) > 0:
                logger.info('所有数据读取时间: {}'.format(time.time()))
                self.each_last_odometer.append(self.odometer[-1])
                self.truss_move()
                self.update_position()

            # 线没被识别到，只做位置更新
            elif not self.check_class(self.yolo_results, 0) and len(self.odometer) > 0:
                logger.warning('Line is not detected!!!')
                self.each_last_odometer.append(self.odometer[-1])
                self.update_position()

            elif len(self.odometer) == 0 or len(self.heading) == 0:
                pass
            # self.data_list.pop(0)
            logger.info('date_list_len:{}'.format(len(self.data_list)))

    def tracking_operation_bak(self, channel_n_port):

        while True:
            data_thread = threading.Thread(target=self.reading_data, args=(channel_n_port,))
            data_thread.start()
            car_thread = threading.Thread(target=self.truss_move_and_position)
            car_thread.start()

    def tracking_operation(self, channel_n_port):

        with grpc.insecure_channel(channel_n_port) as channel:
            stub = zhonghe_tracking_pb2_grpc.ZhongheTrackingStub(channel)

            if len(self.each_last_odometer) > 10:
                self.each_last_odometer = self.each_last_odometer[-5:]
            if len(self.last_heading) > 10:
                self.last_heading = self.last_heading[-5:]
            if len(self.response_flag_lis) > 10:
                self.response_flag_lis = self.response_flag_lis[-5:]
            if len(self.vision_move_to_location_lis) > 10:
                self.vision_move_to_location_lis = self.vision_move_to_location_lis[-5:]

            img_bytes = self.mat2bytes(self.image)
            requests = zhonghe_tracking_pb2.TrackingRequest(request_flag=self.request_flag,
                                                            reset_flag=self.reset_flag,
                                                            timestamp=time.time(),
                                                            image=img_bytes,
                                                            vision_move_speed=self.vision_move_speed,
                                                            vision_move_to_location=self.vision_move_to_location,
                                                            global_x=self.global_x,
                                                            global_y=self.global_y,
                                                            global_distance=self.global_distance,
                                                            part_distance=self.part_distance,
                                                            width_target=self.width_target,
                                                            angle_yaw=self.angle_yaw,
                                                            width_yaw=self.width_yaw)

            time.sleep(0.05)
            logger.info('请求时间: {}'.format(requests.timestamp))
            try:
                response = stub.TrackingOperation(requests)
                logger.info('response_flag: {}'.format(response.response_flag))
                self.vision_move_to_location = response.vision_move_to_location

                logger.info('里程计读数:{}'.format(np.mean(np.array(self.odometer))))
                logger.info('原来桁架位置：{}'.format(response.vision_move_to_location))
                self.response_flag = response.response_flag
                self.response_flag_lis.append(self.response_flag)

                self.heading = response.heading
                self.odometer = response.odometer
                self.gryo_x = response.gyroscope_x
                self.gryo_y = response.gyroscope_y
                self.gryo_z = response.gyroscope_z

                # yolo检测路口和辅助线
                self.yolo_results = self.yolo_model.predict(self.image, save=False, imgsz=640, conf=0.5,
                                                            device=self.device)

                # check 陀螺仪和里程计是否有数据
                if len(response.heading) == 0:
                    logger.critical('陀螺仪没数据！！！')
                if len(response.odometer) == 0:
                    logger.critical('里程计没数据！！！')
                # todo 如果检测到十字路口，重置，flag 1十字路口，0没十字路口
                if self.check_class(self.yolo_results, 1):
                    self.reset_flag = 1
                else:
                    self.reset_flag = 0

                # 处理航向角数据
                self.last_heading = self.process_heading(self.heading, self.global_iteration1)
                logger.info('航向列表：{} | {}'.format(self.last_heading[-1], self.last_heading[-2]))
                logger.info('航向偏移：{}'.format(self.angle_yaw))

                # todo response flag发给算法，更新焊缝起点
                if response.reset_flag == 1 or response.response_flag == 1:
                    self.origin_point = [response.global_x, response.global_y]
                    self.start_point = [response.global_x, response.global_y]
                    self.vision_move_to_location = 0

                elif self.response_flag == 2:
                    # 线被识别到，里程计没数据只做行架控制
                    if self.check_class(self.yolo_results, 0) and len(response.heading) > 0 and \
                            len(response.odometer) == 0:
                        logger.info('只接收到航向角时间: {}'.format(time.time()))
                        print('桁架位置:', self.vision_move_to_location)
                        self.truss_move()

                    # 线被识别到，都有数据，做位置更新和行架控制
                    elif self.check_class(self.yolo_results, 0) and len(response.heading) > 0 and \
                            len(response.odometer) > 0:
                        logger.info('所有数据读取时间: {}'.format(time.time()))
                        self.each_last_odometer.append(response.odometer[-1])
                        self.truss_move()
                        self.update_position()

                    # 线没被识别到，只做位置更新
                    elif not self.check_class(self.yolo_results, 0) and len(response.odometer) > 0:
                        logger.warning('Line is not detected!!!')
                        self.each_last_odometer.append(response.odometer[-1])
                        self.update_position()

                    elif len(response.odometer) == 0 or len(response.heading) == 0:
                        pass

            except Exception as e:
                pass

    def distance_calc2(self):
        border_percent = 9 / 20
        src_h, src_w, _ = self.image.shape

        # 取中间一行
        mid_h = int(src_h / 2)
        mid_w = int(src_w / 2)
        # results = self.yolo_model.predict(self.image, save=True, imgsz=640, conf=0.5)
        black_index_mid, boxes_xy = self.cal_centers_obb(self.yolo_results)

        boxes_y = [i[1] for i in boxes_xy]
        idx = [i for i, num in enumerate(boxes_y) if abs(num - boxes_y[0]) <= 50][1]

        # 没线的情况
        if black_index_mid == [0, 0]:
            black_index_mid_x = 0
            self.pixel_distance = 0
        else:
            black_index_mid_x, black_index_mid_y = int(black_index_mid[0]), int(black_index_mid[1])

        # 线box基准中心点坐标
        base_mid_point = midpoint((boxes_xy[0][0], boxes_xy[0][1]), (boxes_xy[idx][0], boxes_xy[idx][1]))
        # 计算线的斜率和截距
        midTocenter = calculate_slope_and_intercept(black_index_mid_x, src_h - black_index_mid_y, base_mid_point[0],
                                                    src_h - base_mid_point[1])

        # cv2.circle(self.image, (int(boxes_xy[0][0]), int(boxes_xy[0][1])), 5, (0, 0, 255), -1)
        # cv2.circle(self.image, (int(boxes_xy[idx][0]), int(boxes_xy[idx][1])), 5, (0, 0, 255), -1)
        # # 可视化base_mid_point
        # cv2.circle(self.image, (int(base_mid_point[0]), int(base_mid_point[1])), 5, (0, 0, 255), -1)
        # cv2.circle(self.image, (black_index_mid_x, black_index_mid_y), 5, (0, 0, 255), -1)
        # cv2.imshow('box', self.image)
        # cv2.imwrite('test.jpg', self.image)
        # cv2.waitKey(0)

        # 角度偏移导致的像素差计算
        angle_yaw_radians = self.angle_yaw
        bias_from_angle = math.sin(angle_yaw_radians) * self.v_to_h / self.deformation_x
        # 如果没有找到中线，则默认不移动
        if black_index_mid_x == 0:
            self.pixel_distance = 0
            cv2.arrowedLine(self.image, (mid_h, mid_w), (mid_h, mid_w), (0, 0, 255), 5)
        else:
            self.pixel_distance = black_index_mid_x - mid_w
            # 由于分次循迹的偏移量转换像素偏移的像素差计算
            self.target_offset_to_bias()

            self.pixel_distance = self.pixel_distance + self.bias - bias_from_angle
            intersection_point_x = (src_h - mid_h - midTocenter[1]) / midTocenter[0]
            extra_pixel_distance = intersection_point_x - black_index_mid_x

            self.real_movement_distance = 0.0151 * (self.pixel_distance + extra_pixel_distance)
            logger.info('桁架需要移动:{}cm'.format(self.real_movement_distance))

            # 更新新位置给软件
            self.vision_move_to_location = (self.vision_move_to_location +
                                            self.real_movement_distance * self.pulse_distance_coefficient
                                            - self.vision_move_compensate)

            self.vision_move_to_location_lis.append(self.vision_move_to_location)
            logger.info('行架移动后位置：{}'.format(self.vision_move_to_location))

            cv2.arrowedLine(self.image, (mid_w, mid_h),
                            (mid_w + int(self.pixel_distance + extra_pixel_distance), mid_h), (0, 0, 255), 5)
            # 保存图片
            # cv2.imwrite('box2.jpg', self.image)

    def speed_calc(self):
        # 计算当前帧移动时间，这边考虑到加速度过程时间，建议乘以1.5，以起到加速的作用。
        cost_time = 1 / (self.frequency * 1.5)
        # todo 速度 cm/s， 需要达到这个速度
        # self.vision_move_speed = abs((self.real_movement_distance / cost_time) * self.deformation_x)
        self.vision_move_speed = 8000

    def target_offset_to_bias(self):
        self.bias = self.target_offset / self.deformation_x

    def update_position(self):

        self.part_distance, self.global_distance, self.start_point = position_calculate2(self.start_point,
                                                                                         self.last_heading[-1],
                                                                                         self.last_heading[-2],
                                                                                         self.each_last_odometer,
                                                                                         self.origin_point)
        # 更新给软件坐标
        self.global_x = self.start_point[0]
        self.global_y = self.start_point[1]

    def truss_move(self):

        """ 行架控制 """
        # 更新self.bias
        self.target_offset_to_bias()
        # 更新self.pixel_distance 和 self.bias
        self.distance_calc2()
        self.speed_calc()

    def update_data_list(self):
        data = {
            'ts': time.time(),
            'odometer': self.odometer,
            'last_odometer': self.odometer[-1],
            'heading': self.heading,
            'avg_heading': np.mean(np.array(remove_outliers(self.heading))),
            'gryo_x': self.gryo_x,
            'gryo_y': self.gryo_y,
            'gryo_z': self.gryo_z
        }

        self.data_list.append(data)

    def save_list_to_dataframe(self):
        self.data_df = pd.DataFrame(self.data_list)
        self.data_df.to_csv('data3.csv', index=False)


if __name__ == '__main__':
    TA = TrackingAlgo()

    TA.target_offset_to_bias()
    TA.distance_calc2()
    cv2.imshow('box', TA.image)
    cv2.waitKey()

    # try:
    #     while True:
    #         TA.tracking_operation('192.168.6.181:5000')
    #         TA.update_data_list()
    #
    # except KeyboardInterrupt:
    #     TA.save_list_to_dataframe()
    #     # pass

    temp = 1
