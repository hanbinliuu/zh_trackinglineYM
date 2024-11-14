import math
import time
from loguru import logger

import grpc
import cv2
import numpy as np
import pandas as pd

from protos import zhonghe_tracking_pb2_grpc, zhonghe_tracking_pb2
import math


def rotation_warnings(heading_data, last_heading_data):
    """ :param heading_data: 当前方位角float
        :param last_heading_data: list """

    if 0 < heading_data < 90:
        # 如果last_heading_data不是在这个区间
        if last_heading_data < 0 or last_heading_data > 90:
            logger.warning('小车发生转弯！！！')
            return True
    if 90 < heading_data < 180:
        if last_heading_data < 90 or last_heading_data > 180:
            logger.warning('小车发生转弯！！！')
            return True
    if 180 < heading_data < 270:
        if last_heading_data < 180 or last_heading_data > 270:
            logger.warning('小车发生转弯！！！')
            return True
    if 270 < heading_data < 360:
        if last_heading_data < 270 or last_heading_data > 360:
            logger.warning('小车发生转弯！！！')
            return True


# def calculate_input(consective_odometer, raw_gyroscope_z):
#     if len(consective_odometer) == 1:
#         diff_ododata = 0
#     else:
#         diff_ododata = consective_odometer[-1] - consective_odometer[-2]
#
#     v = diff_ododata / 0.8
#
#     mean_yaw = 0
#     u = np.array([[abs(v)], [mean_yaw]])
#     return u


# todo 测试用
def input(diff_t, diff_last_odometer):
    if iter == 0:
        u = np.array([[0], [0]])

    else:
        v = diff_last_odometer / diff_t
        u = np.array([[abs(v)], [0]])

    print('input:', u)
    return u


def input_info(diff_t, diff_last_odometer, angle_yaw):
    if iter == 0:
        u = np.array([[0], [0]])

    else:
        v = diff_last_odometer / diff_t
        u = np.array([[v], [angle_yaw]])
    return u


class EkfSimulation:
    """ EKF simulation class """

    def __init__(self, frequency):
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,
            0.1,
            np.deg2rad(1.0),
            1.0
        ]) ** 2

        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        # 噪声
        self.INPUT_NOISE = np.diag([.0, np.deg2rad(0)]) ** 2
        self.IMU_NOISE = np.diag([0.0, 0.0]) ** 2

        self.DT = frequency

    def observation(self, xTrue, u):
        xTrue = self.motion_model(xTrue, u)  # use the same input u for xTrue
        z = self.observation_model(xTrue) + self.IMU_NOISE @ np.random.randn(2, 1)
        return xTrue, z

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT],
                      [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    def observation_model(self, x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x

        return z

    def jacob_f(self, x, u):
        """
        Jacobian of Motion Model

        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.DT * v * math.sin(yaw), self.DT * math.cos(yaw)],
            [0.0, 1.0, self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF

    @property
    def jacob_h(self):
        # Jacobin of Observation Model
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        return jH

    def ekf_estimation(self, xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f(xEst, u)
        PPred = jF @ PEst @ jF.T + self.Q

        #  Update
        jH = self.jacob_h
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst


class TrackingAlgo:

    def __init__(self) -> None:

        self.tsList = [time.time()]  # 时间戳列表
        self.request_flag = 2
        self.response_flag = 0

        self.default_image = cv2.imread('test_img/test2.png')
        self.image = self.default_image  # 当前帧的图像输出流
        self.frequency = 8  # 目前的请求频率(根据相机帧数)
        self.ekf_frequency = 0.517  # ekf算法频率
        self.lens2head_distance = 0  # 镜头到焊头的实际距离（mm）

        # 当前视觉引导的行架控制
        self.vision_move_speed = 0  # 使用速度控制，速度参数输出，行进方向向左偏移为-，向右为+
        self.vision_move_to_location = 0  # 使用位置控制，位置参数输出

        self.current_move_speed = 0  # 当前行架行进速度
        self.current_location = 0  # 当前行架位置
        self.v_to_h = 11.5  # 镜头中心到探测头中心的实际距离(mm)
        self.deformation_x = 0.2  # 实际距离(mm)/像素(pixel)的换算系数
        self.target_offset = 0  # 实际距离偏移值(mm)，以焊缝中心为标准，焊缝检测循迹时的偏移量，行进方向向左偏移为-，向右为+。
        self.bias = 0  # 偏移像素值

        # 当前陀螺仪，里程计引导的小车定位
        self.global_x = 0  # 在地图中小车的全局坐标x
        self.global_y = 0  # 在 地图中小车的全局坐标y
        self.global_distance = 0  # 在地图中小车的已行动距离
        self.part_distance = 0  # 从焊缝开始计算，小车在单条焊缝检测循迹中的已行动距离

        self.width_target = 0  # 图像中焊缝宽度

        self.angle_yaw = -0.1  # 角度偏航(弧度)
        self.width_yaw = 0  # 单轴运动平台偏航，行进方向向左偏移为-，向右为+.

        self.odometer = [0, 0, 0]  # 里程计数据list
        self.gyroscope_x = []  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.gyroscope_y = []  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.gyroscope_z = []  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        # self.angle_yaw = np.mean(np.array(self.gyroscope_z))
        self.roll = []  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.pitch = []  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.heading = None  # 上一时刻到当前时刻的陀螺仪角度（弧度）列表
        self.border_percent = 9 / 20  # 边界系数
        self.pixel_distance = 0  # 中心偏移的像素值

        # 定义起点，原点坐标, origin_point 为焊缝起点坐标， start_point 需要每次更新
        self.origin_point = [0, 0]
        self.start_point = [self.global_x, self.global_y]

        self.last_odometer = 0
        self.last_odometer2 = [0, 0]

        # 初始化
        self.diff_ts = 0
        self.diff_odometer = 0
        self.last_heading = []
        self.diff_heading = None
        self.sim = EkfSimulation(frequency=self.ekf_frequency)

    @staticmethod
    def mat2bytes(image: np.ndarray) -> bytes:
        return np.array(cv2.imencode('.jpg', image)[1]).tobytes()

    def tracking_operation(self, channel_n_port):
        with grpc.insecure_channel(channel_n_port) as channel:
            stub = zhonghe_tracking_pb2_grpc.ZhongheTrackingStub(channel)
            # self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            # clear
            if len(self.tsList) > 10:
                self.tsList = self.tsList[-5:]
            if len(self.last_odometer2) > 10:
                self.last_odometer2 = self.last_odometer2[-5:]
            if len(self.last_heading) > 10:
                self.last_heading = self.last_heading[-5:]

            # 传图片进来
            logger.info('receive image time: {}'.format(time.time()))
            img_bytes = self.mat2bytes(self.image)
            requests = zhonghe_tracking_pb2.TrackingRequest(request_flag=self.request_flag,
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
                                                            width_yaw=self.width_yaw,
                                                            last_odometer=self.last_odometer)

            time.sleep(0.5)

            # todo response flag发给算法，需要更新原点坐标x
            # if response.response_flag == 1:
            #     self.origin_point = [response.global_x, response.global_y]

            try:
                response = stub.TrackingOperation(requests)
                print('里程计res:', response.odometer)
                logger.info('receive response time: {}'.format(time.time()))

                if len(response.odometer) > 1 and len(response.heading) > 1:
                    # 更新连续里程计列表
                    self.last_odometer2.append(response.odometer[-1])
                    self.odometer = response.odometer
                    self.gyroscope_z = response.gyroscope_z
                    self.angle_yaw = np.mean(np.array(self.gyroscope_z))
                    self.heading = response.heading

                    # 第一次读取的航向角, append两遍
                    if iter == 0:
                        self.last_heading.append(np.mean(np.array(self.heading)))
                        self.last_heading.append(np.mean(np.array(self.heading)))
                    else:
                        self.last_heading.append(np.mean(np.array(self.heading)))
                        # 小车转弯提醒
                        if rotation_warnings(self.last_heading[-1], self.last_heading[-2]):
                            logger.warning('小车发生转弯！！！')

                    # v = s/t
                    self.tsList.append(requests.timestamp)
                    self.diff_ts = self.tsList[-1] - self.tsList[-2]
                    self.diff_odometer = self.last_odometer2[-1] - self.last_odometer2[-2]
                    self.diff_heading = self.last_heading[-1] - self.last_heading[-2]
                    print('偏航角:', self.diff_heading)
                    self.angle_yaw = self.diff_heading
                    self.data_update()

            except Exception as e:
                logger.error('error: {}'.format(e))

    def distance_calc(self):
        pixel_distance = 0
        border_percent = 9 / 20

        src_h, src_w, _ = self.image.shape
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        mid_h = int(src_h / 2)
        mid_w = int(src_w / 2)
        mid_line = dst[mid_h]

        left_border = int(src_w * border_percent)
        right_border = int(src_w * (1 - border_percent))
        black_count = np.sum(mid_line == 0)
        black_index = np.where(mid_line == 0)
        black_index_mid = black_index[0][int(len(black_index[0]) / 2)]
        # 角度偏移导致的像素差计算
        angle_yaw_radians = self.angle_yaw
        bias_from_angle = math.sin(angle_yaw_radians) * self.v_to_h / self.deformation_x
        # 如果没有找到中线，则默认不移动
        if black_index_mid == 0:
            self.pixel_distance = 0
            cv2.arrowedLine(self.image, (mid_h, mid_w), (mid_h, mid_w), (0, 0, 255), 5)
        else:
            self.pixel_distance = black_index_mid - mid_w
            # 由于分次循迹的偏移量转换像素偏移的像素差计算
            self.target_offset_to_bias()

            self.pixel_distance = self.pixel_distance + self.bias - bias_from_angle
            cv2.arrowedLine(self.image, (mid_w, mid_h), (mid_w + int(self.pixel_distance), mid_h), (0, 0, 255), 5)

    def speed_calc(self):
        # 计算当前帧移动时间，这边考虑到加速度过程时间，建议乘以1.5，以起到加速的作用。
        cost_time = 1 / (self.frequency * 1.5)
        self.vision_move_speed = (self.pixel_distance / cost_time) * self.deformation_x

    def target_offset_to_bias(self):
        self.bias = self.target_offset / self.deformation_x

    def run_localization(self):

        """ 定位 """

        # sim = EkfSimulation(frequency=self.ekf_frequency)

        result_array = np.zeros((4, 1))
        result_array[:2, 0] = [item for sublist in [self.start_point] for item in sublist]
        xEst, xTrue = result_array, result_array
        PEst = np.eye(4)

        # 测试本地data.csv数据
        # u = input(diff_ts[iter], diff_odometer[iter])
        u = input_info(self.diff_ts, self.diff_odometer, self.diff_heading)
        # 真实输入数据, ud是input的带噪声版本, z是观测值，正常测量值加上噪声就是z
        xTrue, z = self.sim.observation(xTrue, u)
        # 估计输入数据， 通过ud{速度，角速度}+noise，和观测值， 通过ekf运动模型得到xEst
        xEst2, PEst2 = self.sim.ekf_estimation(xEst, PEst, z, u)

        estimated_distance = np.sqrt(xEst2[0, 0] ** 2 + xEst2[1, 0] ** 2)
        print('小车起点: ', [self.start_point[0], self.start_point[1]],
              '小车终点:', [xEst2[:2][0][0], xEst2[:2][1][0]],
              '距离:', estimated_distance)

        # 更新数据给软件
        self.start_point = [xEst2[:2][0][0], xEst2[:2][1][0]]
        self.part_distance = estimated_distance
        self.global_x = xEst2[0, 0]
        self.global_y = xEst2[1, 0]

    def data_update(self):
        # self.target_offset_to_bias()
        # self.distance_calc()
        # self.speed_calc()
        self.run_localization()


def read_data(path):
    """ 测试用 """
    import pandas as pd
    data = pd.read_csv(path)

    ts = data['ts'].values
    diff_ts = np.diff(ts)
    diff_ts = np.insert(diff_ts, 0, 0)

    odo = data['odometer'].values
    last_numbers = []

    for array in odo:
        # 将字符串转换为列表
        array_list = eval(array)
        # 获取最后一个数字
        last_number = array_list[-1]
        last_numbers.append(last_number)
    last_numbers = np.array(last_numbers)

    # odometer = data['last_odometer'].values
    diff_odometer = np.diff(last_numbers)
    diff_odometer = np.insert(diff_odometer, 0, 0)

    return diff_ts, diff_odometer


if __name__ == '__main__':

    global iter
    iter = 0

    # logger.add('logs/zh.log', backtrace=True, diagnose=True, rotation='10 MB')
    TA = TrackingAlgo()
    # TA.distance_calc()
    # cv2.imshow('box', TA.image)
    # cv2.waitKey()

    # 读取数据
    # diff_ts, diff_odometer = read_data(r"C:\Users\lhb\Desktop\data3.csv")
    # for i in range(len(diff_ts)):
    #     TA.data_update()
    #     iter += 1

    while True:
        TA.tracking_operation('192.168.6.20:5000')
        iter += 1

    temp = 1
