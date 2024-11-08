import numpy as np
import math


def position_calculate(start_point, heading_data, odometer_data):
    """ 计算单步位置信息 """

    global delta_x, delta_y
    if 0 < heading_data < 90:
        delta_x = -odometer_data * math.cos(math.radians(heading_data))
        delta_y = odometer_data * math.sin(math.radians(heading_data))
    if 90 < heading_data < 180:
        delta_x = odometer_data * math.cos(math.radians(180 - heading_data))
        delta_y = odometer_data * math.sin(math.radians(180 - heading_data))
    if 180 < heading_data < 270:
        delta_x = odometer_data * math.cos(math.radians(heading_data - 180))
        delta_y = -odometer_data * math.sin(math.radians(heading_data - 180))
    if 270 < heading_data < 360:
        delta_x = -odometer_data * math.cos(math.radians(360 - heading_data))
        delta_y = -odometer_data * math.sin(math.radians(360 - heading_data))

    # 更新位置
    position = [start_point[0] + delta_x, start_point[1] + delta_y]
    # 计算distance
    distance = np.sqrt(np.power(position[0], 2) + np.power(position[1], 2))
    print('distance: ', distance)
    return distance, position


class Positioning:

    def __init__(self):
        """ 下面数据都得被真实数据所代替 """

        self.end_point = None
        self.heading_data = None  # 航向角
        self.part_distance = None  # 里程计

    def update_position(self, start):
        while True:
            step_meter = self.part_distance[-1] - self.part_distance[0]
            distance, p = position_calculate(start[-1], self.heading_data, step_meter)
            start.append(p)

            self.end_point = start[-1]
            dist = np.sqrt(np.power(self.end_point[0], 2) + np.power(self.end_point[1], 2))
            print('Now position:', start[-1], 'distance: ', dist)

            return start[-1], distance
