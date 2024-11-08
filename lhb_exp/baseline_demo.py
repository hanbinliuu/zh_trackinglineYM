import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    return position


class Positioning:

    def __init__(self, pry_file, meter_file):

        """ 下面数据都得被真实数据所代替 """

        self.end_point = None

        self.pry_data = pd.read_csv(pry_file)
        self.meter_data = pd.read_csv(meter_file)[1:]

        self.pry_ts = self.pry_data['时间'].values
        self.heading_data = np.round(self.pry_data['heading'].values * 57.32, 2)
        self.meter_data_ts = self.meter_data["时间"].values.tolist()
        self.meter = self.meter_data['上排数据'].values * -1

    @staticmethod
    def process_data(self):
        # 数据处理代码
        matches_indices = []
        available_indices = list(range(len(self.pry_ts)))

        for a_val in self.meter_data_ts:
            min_distance = float('inf')
            closest_index = None

            for i, b_val in enumerate(self.pry_ts):
                distance = abs(a_val - b_val)
                if distance < min_distance and i in available_indices:
                    min_distance = distance
                    closest_index = i

            if closest_index is not None:
                matches_indices.append(closest_index)
                available_indices.remove(closest_index)

        # Sort the matches_indices to ensure the order
        matches_indices.sort()

        meters = self.meter[matches_indices]
        new_heading = self.heading_data[matches_indices]
        diff_meter_each_point = np.diff(meters)

        return new_heading, diff_meter_each_point

    @staticmethod
    def match_timestamp(short_ts, long_ts):
        # 数据处理代码
        matches_indices = []
        available_indices = list(range(len(long_ts)))

        for a_val in short_ts:
            min_distance = float('inf')
            closest_index = None

            for i in available_indices:
                b_val = long_ts[i]
                distance = abs(a_val - b_val)

                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

            if closest_index is not None:
                matches_indices.append(closest_index)
                available_indices.remove(closest_index)

        return matches_indices

    def update_position(self, start):

        match_idx = self.match_timestamp(short_ts=self.meter_data_ts, long_ts=self.pry_ts)
        self.heading_data = self.heading_data[match_idx]
        step_meter = np.diff(self.meter)
        for i in range(len(self.heading_data) - 1):
            p = position_calculate(start[-1], self.heading_data[i], step_meter[i])
            start.append(p)

        self.end_point = start[-1]
        distance = np.sqrt(np.power(self.end_point[0], 2) + np.power(self.end_point[1], 2))
        print('distance: ', distance)

        return start

    def update_position2(self, start):
        sp = []
        match_idx = self.match_timestamp(short_ts=self.meter_data_ts, long_ts=self.pry_ts)
        self.heading_data = self.heading_data[match_idx]
        step_meter = np.diff(self.meter)
        for i in range(len(self.heading_data) - 1):
            p = position_calculate3(start, self.heading_data[i], step_meter[i])
            sp.append(p)
        self.end_point = start[-1]
        distance = np.sqrt(np.power(self.end_point[0], 2) + np.power(self.end_point[1], 2))
        print('distance: ', distance)
        return sp


def position_calculate3(start_point, heading_data, diff_ododata):
    """ 计算单步位置信息 """

    # 小车前进
    if 0 <= heading_data < 90 and diff_ododata >= 0:
        # delta_x = -diff_ododata * math.cos(heading_data)
        # delta_y = -diff_ododata * math.sin(heading_data)
        delta_x = -diff_ododata * math.cos(math.radians(heading_data))
        delta_y = diff_ododata * math.sin(math.radians(heading_data))

    if 90 <= heading_data < 180 and diff_ododata >= 0:
        # delta_x = diff_ododata * math.cos(180 - heading_data)
        # delta_y = -diff_ododata * math.sin(180 - heading_data)
        delta_x = diff_ododata * math.cos(math.radians(180 - heading_data))
        delta_y = diff_ododata * math.sin(math.radians(180 - heading_data))

    if 180 <= heading_data < 270 and diff_ododata >= 0:
        # delta_x = diff_ododata * math.cos(heading_data - 180)
        # delta_y = diff_ododata * math.sin(heading_data - 180)
        delta_x = diff_ododata * math.cos(math.radians(heading_data - 180))
        delta_y = -diff_ododata * math.sin(math.radians(heading_data - 180))

    if 270 <= heading_data < 360 and diff_ododata >= 0:
        # delta_x = -diff_ododata * math.cos(360 - heading_data)
        # delta_y = diff_ododata * math.sin(360 - heading_data)
        delta_x = -diff_ododata * math.cos(math.radians(360 - heading_data))
        delta_y = -diff_ododata * math.sin(math.radians(360 - heading_data))

    # 小车回退
    if (0 <= heading_data < 90) and diff_ododata < 0:
        delta_x = diff_ododata * math.cos(math.radians(heading_data))
        delta_y = -diff_ododata * math.sin(math.radians(heading_data))

    if 90 <= heading_data < 180 and diff_ododata < 0:
        delta_x = -diff_ododata * math.cos(math.radians(180 - heading_data))
        delta_y = -diff_ododata * math.sin(math.radians(180 - heading_data))

    if 180 <= heading_data < 270 and diff_ododata < 0:
        delta_x = -diff_ododata * math.cos(math.radians(heading_data - 180))
        delta_y = diff_ododata * math.sin(math.radians(heading_data - 180))

    if 270 <= heading_data < 360 and diff_ododata < 0:
        delta_x = diff_ododata * math.cos(math.radians(360 - heading_data))
        delta_y = diff_ododata * math.sin(math.radians(360 - heading_data))

    # 更新位置
    position = [start_point[0] + delta_x, start_point[1] + delta_y]
    # 计算distance
    distance = np.sqrt(np.power(position[0], 2) + np.power(position[1], 2))
    position = [round(position[0], 2), round(position[1], 2)]

    print('里程差:', round(diff_ododata, 2), 'heading:', round(heading_data, 2), '小车起点:', start_point,
          '小车终点:', position, 'distance:', round(distance, 2))

    return position


def plot_data(x, y):
    plt.figure(figsize=(12, 10))

    plt.plot(x, y, marker='.', linestyle='-', color='b')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Position Update Trajectory')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    pos = Positioning(r"D:\ym\焊缝寻迹定位\positioning\pry_2side.csv",
                      r"D:\ym\焊缝寻迹定位\positioning\odometer_2side.csv")

    position = pos.update_position2(start=[0, 0])

    x, y = [point[0] for point in position], [point[1] for point in position]
    plot_data(x, y)
