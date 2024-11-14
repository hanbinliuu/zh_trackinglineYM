import math

import numpy as np
import pandas as pd
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def distance_between_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


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

        print('此时里程计数据:', round(last_odometer_list[-1], 2), '上一时刻里程计数据:',
              round(last_odometer_list[-2], 2), '里程差:', diff_ododata)
        print('小车起点:', [round(x, 2) for x in start_point], '小车终点:', [round(x, 2) for x in position],
              'part_distance:', round(part_distance, 2))

        return part_distance, global_distance, position


if __name__ == '__main__':

    data = pd.read_csv(r"C:\Users\lhb\Desktop\data\data2.csv")
    data_len = len(data)

    x, y = [], []
    start_point = [0, 0]
    origin_point = [0, 0]

    for i in range(1, data_len):
        heading = data['avg_heading'][i]
        last_heading = data['avg_heading'][i-1]
        last_odometer = eval(data['odometer'][i])
        part_distance, global_distance, position = position_calculate2(start_point, heading, last_heading,
                                                                  last_odometer, origin_point)
        start_point = position
        x.append(position[0])
        y.append(position[1])

    plt.plot(x, y)
    plt.show()

    temp = 1

