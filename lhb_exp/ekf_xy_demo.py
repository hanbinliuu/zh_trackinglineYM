import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot


def calc_input(v, yawrate):
    if v >= 0:
        u = np.array([[v], [yawrate]])
    else:
        u = np.array([[-v], [yawrate]])
    return u


class EKF_Simulation:
    """ 仿真 """
    def __init__(self):
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,
            0.1,
            np.deg2rad(1.0),
            1.0
        ]) ** 2

        self.R = np.diag([1.0, 1.0]) ** 2

        #  Simulation parameter
        self.INPUT_NOISE = np.diag([.0, np.deg2rad(0)]) ** 2
        self.IMU_NOISE = np.diag([0.0, 0.01]) ** 2

        self.DT = 0.2
        self.show_animation = True

    def observation(self, xTrue, u):
        xTrue = self.motion_model(xTrue, u)
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

    def jacob_h(self):
        # Jacobian of Observation Model
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
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    def plot_covariance_ellipse(self, xEst, PEst):
        Pxy = PEst[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
        rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
        fx = rot @ (np.array([x, y]))
        px = np.array(fx[0, :] + xEst[0, 0]).flatten()
        py = np.array(fx[1, :] + xEst[1, 0]).flatten()
        plt.plot(px, py, "--r")

    def run_localization(self, max_steps, all_v, all_yawrate):

        # State Vector [x y yaw v]'
        # initial the state vector with 0
        xEst = np.zeros((4, 1))
        xTrue = np.zeros((4, 1))
        PEst = np.eye(4)
        # history
        hxEst = xEst
        hxTrue = xTrue

        actual_distances = []
        estimated_distances = []

        for step in range(max_steps):  # 限制仿真步数

            # yawrate = yaw_rates[step]
            # v = 0.5

            # 每次产生新数据
            v, yawrate = all_v[step], all_yawrate[step]
            u = calc_input(v, yawrate)
            print('step:', step, 'u:', u)

            # 真实输入数据, ud是input的带噪声版本, z是观测值，正常测量值加上噪声就是z
            xTrue, z = self.observation(xTrue, u)
            # 估计输入数据， 通过ud{速度，角速度}+noise，和观测值， 通过ekf运动模型得到xEst
            xEst, PEst = self.ekf_estimation(xEst, PEst, z, u)
            # Calculate distances and append to the lists
            actual_distance = np.sqrt(xTrue[0, 0] ** 2 + xTrue[1, 0] ** 2)
            estimated_distance = np.sqrt(xEst[0, 0] ** 2 + xEst[1, 0] ** 2)
            actual_distances.append(actual_distance)
            estimated_distances.append(estimated_distance)
            print('step:', step, 'xEst: ', xEst[:2], 'yaw-rate:', yawrate, 'distance:', estimated_distance)

            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hxTrue = np.hstack((hxTrue, xTrue))

            if self.show_animation:
                plt.subplot(1, 2, 1)
                plt.cla()
                plt.plot(actual_distances, label='Actual Distance', color='green')
                plt.plot(estimated_distances, label='Estimated Distance', color='purple')
                plt.xlabel("Time Steps")
                plt.ylabel("Distance")
                plt.legend()
                plt.grid(True)
                plt.pause(0.001)

                plt.subplot(1, 2, 2)
                plt.cla()
                plt.plot(hxTrue[0, :].flatten(),
                         hxTrue[1, :].flatten(), "-b")
                plt.plot(hxEst[0, :].flatten(),
                         hxEst[1, :].flatten(), "-r")
                # self.plot_covariance_ellipse(xEst, PEst)
                plt.axis("equal")
                plt.legend(["Observation", "True", "Estimated"])  # Add legend
                plt.grid(True)
                plt.title('trajectory')
                plt.pause(0.001)

            if step == max_steps - 1:
                print("Final estimate distance from origin: {:.2f}".format(estimated_distance))
                print("Final actual distance from origin:", actual_distance)
                break  # 到达最大步数后退出仿真

        return estimated_distances


def get_new_data2():
    global step_counter

    if step_counter < 50:  # 前100步走直线
        v = 2  # 直线速度
        yawrate = np.random.uniform(-0.1, 0.1)
    elif step_counter < 60:  # 接下来的100步原地转弯90度
        v = 0.5
        yawrate = np.pi / 2  # 转弯90度
    elif step_counter < 100:
        v = 5  # 直线速度
        yawrate = np.random.uniform(-0.1, 0.1)
    elif step_counter < 120:
        v = 0.5
        yawrate = np.pi / 2
    elif step_counter < 150:
        v = 2
        yawrate = 0
    else:
        v = 3
        yawrate = 0

    step_counter += 1

    return v, yawrate


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


if __name__ == '__main__':
    """ observation model 根据前点xy更新 """

    dt = 1/8

    imu_data = pd.read_csv(r"D:\ym\焊缝寻迹定位\positioning\raw\imu_2side.csv")[20:]
    meter_data = pd.read_csv(r"D:\ym\焊缝寻迹定位\positioning\raw\odometer_2side.csv")[20:]
    pry_data = pd.read_csv(r"D:\ym\焊缝寻迹定位\positioning\raw\pry_2side.csv")
    raw = pd.read_csv(r"D:\ym\焊缝寻迹定位\positioning\raw\raw_2side.csv")

    yaw = raw['z'].values
    gyr_data = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
    raw_ts = raw['时间'].values
    imu_ts = imu_data['时间'].values
    meter_ts = meter_data["时间"].values
    matches_indices = match_timestamp(meter_ts, imu_ts)

    # 速度
    meter = meter_data['上排数据'].values * -1
    vel2 = np.insert(np.diff(meter) / np.diff(meter_ts), 0, 0)

    # 角速度
    step_counter = 0
    sim = EKF_Simulation()
    estimated_distances = sim.run_localization(len(vel2), vel2, yaw)
    temp = 1
