import math
import numpy as np
import matplotlib.pyplot as plt


def calc_input(odometer, raw_gyroscope_z, delta_t):

    """ :param delta_t: 采样时间
        :param odometer 里程计数据list
        :param raw_gyroscope_z 陀螺仪数据list """

    # 时间段内里程计路程
    # diff_meter = max(odometer) - min(odometer)
    diff_meter = odometer[-1] - odometer[0]
    v = diff_meter / delta_t
    mean_yaw = np.mean(np.array(raw_gyroscope_z))

    if v >= 0:
        u = np.array([[v], [mean_yaw]])
    else:
        u = np.array([[-v], [mean_yaw]])

    return u


class EkfSimulation:
    """ EKF simulation class """

    def __init__(self, gyro_z, odometer, frequency):
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,
            0.1,
            np.deg2rad(1.0),
            1.0
        ]) ** 2

        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        #  Simulation parameter
        self.INPUT_NOISE = np.diag([.0, np.deg2rad(0)]) ** 2
        self.IMU_NOISE = np.diag([0.0, 0.0]) ** 2

        self.DT = frequency
        self.show_animation = False

        # 软件返回值读取
        self.gyro_z = gyro_z
        self.odometer = odometer

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

    def run_localization(self, start_point=None):

        # 定义起始点坐标
        if start_point is None:
            start_point = [0, 0]

        result_array = np.zeros((4, 1))
        result_array[:len(start_point), 0] = start_point
        xEst, xTrue = result_array, result_array
        PEst = np.eye(4)
        hxEst, hxTrue = xEst, xTrue

        estimated_distances = []
        while True:

            # todo 读取更新数据
            u = calc_input(self.odometer, self.gyro_z, self.DT)

            # 真实输入数据, ud是input的带噪声版本, z是观测值，正常测量值加上噪声就是z
            xTrue, z = self.observation(xTrue, u)
            # 估计输入数据， 通过ud{速度，角速度}+noise，和观测值， 通过ekf运动模型得到xEst
            xEst, PEst = self.ekf_estimation(xEst, PEst, z, u)

            estimated_distance = np.sqrt(xEst[0, 0] ** 2 + xEst[1, 0] ** 2)
            estimated_distances.append(estimated_distance)
            print('xEst: ', xEst[:2], 'distance:', estimated_distance)

            hxEst = np.hstack((hxEst, xEst))
            hxTrue = np.hstack((hxTrue, xTrue))

            if self.show_animation:
                plt.subplot(1, 2, 1)
                plt.cla()
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

            return estimated_distances, xEst[:2]