#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from cubic_spline import CubicSpline2D
from polymal import Polynomial

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

# Parameter
ROBOT_RADIUS = 0.33
MAX_SPEED = 1.2  
MAX_ACCEL = 1.2  
MAX_CURVATURE = 3.3  

MAX_ROAD_WIDTH = ROBOT_RADIUS * 10.0  
D_ROAD_W = ROBOT_RADIUS * 2 
DT = 0.4125
MAX_T = ROBOT_RADIUS * 8.0 * 2  
MIN_T = ROBOT_RADIUS * 6.0 * 2 
# MAX_T = 4
# MIN_T = 1.3

TARGET_SPEED = 0.8  
D_T_S = 0.1
N_S_SAMPLE = 1 

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

K_LINEAR = [1.5, 0.9, 0]  # P, I, D
K_ANGULAR = [1.1, 0.4, 0]  # P, I, D

MAXIMUM_VLINEAR = 1.2
MAXIMUM_VANGULAR = 1.2  

waypoint_x = []
waypoint_y = []

def point_callback(msg):
    global waypoint_x, waypoint_y

    waypoint_x = []
    waypoint_y = []

    data = msg.data
    for i in range(0, len(data), 2):
        x = data[i]
        y = data[i + 1]
        waypoint_x.append(x)
        waypoint_y.append(y)

def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            lat_qp = Polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = Polynomial(xs=s0, vxs=c_speed, axs=c_accel, vxe=tv, axe=0.0, time=Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv
                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):
            continue
        # elif any([abs(c) > MAX_CURVATURE for c in
        #           fplist[i].c]):
        #     continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry= [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)

    return rx, ry, csp


def main():
    tx, ty, csp = generate_target_course(waypoint_x, waypoint_y)
    len_pathposes = len(tx)
    # obstacle lists
    ob = np.array([[tx[int(len_pathposes * 0.2)], ty[int(len_pathposes * 0.2)]],
                   [tx[int(len_pathposes * 0.25)], ty[int(len_pathposes * 0.25)]],
                   [tx[int(len_pathposes * 0.5)], ty[int(len_pathposes * 0.5)]],
                   [tx[int(len_pathposes * 0.6)], ty[int(len_pathposes * 0.6)]],
                   [tx[int(len_pathposes * 0.8)], ty[int(len_pathposes * 0.8)]],
                   [tx[int(len_pathposes * 0.82)], ty[int(len_pathposes * 0.82)]],
                   [tx[int(len_pathposes * 0.84)], ty[int(len_pathposes * 0.84)]],
                   [tx[int(len_pathposes * 0.86)], ty[int(len_pathposes * 0.86)]],
                   ])

    # ob = np.array([[0, 0]])

    # initial state
    c_speed = 0.0  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    
    plt.figure(figsize=(15, 10))

    prev_error_linear = 0.0
    prev_error_angular = 0.0

    while True:
        inst  = input("Press 'y' next step: ")
        if inst == 'y':        
            path = frenet_optimal_planning(
                csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)

            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]
            c_accel = path.s_dd[1]

            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 0.17:
                print("Goal")
                break

            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(tx, ty, "-xk")
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")

            plt.plot(path.x[0], path.y[0], "vc")
            # plt.plot(path.x[1], path.y[1], "vr")

            plt.xlim([-11, 11])
            plt.ylim([-11, 11])
            plt.gca().set_aspect('equal', adjustable='box')

            plt.title("v[m/s]:" + str(c_speed))
            plt.grid(True)
            plt.pause(0.0001)

            path_pub = Path()
            path_pub.header.frame_id = "map"
            path_pub.header.stamp = rospy.Time.now()

            # Iterate over each point (x, y) in self.generator.curves
            for i in range(len(path.x)):
                if i != 0:
                    curve_x, curve_y = path.x[i], path.y[i]
                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = "map"
                    pose_stamped.header.stamp = rospy.Time.now()

                    # Set position based on the curve points
                    pose_stamped.pose.position.x = float(curve_x)
                    pose_stamped.pose.position.y = float(curve_y)

                    # Append the PoseStamped to the path
                    path_pub.poses.append(pose_stamped)

            # Publish the entire path
            publish_topic_path.publish(path_pub)

    plt.grid(True)
    plt.pause(0.0001)
    plt.show()


if __name__ == '__main__':
    rospy.init_node('waypoint_subscriber', anonymous=True)
    rospy.Subscriber('/points', Float32MultiArray, point_callback)
    optimal_path = rospy.Publisher('/otimal_path', Float32MultiArray, queue_size=10)
    publish_topic_path = rospy.Publisher('/path', Path, queue_size=10)

    inst  = input("Press 'y' to start: ")
    if inst == 'y':
        main()
