#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf
import matplotlib.pyplot as plt
import numpy as np

ROBOT_RADIUS = 0.22

class CurveGenerator:
    def __init__(self, tension=0.5):
        self.points = []
        self.curves = []
        self.tension = tension

    def hermite_curve(self, p0, p1, t0, t1):
        distance = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        num_points = int(distance // ROBOT_RADIUS + 1)

        t = np.linspace(0, 1, num_points)
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
        y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]
        return x.tolist(), y.tolist()  # Ensure the return values are lists
            
    def create_tangent_points(self):
        self.tangent_points = []
        if len(self.points) < 2:
            return

        for i in range(len(self.points)):
            if i == 0:
                x_diff = self.points[1][0] - self.points[0][0]
                y_diff = self.points[1][1] - self.points[0][1]
            elif i == len(self.points) - 1:
                x_diff = self.points[i][0] - self.points[i - 1][0]
                y_diff = self.points[i][1] - self.points[i - 1][1]
            else:
                x_diff = self.points[i + 1][0] - self.points[i - 1][0]
                y_diff = self.points[i + 1][1] - self.points[i - 1][1]
            self.tangent_points.append(((1 - self.tension) * x_diff, (1 - self.tension) * y_diff))

    def create_curve(self):
        self.curves = []
        if len(self.points) < 2:
            return

        self.create_tangent_points()

        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            t0 = self.tangent_points[i]
            p1 = self.points[i + 1]
            t1 = self.tangent_points[i + 1]

            curve_x, curve_y = self.hermite_curve(p0, p1, t0, t1)
            self.curves.append((curve_x, curve_y))

class InteractivePlot:
    def __init__(self):
        self.generator = CurveGenerator()
        self.path = Path()
        self.path.header.frame_id = "odom"

        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)

        self.ax.set_title('Hermite curve')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.grid(True)

    def update_path(self):
        self.path.poses = []

        for curve_idx, curve in enumerate(self.generator.curves):
            curve_x, curve_y = curve
            for i in range(len(curve_x)):
                x, y = curve_x[i], curve_y[i]
                
                if i < len(curve_x) - 1:
                    x_next, y_next = curve_x[i+1], curve_y[i+1]
                else:
                    if curve_idx < len(self.generator.curves) - 1:
                        next_curve = self.generator.curves[curve_idx + 1]
                        x_next, y_next = next_curve[0][0], next_curve[1][0]
                    else:
                        break

                angle = np.arctan2(y_next - y, x_next - x)
                quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)

                pose = PoseStamped()
                pose.header.frame_id = "odom"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]

                self.path.header.stamp = rospy.Time.now()
                self.path.poses.append(pose)

        path_pub.publish(self.path)

    def plot_curves(self):
        self.generator.create_curve()

        self.ax.cla()  # Clear the axes
        self.ax.grid(True)
        self.ax.scatter(*zip(*self.generator.points), s=100)  # Plot các điểm
        for curve_x, curve_y in self.generator.curves:
            self.ax.plot(curve_x, curve_y, 'r')
        self.fig.canvas.draw()

    def show(self):
        plt.show()

if __name__ == "__main__":
    rospy.init_node('generate_path')
    path_pub = rospy.Publisher('/path', Path, queue_size=10)

    plot = InteractivePlot()
    plot.generator.points = [(0, 0), (20, 20), (60, -20), (100, 40), (120, 30), (140, -10)]

    plot.plot_curves()
    plot.update_path()
    plot.show()
    rospy.spin()
