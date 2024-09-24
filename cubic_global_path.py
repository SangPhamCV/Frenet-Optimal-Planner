#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from cubic_spline import CubicSpline2D

def generate_target_course(points):
    x, y = zip(*points)
    x, y = list(x), list(y)
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry= [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
    
    combined_array = [[ix, iy] for ix, iy in zip(rx, ry)]

    return combined_array


class MapViewer:
    def __init__(self, pgm_file, yaml_file):
        self.pgm_file = pgm_file
        self.yaml_file = yaml_file
        self.img = self.read_pgm()
        self.params = self.read_yaml()
        # Flip the image along the x-axis
        self.img = np.flipud(self.img)

    def read_pgm(self):
        """Read a PGM file and return the image as a NumPy array."""
        with open(self.pgm_file, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            if header != 'P5':
                raise ValueError("File is not in PGM format")

            # Read until we get width and height, ignoring comment lines
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('#'):
                    continue
                else:
                    width, height = map(int, line.split())
                    break

            maxval = int(f.readline().decode('ascii'))
            dtype = np.uint8 if maxval < 256 else np.uint16
            img = np.fromfile(f, dtype=dtype).reshape((height, width))
        return img

    def read_yaml(self):
        """Read a YAML file and return its contents as a dictionary."""
        with open(self.yaml_file, 'r') as f:
            params = yaml.safe_load(f)
        return params


class InteractivePlot:
    def __init__(self, viewer):
        self.points = []
        self.pathPoints = []

        self.flag = False
 
        # Display a PGM image with parameters from a YAML file.
        resolution = viewer.params.get('resolution', 0.05)  # Default resolution
        origin = viewer.params.get('origin', [0, 0, 0])     # Default origin
        negate = viewer.params.get('negate', 0)             # Default negate value

        if negate:
            viewer.img = 255 - viewer.img

        self.extent = [
            origin[0], origin[0] + viewer.img.shape[1] * resolution,
            origin[1], origin[1] + viewer.img.shape[0] * resolution
        ]

        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)

        self.selected_point_index = None
        self.hit_radius = 0.3  # Radius for detecting points
        self.curve_creation_complete = False

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)

        self.ax.imshow(viewer.img, cmap='gray', origin='lower', extent=self.extent)
        self.ax.set_title('Hermite curve')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.grid(True)
        self.scatter = self.ax.scatter([], [], s=100)  # Increased size of points

    def amcl_callback(self, msg):
        if not self.flag:
            self.points.append((msg.pose.pose.position.x, 
                                            msg.pose.pose.position.y))
            self.flag = True

    def onclick(self, event):
        if event.dblclick:
            self.curve_creation_complete = True
            self.stop_curve_creation()
            return
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.curve_creation_complete:
            # Check if we clicked on an existing point
            for i, (px, py) in enumerate(self.points):
                if np.hypot(px - x, py - y) < self.hit_radius:
                    self.selected_point_index = i
                    return
        else:
            self.points.append((x, y))
            self.update_plot()

    def onrelease(self, event):
        self.selected_point_index = None

    def onmotion(self, event):
        if not self.selected_point_index or event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata

        self.points[self.selected_point_index] = (x, y)
        self.update_plot()

    def publish_path(self, pathPoints):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        # Iterate over each point (x, y) in self.generator.curves
        for curve in pathPoints:
            curve_x, curve_y = curve
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            # Set position based on the curve points
            pose_stamped.pose.position.x = float(curve_x)
            pose_stamped.pose.position.y = float(curve_y)

            # Append the PoseStamped to the path
            path.poses.append(pose_stamped)

        # Publish the entire path
        path_pub.publish(path)


    def update_plot(self):
        if len(self.points) < 2:
            return

        point_msg = Float32MultiArray()
        for point in self.points:
            point_msg.data.extend([point[0], point[1]])
        point_pub.publish(point_msg)

        self.pathPoints = generate_target_course(self.points)
        self.publish_path(self.pathPoints)
        path_x, path_y = zip(*self.pathPoints)
        self.ax.cla()  # Clear the axes
        self.ax.imshow(viewer.img, cmap='gray', origin='lower', extent=self.extent)
        self.ax.grid(True)
        self.ax.scatter(*zip(*self.points), s=100)  # Re-plot points

        self.ax.plot(path_x, path_y, 'r')
        self.fig.canvas.draw()

    def stop_curve_creation(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def show(self):
        plt.show()

def amcl_callback(msg, plot):
    plot.amcl_callback(msg)

if __name__ == "__main__":
    rospy.init_node('generate_path')
    pgm_file = '../maps/tb3_house_map.pgm'
    yaml_file = '../maps/tb3_house_map.yaml'

    viewer = MapViewer(pgm_file, yaml_file)
    plot = InteractivePlot(viewer)

    path_pub = rospy.Publisher('/path', Path, queue_size=10)
    point_pub = rospy.Publisher('/points', Float32MultiArray, queue_size=10)
    amcl_subscriber = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_callback, plot)
    
    plot.show()
    rospy.spin()
