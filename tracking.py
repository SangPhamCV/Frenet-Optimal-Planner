#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion

K_LINEAR = [1.5, 0.9, 0]  # P, I, D
K_ANGULAR = [1, 0.2, 0]  # P, I, D

MAXIMUM_VLINEAR =0.8
MAXIMUM_VANGULAR = 0.8  

robot_pose = [0.0, 0.0, 0.0]  # [x, y, theta]
path_poses = []

# Callback function to get robot position and orientation from the odom topic
def odom_callback(msg):
    global robot_pose
    robot_pose[0] = msg.pose.pose.position.x
    robot_pose[1] = msg.pose.pose.position.y
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    _, _, yaw = euler_from_quaternion(orientation_list)
    robot_pose[2] = yaw

# Path callback function
def path_callback(msg):
    global path_poses
    path_poses = []
    for pose in msg.poses:
        p = [0.0, 0.0, 0.0]  # [x, y, theta]
        p[0] = pose.pose.position.x
        p[1] = pose.pose.position.y
        orientation_q = pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        p[2] = yaw
        path_poses.append(p)

# Function to calculate the Euclidean distance between two points
def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Main function
def main():
    global path_poses
    prev_error_linear = 0.0
    prev_error_angular = 0.0

    rate = rospy.Rate(10)  # 10 Hz

    user_input = input("Do you want tracking? Press 'y' to run: ")
    if user_input == "y":
        while not rospy.is_shutdown():
            if path_poses:
                qd = path_poses[0]  # Current desired pose
                # Calculate tracking errors
                error_linear = get_distance(robot_pose[0], robot_pose[1], qd[0], qd[1])
                error_angular = np.arctan2(qd[1] - robot_pose[1], qd[0] - robot_pose[0]) - robot_pose[2]
                error_angular = np.arctan2(np.sin(error_angular), np.cos(error_angular))

                # PID control
                p_linear = K_LINEAR[0] * error_linear
                i_linear = K_LINEAR[1] * ((error_linear + prev_error_linear) * 0.1)
                d_linear = K_LINEAR[2] * ((error_linear - prev_error_linear) / 0.1)

                p_angular = K_ANGULAR[0] * error_angular
                i_angular = K_ANGULAR[1] * ((error_angular + prev_error_angular) * 0.1)
                d_angular = K_ANGULAR[2] * ((error_angular - prev_error_angular) / 0.1)

                instant_vlinear = p_linear + i_linear + d_linear
                instant_vangular = p_angular + i_angular + d_angular

                # Limit control inputs
                if instant_vlinear > MAXIMUM_VLINEAR:
                    instant_vlinear = MAXIMUM_VLINEAR
                if instant_vangular > MAXIMUM_VANGULAR:
                    instant_vangular = MAXIMUM_VANGULAR

                # Publish the velocities to the robot
                vel_msg = Twist()
                vel_msg.linear.x = instant_vlinear
                vel_msg.angular.z = instant_vangular
                robot_vel_pub.publish(vel_msg)

                prev_error_linear = error_linear
                prev_error_angular = error_angular

                # Remove the first element from the list if the robot is close enough to the current target
                if error_linear < 0.1:
                    path_poses.pop(0)

            if not path_poses:
                print("stop")
                vel_msg = Twist()
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                robot_vel_pub.publish(vel_msg)
                break

            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('pid_trajectory_tracking_node', anonymous=True)
        robot_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/path", Path, path_callback)
        rospy.Subscriber("/odom", Odometry, odom_callback)
        main()
    except rospy.ROSInterruptException:
        pass
