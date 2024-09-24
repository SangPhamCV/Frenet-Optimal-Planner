#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path

# Initialize the global variable to None
path_pub = None

def path_callback(msg):
    global path_pub
    path_pub = msg

if __name__ == '__main__':
    rospy.init_node('path_subscriber_node', anonymous=True)
    rospy.Subscriber('/path', Path, path_callback)
    pub = rospy.Publisher('/path_request', Path, queue_size=10)

    array = []
    user_input = input("Do you want tracking? Press 'y' to run: ")
    if user_input == "y":
        # Check if path_pub has been updated by the callback
        rospy.sleep(1)  # Allow some time for the callback to receive data
        if path_pub is not None:
            pub.publish(path_pub)
            # for pose in path_pub.poses:
            #     array.append((round(pose.pose.position.x, 3), round(pose.pose.position.y, 3)))
            # # Save the array to a text file
            # with open('path_positions.txt', 'w') as file:
            #     for position in array:
            #         file.write(f"{position[0]} {position[1]}\n")
        else:
            print("Path data not received yet.")
    rospy.spin()
