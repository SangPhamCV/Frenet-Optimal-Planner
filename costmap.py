#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import numpy as np

class CostmapSubscriber:
    def __init__(self):
        rospy.init_node('costmap_subscriber_node', anonymous=True)
        self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.costmap_callback)
        self.costmap_data = None
        rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        rospy.spin()

    def costmap_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        self.costmap_data = np.array(msg.data).reshape((height, width))

    def timer_callback(self, event):
        if self.costmap_data is not None:
            self.plot_costmap()

    def plot_costmap(self):
        plt.imshow(self.costmap_data, cmap='gray', origin='lower')
        plt.title('Local Costmap')
        plt.colorbar()
        plt.savefig('/home/sangpham/catkin_ws/src/mascot/costmap_image.png')  # Save the image
        plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    try:
        CostmapSubscriber()
    except rospy.ROSInterruptException:
        pass
