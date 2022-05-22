#!/usr/bin/env python
import rospy


class RosNode:
    def __init__(self):
        rospy.init_node("ros_node")
        rospy.loginfo("Starting RosNode.")

        pass


if __name__ == "__main__":
    ros_node = RosNode()
    rospy.spin()
