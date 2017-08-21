#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PointStamped

class Node:
    def __init__(self, num_published_points, published_point_base_topic):
        self.num_published_points = num_published_points
        self.published_point_base_topic = published_point_base_topic
        for i in range(num_published_points):
            rospy.Subscriber(self.published_point_base_topic + str(i), PointStamped, self.point_published_callback, i)

    def point_published_callback(self, data, point_id):
        rospy.loginfo("point id: " + str(point_id) + " x: " + str(data.point.x) + " y: " + str(data.point.y) + " z: " + str(data.point.z) + " frame id: " + data.header.frame_id)



def main():
    rospy.init_node('point_listener', anonymous=True)

    num_published_points = 4
    published_point_base_topic = "/object_point"

    node = Node(num_published_points, published_point_base_topic)

    rospy.spin()




if __name__ == "__main__":
    main()
