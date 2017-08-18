#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PointStamped, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import Int32
from tf import transformations

def main():

    # ar_tag_frame = rospy.get_param('ar_tag_frame')
    # point_topic = rospy.get_param('point_topic')

    ar_tag_frame = '/april_tag_0'
    published_point_base_topic = '/object_point'
    published_marker_base_topic = '/object_point_marker'
    num_published_points = 4
    point_values = [(0.202322440972, -0.0765477815569, 0.00514242199069), (0.373207820803, -0.0636963110203, 0.0356552587021), (0.204910324165, -0.134972046932, 0.00931300548683), (0.359188795358, -0.142996130769, 0.010604557782)]   # x, y, z

    rospy.init_node('static_point_publisher', anonymous=True)
    num_pub = rospy.Publisher('/object_point_num', Int32, queue_size=10)

    # point publishers
    point_pubs = []
    marker_pubs = []
    points= []
    markers = []
    for i in range(num_published_points):
        point_pub = rospy.Publisher(published_point_base_topic + str(i), PointStamped, queue_size=10)
        marker_pub = rospy.Publisher(published_marker_base_topic + str(i), Marker, queue_size = 10)

        point = PointStamped()
        point.header.frame_id = ar_tag_frame
        point.point.x = point_values[i][0]
        point.point.y = point_values[i][1]
        point.point.z = point_values[i][2]

        marker = Marker()
        marker.header.frame_id = ar_tag_frame
        marker.ns = "/clicked_points/"
        marker.id = i
        marker.type = 2
        marker.action = 0
        marker.pose.position = point.point
        quat = transformations.quaternion_from_euler(0, 0, 0)
        orientation = Quaternion()
        orientation.x = quat[0]
        orientation.y = quat[1]
        orientation.z = quat[2]
        orientation.w = quat[3]
        marker.pose.orientation = orientation
        marker.scale.x = .01
        marker.scale.y = .01
        marker.scale.z = .01
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration.from_sec(1)
        marker.frame_locked = True

        point_pubs.append(point_pub)
        marker_pubs.append(marker_pub)
        points.append(point)
        markers.append(marker)

    seq = 0
    rate = rospy.Rate(10) # 10hz
    while not (rospy.is_shutdown()):
        num_pub.publish(num_published_points)

        for i in range(num_published_points):
            point_pub = point_pubs[i]
            marker_pub = marker_pubs[i]
            point = points[i]
            marker = markers[i]

            point.header.seq = seq
            point.header.stamp = rospy.Time.now()

            marker.header.seq = seq
            marker.header.stamp = rospy.Time.now()

            point_pub.publish(point)
            marker_pub.publish(marker)

            seq += 1

        rate.sleep()
        

if __name__ == "__main__":
    main()
