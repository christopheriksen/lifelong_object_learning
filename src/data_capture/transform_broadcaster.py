#!/usr/bin/env python
import rospy
from tf import TransformListener, TransformBroadcaster
from geometry_msgs.msg import PoseArray

class Node:
    def __init__(self, camera_frame, ar_tag_frame, detections_pose_topic):
        self.trans = None
        self.rot = None
        self.camera_frame = camera_frame
        self.ar_tag_frame = ar_tag_frame
        self.tf = TransformListener()
        rospy.Subscriber(detections_pose_topic, PoseArray, self.pose_array_callback)
        self.tf_broadcaster = TransformBroadcaster()

    def pose_array_callback(self, data):
        try:
            latest_time = self.tf.getLatestCommonTime(self.camera_frame, self.ar_tag_frame)
            (pos, quat) = self.tf.lookupTransform(self.camera_frame, self.ar_tag_frame, latest_time)
            self.trans = pos
            self.rot = quat
        except:
            pass


def main():

    rospy.init_node('transform_broadcaster', anonymous=True)

    # camera_frame = "camera_rgb_optical_frame"
    camera_frame = "head_mount_kinect_rgb_optical_frame"
    ar_tag_frame = "tag_0"
    new_ar_tag_frame = "april_tag_0"
    detections_pose_topic = "tag_detections_pose"

    node = Node(camera_frame, ar_tag_frame, detections_pose_topic)

    rate = rospy.Rate(50) # 50hz
    while not rospy.is_shutdown():
        if (node.trans != None) and (node.rot != None):
            node.tf_broadcaster.sendTransform(node.trans, node.rot, rospy.Time.now(), new_ar_tag_frame, camera_frame)
        rate.sleep()


if __name__ == "__main__":
    main()
