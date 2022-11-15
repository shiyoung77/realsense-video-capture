# Author: Shiyang Lu, 2022
# Python >= 3.6 required, else modify os.makedirs and f-strings
# opencv-python: 4.2.0.34  (This version does not have Qthread issues.)

import numpy as np
import pyrealsense2 as rs

import rospy
import rosgraph
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image


class RealSenseCamera:

    def __init__(self, config=None, fps=60):
        if config is not None:
            self.config = config
        else:
            self.config = rs.config()

            # for L515
            # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

            # for D435 and D415
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

        self.bridge = CvBridge()

        # ros publisher
        self.bgr_topic = "bgr_images"
        self.depth_topic = "depth_images"
        self.cam_info_topic = "cam_info"
        self.color_im_pub = rospy.Publisher(self.bgr_topic, Image, queue_size=10)
        self.depth_im_pub = rospy.Publisher(self.depth_topic, Image, queue_size=10)
        self.cam_info_pub = rospy.Publisher(self.cam_info_topic, Float64MultiArray, queue_size=10)
        self.ros_rate = rospy.Rate(fps)

    def run(self):
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)

        # depth align to color
        align = rs.align(rs.stream.color)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = int(round(1 / depth_sensor.get_depth_scale()))

        print(f"color_intrinsics:\n", color_intrinsics)
        im_w = color_intrinsics.width
        im_h = color_intrinsics.height
        fx, fy = color_intrinsics.fx, color_intrinsics.fy
        cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
        cam_info_msg = Float64MultiArray()
        cam_info_msg.data = [im_w, im_h, depth_scale, fx, fy, cx, cy]
        print(f"{self.bgr_topic = }, {self.depth_topic = }, {self.cam_info_topic = }")
        print("Start publishing ... (ctrl-c to stop)")

        try:
            while not rospy.is_shutdown():
                frames = pipeline.wait_for_frames()
                timestamp = rospy.get_rostime()

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                bgr_im = np.asanyarray(color_frame.get_data())
                depth_im = np.asanyarray(depth_frame.get_data())

                # ROS publisher
                color_msg = self.bridge.cv2_to_imgmsg(bgr_im, 'bgr8')
                depth_msg = self.bridge.cv2_to_imgmsg(depth_im, 'mono16')
                color_msg.header.stamp = timestamp
                depth_msg.header.stamp = timestamp

                self.color_im_pub.publish(color_msg)
                self.depth_im_pub.publish(depth_msg)
                self.cam_info_pub.publish(cam_info_msg)

                self.ros_rate.sleep()
        finally:
            pipeline.stop()


def main():
    if not rosgraph.is_master_online():
        print("roscore is not running! Either comment out ROS related stuff or run roscore first!")
        exit(0)

    rospy.init_node("realsense", anonymous=True)
    camera = RealSenseCamera(fps=60)
    camera.run()


if __name__ == '__main__':
    main()
