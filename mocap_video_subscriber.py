# Author: Shiyang Lu, 2022
# Python >= 3.6 required, else modify os.makedirs and f-strings
# opencv-python: 4.2.0.34  (This version does not have Qthread issues.)

import os
import json
import shutil
from pprint import pprint
from argparse import ArgumentParser

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rospy
import rosgraph
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class VideoReceiver:

    def __init__(self, bgr_topic, depth_topic, mocap_topic, cam_info_topic, output_dir=None):
        self.bridge = CvBridge()
        bgr_im_sub = Subscriber(bgr_topic, Image)
        depth_im_sub = Subscriber(depth_topic, Image)
        mocap_sub = Subscriber(mocap_topic, PoseStamped)

        subscribers = [bgr_im_sub, depth_im_sub, mocap_sub]
        self.time_synchronizer = ApproximateTimeSynchronizer(subscribers, queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.callback)

        self.bgr_im = None
        self.depth_im = None
        self.record = False
        self.count = 0
        self.output_dir = output_dir

        cam_info_msg = rospy.wait_for_message(cam_info_topic, Float64MultiArray)
        im_h, im_w, depth_scale, fx, fy, cx, cy = cam_info_msg.data
        cam_info = dict()
        cam_info['im_w'] = int(im_w)
        cam_info['im_h'] = int(im_h)
        cam_info['depth_scale'] = depth_scale
        cam_info['cam_intr'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        if output_dir is not None:
            cam_info['id'] = os.path.basename(output_dir)
            cam_config_path = os.path.join(self.output_dir, 'config.json')
            with open(cam_config_path, 'w') as f:
                print(f"Camera info has been saved to: {cam_config_path}.")
                json.dump(cam_info, f, indent=4)

        pprint(cam_info)
        self.cam_info = cam_info

    def callback(self, bgr_msg: Image, depth_msg: Image, pose_stamped_msg: PoseStamped):
        self.bgr_im = self.bridge.imgmsg_to_cv2(bgr_msg, 'bgr8')
        self.depth_im = self.bridge.imgmsg_to_cv2(depth_msg, 'mono16')

        if self.record and self.output_dir is not None:
            stamp = pose_stamped_msg.header.stamp
            position = pose_stamped_msg.pose.position
            orientation = pose_stamped_msg.pose.orientation

            pose = np.eye(4)
            pose[:3, 3] = [position.x, position.y, position.z]
            pose[:3, :3] = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_matrix()

            np.savez(os.path.join(self.output_dir, 'mocap', f"{self.count:04d}-mocap.npz"),
                     time=stamp.secs + stamp.nsecs * 1e-9,
                     pose=pose)
            cv2.imwrite(os.path.join(self.output_dir, 'color', f"{self.count:04d}-color.jpg"), self.bgr_im)
            cv2.imwrite(os.path.join(self.output_dir, 'depth', f"{self.count:04d}-depth.png"), self.depth_im)
            self.count += 1
            if self.count % 100 == 0:
                print(f"{self.count} frames have been saved.")


def main():
    if not rosgraph.is_master_online():
        print("roscore is not running! Either comment out ROS related stuff or run roscore first!")
        exit(0)

    rospy.init_node("video_receiver", anonymous=True)

    parser = ArgumentParser(description='video receiver')
    parser.add_argument('--dataset', default=os.path.expanduser("~/dataset/collected_videos"))
    parser.add_argument('-v', '--video', default="mocap_0001", help='video id')
    parser.add_argument('--bgr_topic', default="/bgr_images")
    parser.add_argument('--depth_topic', default="/depth_images")
    parser.add_argument('--cam_info_topic', default="/cam_info")
    parser.add_argument('--mocap_topic', default="/vrpn_client_node/RigidBody01/pose")
    parser.add_argument('-r', '--rate', default=60, type=int, help='ros rate (ideal fps)')
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    output_dir = None
    if args.save:
        output_dir = os.path.join(args.dataset, args.video)
        if os.path.exists(output_dir):
            key = input(f"{output_dir} has already existed, overwrite? [y/N]")
            if not key or key.upper() != "Y":
                exit(0)
            else:
                shutil.rmtree(output_dir)
        os.makedirs(os.path.join(output_dir, 'color'))
        os.makedirs(os.path.join(output_dir, 'depth'))
        os.makedirs(os.path.join(output_dir, 'mocap'))

    vr = VideoReceiver(
        bgr_topic=args.bgr_topic,
        depth_topic=args.depth_topic,
        mocap_topic=args.mocap_topic,
        cam_info_topic=args.cam_info_topic,
        output_dir=output_dir
    )

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        bgr_im = vr.bgr_im
        depth_im = vr.depth_im

        if bgr_im is not None and depth_im is not None:
            depth_im_vis = np.clip(depth_im.astype(np.float32) / vr.cam_info['depth_scale'] * 100, a_min=0, a_max=255)
            depth_im_vis = np.repeat(depth_im_vis, 3).reshape(bgr_im.shape).astype(np.uint8)
            im_vis = np.hstack((bgr_im, depth_im_vis))

            cv2.imshow('RealSense', im_vis)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord('r'):
                if output_dir is None:
                    print("Output directory is not set. Unable to record.")
                else:
                    vr.record = True
                    print("Start recording...")
            elif key == ord('s') and vr.record:
                vr.record = False
                print("Stop recording.")

        rate.sleep()


if __name__ == '__main__':
    main()
