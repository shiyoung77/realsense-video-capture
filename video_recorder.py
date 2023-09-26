# Author: Shiyang Lu, 2021

import os
import json
import shutil
from argparse import ArgumentParser

import pyrealsense2 as rs
import numpy as np
import cv2


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

            # for D455
            # self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
            # self.config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, fps)

        self.record = False

    def run(self, output_dir=None):
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)

        # depth align to color
        align = rs.align(rs.stream.color)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = int(round(1 / depth_sensor.get_depth_scale()))

        print(color_intrinsics)
        cam_info = dict()
        cam_info['im_w'] = color_intrinsics.width
        cam_info['im_h'] = color_intrinsics.height
        cam_info['depth_scale'] = depth_scale
        fx, fy = color_intrinsics.fx, color_intrinsics.fy
        cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
        cam_info['cam_intr'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        if output_dir is not None:
            cam_info['id'] = os.path.basename(output_dir)
            cam_config_path = os.path.join(output_dir, 'config.json')
            with open(cam_config_path, 'w') as f:
                print(f"Camera info has been saved to: {cam_config_path}.")
                json.dump(cam_info, f, indent=4)

        depth_vis_scale = 100

        def mouse_callback(event, x, y, flags, params):
            im_h, im_w, _ = im_vis.shape
            H, W = im_h, im_w // 2
            if event == cv2.EVENT_LBUTTONDOWN:
                r, g, b = im_vis[y, x]
                if x < W:
                    print(f"({x}, {y}), rgb: ({r}, {g}, {b})")
                else:
                    if r == 255:
                        print(f"({x}, {y}), depth: >2.55 m")
                    else:
                        d = r / depth_vis_scale
                        print(f"({x}, {y}), depth: {d}m")

        count = 0
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                bgr_im = np.asanyarray(color_frame.get_data())
                depth_im = np.asanyarray(depth_frame.get_data())
                im_h, im_w = depth_im.shape

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_im_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_im, alpha=0.03), cv2.COLORMAP_JET)

                depth_im_vis = np.clip(depth_im.astype(np.float32) / depth_scale * depth_vis_scale, a_min=0, a_max=255)
                depth_im_vis = np.repeat(depth_im_vis, 3).reshape((im_h, im_w, 3)).astype(np.uint8)

                # Stack both images horizontally
                im_vis = np.hstack((bgr_im, depth_im_vis))

                # Show images
                cv2.imshow('RealSense', im_vis)
                cv2.setMouseCallback("RealSense", mouse_callback)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and output_dir is not None:
                    self.record = True
                    print("Start recording...")
                elif key == ord('s') and self.record:
                    self.record = False
                    print("Stop recording.")

                if self.record:
                    cv2.imwrite(os.path.join(output_dir, 'color', f"{count:04d}-color.jpg"), bgr_im)
                    cv2.imwrite(os.path.join(output_dir, 'depth', f"{count:04d}-depth.png"), depth_im)
                    count += 1
                    if count % 100 == 0:
                        print(f"{count} frames have been saved.")
        finally:
            pipeline.stop()

        cv2.destroyAllWindows()


def main():
    parser = ArgumentParser(description='video recorder')
    parser.add_argument('--dataset', default=os.path.expanduser("~/dataset/collected_videos"))
    parser.add_argument('-v', '--video', default="0001", help='video name')
    parser.add_argument('-r', '--rate', default=60, type=int, help='ideal fps')
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

    camera = RealSenseCamera(fps=args.rate)
    camera.run(output_dir=output_dir)


if __name__ == '__main__':
    main()
