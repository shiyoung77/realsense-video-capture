# realsense-video-capture

This repo stores some useful scripts to capture RGB-D videos using Intel RealSense cameras.

## Dependencies
- pyrealsense2 (pip install pyrealsense2)
- opencv-python (pip install opencv-python)
- open3d (pip install open3d) for visualizing point cloud

## Usage
First specify the realsense camera version in the `video_recorder.py` file. \

Run `python video_recorder.py --dataset {dataset_path} --video {video_name} --rate {fps} --save` \
A window will pop up and show the RGB-D video.
Press `r` to start recording, `s` to pause/stop, and `q` to end recording. \
The RGB-D video will be saved in the specified dataset path, i.e. `{dataset_path}/{video_name}`.

ROS publisher / subscriber scripts for recording RGB-D videos are also provided. \

The recorded RGB-D videos / point cloud can be visualized with `visualize_pointcloud.py` \
