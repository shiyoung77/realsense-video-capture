import os
import argparse

import numpy as np
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/collected_videos')
    parser.add_argument("-v", "--video", type=str, default="mocap_0001")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video)
    mocap_files = sorted(os.listdir(os.path.join(video_path, 'mocap')))
    mocap_files = [f for f in mocap_files if f.endswith("npz")]

    coordinate_frames = []
    triangle_mesh = o3d.geometry.TriangleMesh()
    for i, mocap_file in enumerate(mocap_files):
        info = np.load(os.path.join(video_path, 'mocap', mocap_file))
        pose = info['pose']
        timestamp = info['time']
        coord = triangle_mesh.create_coordinate_frame(size=0.05)
        coord.transform(pose)
        coordinate_frames.append(coord)
    o3d.visualization.draw_geometries(coordinate_frames)
