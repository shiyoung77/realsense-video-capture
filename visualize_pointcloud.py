import os
import json
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset = os.path.expanduser('~/dataset/collected_videos/0001')
with open(os.path.join(dataset, 'config.json'), 'r') as f:
    cam_info = json.load(f)

prefix = 0
color_path = os.path.join(dataset, 'color', f'{prefix:04d}-color.jpg')
depth_path = os.path.join(dataset, 'depth', f'{prefix:04d}-depth.png')

rgb_im = cv2.cvtColor(cv2.imread(color_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cam_info['depth_scale']

def get_color(event):
    if event.xdata and event.ydata:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        print(f"{x = }, {y = }, rgb: {rgb_im[y, x]}, hsv: {hsv_im[y, x]}")


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].imshow(rgb_im)
ax[1].imshow(depth_im)
# event_id_1 = fig.canvas.mpl_connect('button_press_event', get_color)
event_id_2 = fig.canvas.mpl_connect('motion_notify_event', get_color)
plt.show()

depth_im_o3d = o3d.geometry.Image(depth_im)
color_im_o3d = o3d.geometry.Image(rgb_im)

cam_intr_o3d = o3d.camera.PinholeCameraIntrinsic()
cam_intr_o3d.intrinsic_matrix = np.array(cam_info['cam_intr'])
rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im_o3d, depth_im_o3d, depth_scale=1,
                                                              convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr_o3d)
o3d.visualization.draw_geometries_with_editing([pcd])

# show camera frame
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, mesh_frame])
