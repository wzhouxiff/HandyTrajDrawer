import os
import argparse
import imageio

import cv2
import numpy as np
import torch
from utils.flow_viz import flow_to_image

from utils.utils import bivariate_Gaussian

def get_trajectory(points_set, bg_img=None):
    # white background
    global height, width, video_len
    if bg_img is None:
        bg_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for points in points_set:
        for i in range(video_len-1):
            p = points[i]
            if i == 0:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (0, 0, 255), 2)

        p = points[-1]
        cv2.circle(bg_img, p, 2, (255, 0, 0), 2)
        
    return bg_img

def get_flow(points_set, filename):
    global height, width, video_len
    optical_flow = np.zeros((video_len, height, width, 2), dtype=np.float32)
    for points in points_set:
        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            optical_flow[i+1, p[1], p[0], 0] = p1[0] - p[0]
            optical_flow[i+1, p[1], p[0], 1] = p1[1] - p[1]
        for i in range(1, video_len):
            optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)

    np.save(f'{filename}.npy', optical_flow)

    video = []
    for i in range(0, video_len):
        flow_img = flow_to_image(optical_flow[i])
        flow_img = torch.Tensor(flow_img) # 256 x 256 x 3
        flow_img = flow_img.numpy().astype(np.uint8)
        video.append(flow_img)

    return video

def read_points(file, reverse=False):
    global video_len
    with open(file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(',')
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]
    
    return points

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', required=True, type=str, default='traj', help='Name of the output folder')
parser.add_argument('--height', required=False, type=int, default=256, help='Height of the image')
parser.add_argument('--width', required=False, type=int, default=256, help='Width of the image')
parser.add_argument('-i', '--inputs', required=True, nargs='+', type=str, help='Path of input folder')
parser.add_argument('--reverse', required=True, nargs='+', type=int, help='Reverse the order of the points, 0 or 1')
parser.add_argument('-o', '--output', required=True, type=str, default='./outputs', help='Path of output image')
parser.add_argument('--video_len', required=False, type=int, default=16, help='Length of the video')
parser.add_argument('--kernel_size', required=False, type=int, default=99, help='Size of the kernel')
parser.add_argument('--sigma_x', required=False, type=float, default=10, help='Sigma x')
parser.add_argument('--sigma_y', required=False, type=float, default=10, help='Sigma y')


args = parser.parse_args()

height = args.height
width = args.width

video_len = args.video_len

name = args.name
output_path = os.path.join(args.output, name)
os.makedirs(output_path, exist_ok=True)

kernel_size = args.kernel_size
sigma_x = args.sigma_x
sigma_y = args.sigma_y
blur_kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, 0, grid=None, isotropic=True)

traj_files = args.inputs
reverse = args.reverse
print(traj_files)
print(reverse)

points_sets = []
output_name = ''
for i in range(len(traj_files)):
    traj_files[i] = traj_files[i][:-1] if traj_files[i][-1] == '/' else traj_files[i]
    points_sets.append(read_points(f'{traj_files[i]}/trajectory.txt', reverse[i]))
    output_name += traj_files[i].split('/')[-1]
    if reverse[i]:
        output_name += '_rev'
    output_name += '_'
output_name = output_name[:-1]

opt_video = get_flow(points_sets, f'{output_path}/{output_name}')

vis_video = []
for i in range(len(opt_video)):
    vis_video.append(get_trajectory(points_sets, opt_video[i]))

imageio.mimsave(f'{output_path}/{output_name}.gif', vis_video, fps=10, loop=0)
