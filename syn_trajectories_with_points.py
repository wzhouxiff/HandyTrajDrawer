import glob
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from flow_viz import flow_to_image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils import bivariate_Gaussian

blur_kernel = bivariate_Gaussian(99, 10, 10, 0, grid=None, isotropic=True)

video_len = 16

def get_trajectory(points_set, filename):
    # white background
    # 256 x 256
    bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    for points in points_set:
        for i in range(video_len-1):
            p = points[i]
            if i == 0:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (0, 0, 255), 2)

        p = points[-1]
        cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

    cv2.imwrite(f'{filename}_draw.png', bg_img)

def get_flow(points_set, filename):
    optical_flow = np.zeros((video_len, 256, 256, 2), dtype=np.float32)
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
    for i in range(1, video_len):
        flow_img = flow_to_image(optical_flow[i])
        flow_img = torch.Tensor(flow_img) # 256 x 256 x 3
        video.append(flow_img)
    video = torch.stack(video, dim=0) # 15 x 256 x 256 x 3
    torchvision.io.write_video(f'{filename}.mp4', video, 10, video_codec='h264', options={'crf': '10'})

    return optical_flow

def read_points(file, reverse=False):
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

# single trajectory
# traj_files = glob.glob('scripts/trajectories/*.txt')
# for traj_file in traj_files:
#     with open(traj_file, 'r') as f:
#         lines = f.readlines()
#     points = []
#     for line in lines:
#         x, y = line.strip().split(',')
#         points.append((int(x), int(y)))
#     filename = traj_file.replace('.txt', '')
#     if len(points) > video_len:
#         skip = len(points) // video_len
#         points = points[::skip]
#     points = points[:video_len]
#     assert len(points) == video_len, f'{len(points)} != {video_len}'
#     get_trajectory(points, filename)
#     get_flow(points, filename)

# multiple trajectories
horizon_1 = 'scripts/trajectories/horizon_1.txt'
horizon_2 = 'scripts/trajectories/horizon_2.txt'
horizon_3 = 'scripts/trajectories/horizon_3.txt'
horizon_4 = 'scripts/trajectories/horizon_4.txt'
horizon_5 = 'scripts/trajectories/horizon_5.txt'
horizon_6 = 'scripts/trajectories/horizon_6.txt'

vertical_1 = 'scripts/trajectories/verticle_1.txt'
vertical_2 = 'scripts/trajectories/verticle_2.txt'
vertical_3 = 'scripts/trajectories/verticle_3.txt'

curve_1 = 'scripts/trajectories/curve_1.txt'
curve_2 = 'scripts/trajectories/curve_2.txt'
curve_3 = 'scripts/trajectories/curve_3.txt'
curve_4 = 'scripts/trajectories/curve_4.txt'

turn_left = 'scripts/trajectories/turn_left.txt'
turn_left_1 = 'scripts/trajectories/turn_left_1.txt'
turn_right = 'scripts/trajectories/turn_right.txt'
turn_right_1 = 'scripts/trajectories/turn_right_1.txt'

shake_1 = 'scripts/trajectories/shake_1.txt'
shake_2 = 'scripts/trajectories/shake_2.txt'
swing_1 = 'scripts/trajectories/swing_1.txt'
swing_2 = 'scripts/trajectories/swing_2.txt'

diagonal_1 = 'scripts/trajectories/diagonal_1.txt'
diagonal_2 = 'scripts/trajectories/diagonal_2.txt'
diagonal_3 = 'scripts/trajectories/diagonal_3.txt'
diagonal_4 = 'scripts/trajectories/diagonal_4.txt'

vertical_sun_rise = 'scripts/trajectories/vertical_sun_rise.txt'
vertical_sun_rise_1 = 'scripts/trajectories/vertical_sun_rise_1.txt'
vertical_sun_rise_2 = 'scripts/trajectories/vertical_sun_rise_2.txt'

vertical_sunset_1 = 'scripts/trajectories/vertical_sunset_1.txt'
vertical_sunset_2 = 'scripts/trajectories/vertical_sunset_2.txt'

horizon_right_p1 = 'scripts/trajectories/horizon_right_p1.txt'
horizon_right_p2 = 'scripts/trajectories/horizon_right_p2.txt'

horizon_mix_p1 = 'scripts/trajectories/horizon_mix_p1.txt'
horizon_mix_p2 = 'scripts/trajectories/horizon_mix_p2.txt'
horizon_mix_p3 = 'scripts/trajectories/horizon_mix_p3.txt'
horizon_mix_p4 = 'scripts/trajectories/horizon_mix_p4.txt'
horizon_mix_p5 = 'scripts/trajectories/horizon_mix_p5.txt'

horizon_mix_v1_p1 = 'scripts/trajectories/horizon_mix_v1_p1.txt'
horizon_mix_v1_p2 = 'scripts/trajectories/horizon_mix_v1_p2.txt'
horizon_mix_v1_p3 = 'scripts/trajectories/horizon_mix_v1_p3.txt'
horizon_mix_v1_p4 = 'scripts/trajectories/horizon_mix_v1_p4.txt'
horizon_mix_v1_p5 = 'scripts/trajectories/horizon_mix_v1_p5.txt'
horizon_mix_v1_p6 = 'scripts/trajectories/horizon_mix_v1_p6.txt'

traj_files = [horizon_1, horizon_3]
reverse = [False, False]
name = 'horizon13'

traj_files = [horizon_1, horizon_6]
reverse = [False, True]
name = 'horizon16v'

traj_files = [horizon_1, horizon_2, horizon_3]
reverse = [False, False, False]
name = 'horizon123'

traj_files = [vertical_1, vertical_2, vertical_3]
reverse = [False, False, False]
name = 'vertical123'

traj_files = [vertical_1, vertical_3]
reverse = [False, True]
name = 'vertical13v'

traj_files = [turn_left]
reverse = [False]
name = 'turn_left'

traj_files = [turn_left_1]
reverse = [False]
name = 'turn_left_1'

traj_files = [turn_right]
reverse = [False]
name = 'turn_right'

# traj_files = [turn_right_1]
# reverse = [False]
# name = 'turn_right_1'

traj_files = [shake_1]
reverse = [False]
name = 'shake_1'

traj_files = [shake_2]
reverse = [False]
name = 'shake_2'

traj_files = [swing_1]
reverse = [False]
name = 'swing_1'

traj_files = [swing_2]
reverse = [False]
name = 'swing_2'

traj_files = [diagonal_1]
reverse = [False]
name = 'diagonal_1'

traj_files = [diagonal_2]
reverse = [False]
name = 'diagonal_2'

traj_files = [diagonal_3]
reverse = [False]
name = 'diagonal_3'

traj_files = [diagonal_4]
reverse = [False]
name = 'diagonal_4'

traj_files = [vertical_sun_rise]
reverse = [False]
name = 'vertical_sun_rise'

traj_files = [vertical_sun_rise_1]
reverse = [False]
name = 'vertical_sun_rise_1'

traj_files = [vertical_sun_rise_2]
reverse = [False]
name = 'vertical_sun_rise_2'

traj_files = [vertical_sunset_1]
reverse = [False]
name = 'vertical_sunset_1'

traj_files = [vertical_sunset_2]
reverse = [False]
name = 'vertical_sunset_2'

traj_files = [horizon_1]
reverse = [True]
name = 'horizon_1v'

traj_files = [horizon_2]
reverse = [True]
name = 'horizon_2v'

traj_files = [horizon_3]
reverse = [True]
name = 'horizon_3v'

traj_files = [horizon_right_p1, horizon_right_p2]
reverse = [False, False]
name = 'horizon_right_p12'

traj_files = [horizon_right_p1, horizon_right_p2]
reverse = [False, True]
name = 'horizon_right_p12v'

traj_files = [horizon_mix_p1, horizon_mix_p2, horizon_mix_p3, horizon_mix_p4, horizon_mix_p5]
reverse = [False, False, False, False, False]
name = 'horizon_mix'

traj_files = [horizon_mix_v1_p1, horizon_mix_v1_p2, horizon_mix_v1_p3, horizon_mix_v1_p4, horizon_mix_v1_p5, horizon_mix_v1_p6]
reverse = [False, False, False, False, False, False]
name = 'horizon_mix_v1'

z_curve_1 = 'scripts/trajectories/z_curve_1.txt'
z_curve_2 = 'scripts/trajectories/z_curve_2.txt'
z_curve_3 = 'scripts/trajectories/z_curve_3.txt'
spiral_curve_1 = 'scripts/trajectories/spiral_curve_1.txt'
spiral_curve_2 = 'scripts/trajectories/spiral_curve_2.txt'
spiral_curve_3 = 'scripts/trajectories/spiral_curve_3.txt'
circle_curve_1 = 'scripts/trajectories/circle_curve_1.txt'
circle_curve_2 = 'scripts/trajectories/circle_curve_2.txt'

traj_files = [z_curve_1, z_curve_2, z_curve_3, 
              spiral_curve_1, spiral_curve_2, spiral_curve_3,
              circle_curve_1, circle_curve_2]
reverse = [False, False, False,
              False, False, False,
                False, False]
name = ['z_curve_1', 'z_curve_2', 'z_curve_3',
        'spiral_curve_1', 'spiral_curve_2', 'spiral_curve_3',
        'circle_curve_1', 'circle_curve_2']

shaking_10 = 'scripts/trajectories/shaking_10.txt'

traj_files = [shaking_10]
reverse = [False]
name = ['shaking_10']

horizon_10 = 'scripts/trajectories/horizon_10.txt'
traj_files = [horizon_10]
reverse = [False]
name = ['horizon_10']


s_curve_1 = 'scripts/trajectories/s_curve_1.txt'
s_curve_2 = 'scripts/trajectories/s_curve_2.txt'
s_curve_3 = 'scripts/trajectories/s_curve_3.txt'
s_curve_4 = 'scripts/trajectories/s_curve_4.txt'

name = ['s_curve_1', 's_curve_2', 's_curve_3', 's_curve_4']
traj_files = [s_curve_1, s_curve_2, s_curve_3, s_curve_4]
reverse = [False, False, False, False]

z_curve_5 = 'scripts/trajectories/z_curve_5.txt'
name = ['z_curve_5']
traj_files = [z_curve_5]
reverse = [False]

for i in range(len(traj_files)):
    points = read_points(traj_files[i], reverse[i])
    get_trajectory([points], f'scripts/trajectories/{name[i]}')
    get_flow([points], f'scripts/trajectories/{name[i]}')

# points_sets = []
# for i in range(len(traj_files)):
#     points_sets.append(read_points(traj_files[i], reverse[i]))

# get_trajectory(points_sets, f'scripts/trajectories/{name}')
# get_flow(points_sets, f'scripts/trajectories/{name}')