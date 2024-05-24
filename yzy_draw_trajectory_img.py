import glob
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
import imageio

# from flow_viz import flow_to_image

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
# from lvdm.data.utils import bivariate_Gaussian

# blur_kernel = bivariate_Gaussian(99, 10, 10, 0, grid=None, isotropic=True)

video_len = 16


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




def get_trajectory_1(points, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    for idx, img_name in enumerate(img_list):
        bg_img = cv2.imread(os.path.join(img_dir, img_name))

        for i in range(idx+1):
            p = points[i]
            if i == 0:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
            if i < idx:
                p1 = points[i+1]
                cv2.line(bg_img, p, p1, (0, 0, 255), 2)
        if idx > 0:
            p = points[idx]
            cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)

def get_trajectory_2(points, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    for idx, img_name in enumerate(img_list):
        bg_img = cv2.imread(os.path.join(img_dir, img_name))
        # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (0, 0, 255), 2)

            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
        
        if idx==(video_len-1):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 2)
        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        
        # break


def get_trajectory_p12v(points, points2, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    for idx, img_name in enumerate(img_list):
        bg_img = cv2.imread(os.path.join(img_dir, img_name))
        bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)
            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
            p = points2[i]
            p1 = points2[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)
            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)

        
        if idx==(video_len-1):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 2)
            cv2.circle(bg_img, points2[-1], 2, (0, 255, 0), 2)

        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        # break

def get_trajectory_p12v_img(points, points2, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    for idx, img_name in enumerate(img_list):
        bg_img = cv2.imread(os.path.join(img_dir, img_name))
        # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)
            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
            p = points2[i]
            p1 = points2[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)
            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)

        
        if idx==(video_len-1):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 2)
            cv2.circle(bg_img, points2[-1], 2, (0, 255, 0), 2)

        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        # break

def get_trajectory_3(points, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    video = []
    video_org = []
    for idx, img_name in enumerate(img_list):
        bg_img = cv2.imread(os.path.join(img_dir, img_name))
        # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        video_org.append(bg_img[:,:,::-1].copy())


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            # p = [int(p[0]*2), int(p[1]*2)]
            # p1 = [int(p1[0]*2), int(p1[1]*2)]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)

            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
        
        p_end = points[-1]
        # p_end = [int(p_end[0]*2), int(p_end[1]*2)]
        if idx==(video_len-1):
            cv2.circle(bg_img, p_end, 2, (0, 255, 0), 2)
        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        video.append(bg_img[:,:,::-1])
        # break
    video = torch.Tensor(video) # 16 x 256 x 256 x 3
    torchvision.io.write_video(f'{save_dir}_img.mp4', video, 10, video_codec='h264', options={'crf': '10'})

    video_org = torch.Tensor(video_org) # 16 x 256 x 256 x 3
    torchvision.io.write_video(f'{save_dir}_img_org.mp4', video_org, 10, video_codec='h264', options={'crf': '10'})

        
def get_trajectory_4(points, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    video = []
    for idx, img_name in enumerate(img_list):
        # bg_img = cv2.imread(os.path.join(img_dir, img_name))
        bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)

            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
        
        if idx==(video_len-1):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 2)
        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        # bg_img = cv2.resize(bg_img, (512, 512))
        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        video.append(bg_img[:,:,::-1])
        # break
    video = torch.Tensor(video) # 16 x 256 x 256 x 3
    torchvision.io.write_video(f'{save_dir}_white.mp4', video, 10, video_codec='h264', options={'crf': '10'})

def get_trajectory_onflow(in_path, points, img_dir, save_dir):
    # white background
    # 256 x 256
    # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    img_list = os.listdir(img_dir)
    img_list.sort()

    frame_list = [np.ones((256, 256, 3), dtype=np.uint8) * 255]

    video_reader = VideoReader(in_path, ctx=cpu(0))
    frame_num = len(video_reader)
    cnt = 0
    while True:
        try:
            frame = video_reader.next().asnumpy()
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_list.append(frame)
            cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break

    video = []
    for idx, img_name in enumerate(img_list):
        # bg_img = cv2.imread(os.path.join(img_dir, img_name))
        bg_img = frame_list[idx]
        # bg_img = np.ones((256, 256, 3), dtype=np.uint8) * 255


        for i in range(video_len-1):
            p = points[i]
            p1 = points[i+1]
            cv2.line(bg_img, p, p1, (255, 0, 0), 2)

            if i == idx:
                cv2.circle(bg_img, p, 2, (0, 255, 0), 2)
        
        if idx==(video_len-1):
            cv2.circle(bg_img, points[-1], 2, (0, 255, 0), 2)
        # p = points[-1]
        # cv2.circle(bg_img, p, 2, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, img_name), bg_img)
        video.append(bg_img[:,:,::-1])
    
    imageio.mimsave(f'{save_dir}_onflow.gif', video, fps=10, loop=0)

    video = torch.Tensor(video) # 16 x 256 x 256 x 3
    torchvision.io.write_video(f'{save_dir}_onflow.mp4', video, 10, video_codec='h264', options={'crf': '10'})
    


# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/shaking_10.txt"

# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/curve_4.txt"
# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/s_curve_3.txt"
# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_4.txt"

# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/shake_1.txt"
# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/s_curve_3.txt"
# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_right_p12v.txt"

# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/curve_3.txt"
# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_4.txt"

# traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/curve_4.txt"

traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/curve_3.txt"

traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_3.txt"
traj_file = "/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_10.txt"


## supp traj
# img_dir = "/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_supp_123/samples/s_curve_3/a_paper_plane_floating_in_the_air__slowly_falling_down/2/images"
# img_dir = "/group/30042/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_curve_7.5/samples/horizon_4/a_horse_running_on_Mars/0/images"
# img_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/trajectory_kernel33/horizon_4/a_horse_running_on_Mars/images"
# img_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/trajectory_kernel11/curve_3/a_girl_skiing/images"
img_dir = "/group/30042/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_skiinggirl_1086/samples/curve_3/skiing__snow/3/images"
img_dir = "/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_supp_1086/samples/horizon_3v/a_horse_running_on_Mars/7/images"
img_dir = '/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_supp_123/samples/horizon_3v/a_horse_running_on_Mars/4/images'

img_dir = '/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0046_1_specific_object_camera_motiontype_trajectory_uc_neg_condT800_7.5_supp_7001/samples/I_1.5x/horizon_10/a_car_running_on_the_concrete_road/7/images'

# supp traj zhouxia
img_dir='tmp/supp_images/chime1/images'
img_dir='tmp/supp_images/sunflower1/images'
traj_file='/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/shake_1.txt'

img_dir='tmp/supp_images/leaf2/images'
img_dir='tmp/supp_images/plane3'
traj_file='/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/s_curve_3.txt'
traj_file='/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_10.txt'

traj_file='/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/curve_1.txt'

traj_file='scripts/trajectories/diagonal_1.txt'
img_dir='/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/diagonal_1/a_hot_air_balloonist_flying_in_the_sky/1/images'

traj_file='scripts/trajectories/diagonal_2.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_2/a_mountain_biker_on_steep_slopes__bird's_view/2/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_2/a_mountain_biker_on_steep_slopes__bird's_view/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_2/a_mountain_biker_on_steep_slopes__bird's_view/1/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/diagonal_2/a_sailor_sailing_on_the_sea/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_2/a_sailor_sailing_on_the_sea__bird's_view/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_2/a_sailor_sailing_on_the_sea__bird's_view/1/images"

traj_file='scripts/trajectories/diagonal_3.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/diagonal_3/a_diver_diving_at_the_bottom_of_the_sea/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/diagonal_3/a_hot_air_balloonist_flying_in_the_sky/2/images"
# img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/diagonal_3/a_kayakers_kayaking_on_the_lake/3/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_3/a_paraglider_flying_in_the_sky__bird's_view/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_3/a_paraglider_flying_in_the_sky__bird's_view/3/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_3/a_sailor_sailing_on_the_sea__bird's_view/2/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_3/a_sailor_sailing_on_the_sea__bird's_view/3/images"

traj_file='scripts/trajectories/diagonal_4.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_4/a_mountain_biker_on_steep_slopes__bird's_view/3/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_4/a_parachutist_jumping_out_of_a_plane__bird's_view/0/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/diagonal_4/a_sailor_sailing_on_the_sea__bird's_view/0/images"

traj_file='scripts/trajectories/horizon_2.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/horizon_2v/a_kayakers_kayaking_on_the_lake/0/images"

traj_file='scripts/trajectories/horizon_10.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/horizon_10/a_paraglider_flying_in_the_sky__bird's_view/1/images"


traj_file='scripts/trajectories/vertical_sun_rise.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/vertical_sun_rise_1/a_paraglider_flying_in_the_sky__bird's_view/3/images"
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo/samples/vertical_sun_rise_1/a_mountain_biker_on_steep_slopes/0/images"

traj_file='scripts/trajectories/vertical_sunset_2.txt'
img_dir="/group/30098/zhouxiawang/outputs/LDVMPose/outputs/motion_0028_3_specific_object_motion_motiontype_trajectory_uc_neg_condT800_7.5_7001_demo_birdsview/samples/vertical_sunset_2/a_parachutist_jumping_out_of_a_plane__bird's_view/2/images"

# img_dir='tmp/supp_images/twocats1/images'
# # img_dir='tmp/supp_images/twocats0/images'
# # img_dir='tmp/supp_images/twocats2/images'
# # img_dir='tmp/supp_images/twozebras0/images'
# # img_dir='tmp/supp_images/twozebras1/images'
# # img_dir='tmp/supp_images/twozebras2/images'

# # supp mix camera and object
# img_dir='tmp/supp_images/horse_in_left/images'
# img_dir='tmp/supp_images/horse_left_left/images'
# img_dir = '/group/30042/yaoweili/share_page/traj/curve_1'

# save_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/paper_vis/type_2/ours"
# save_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/paper_vis/type_2/videocomposer"
# save_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/paper_vis/type_2/traj"
# save_dir = "/group/30098/zhouxiawang/outputs/LDVMPose/paper_draw_traj/"
save_dir = "tmp/demo/"


img_name_list = img_dir.split("/")
# sample_name = img_dir.split("/")[-2]+'_7001'

sample_name = img_dir.split("/")[-3] + '_' + img_dir.split("/")[-2]
# sample_name = img_dir.split("/")[-1]

# save_dir = os.path.join(save_dir, *img_name_list[-6:])
img_save_dir = os.path.join(save_dir, traj_file.split("/")[-1][:-4], sample_name)

# save_dir = "."
# img_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/trajectory_kernel11/curve_3/a_girl_skiing/images"
# save_dir = "/group/30042/ziyangyuan/projects/diffusion/videocomposer/yzy_trajectory_outputs/paper_vis/type_2/videocomposer/images"


# os.makedirs(img_save_dir, exist_ok=True)

# points = read_points(traj_file, reverse=False)
# get_trajectory_3(points, img_dir, img_save_dir)


# white_save_dir = os.path.join(save_dir, traj_file.split("/")[-1][:-4], 'white_traj')
# os.makedirs(white_save_dir, exist_ok=True)

# ## 画condition 白色背景
# get_trajectory_4(points, img_dir, white_save_dir)



# points1 = read_points("/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_right_p1.txt")
# points2 = read_points("/group/30042/zhouxiawang/project/VideoGeneration/scripts/trajectories/horizon_right_p2.txt", True)
# # get_trajectory_p12v(points1, points2, img_dir, white_save_dir)
# get_trajectory_p12v_img(points1, points2, img_dir, img_save_dir)



# print(save_dir)

# points_sets = []
# for i in range(len(traj_files)):
#     points_sets.append(read_points(traj_files[i], reverse[i]))

# get_trajectory(points_sets, f'scripts/trajectories/{name}')
# get_flow(points_sets, f'scripts/trajectories/{name}')

traj_file = "scripts/trajectories/curve_4.txt"
save_dir = "scripts/trajectories/"
in_path = traj_file[:-4]+".mp4"

points = read_points(traj_file, reverse=False)

flow_save_dir = os.path.join(save_dir, traj_file.split("/")[-1][:-4], 'flow')
os.makedirs(flow_save_dir, exist_ok=True)
get_trajectory_onflow(in_path, points, img_dir, flow_save_dir)