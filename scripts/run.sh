#! /bin/bash

# draw traj

name='swaying_0'
name='curve_0'
name='horizontal_0'

name='swaying_0'

height=512
width=512
step1_out_path='./outputs'

# step 1: draw traj
# python draw_curve.py \
#     --name $name \
#     --height $height \
#     --width $width \
#     --output $step1_out_path

in_path_0=$step1_out_path'/'$name'/'$name'.txt'
reverse_0=0

step2_out_path=$step1_out_path'/'$name

# step 2: traj to flow video
python syn_traj_with_points.py \
    --name $name \
    --height $height \
    --width $width \
    --output $step2_out_path \
    --input $in_path_0 \
    --reverse $reverse_0
