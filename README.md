# HandyTrajDrawer

This repo aims to customize moving trajectories in a video.

## :wrench: Dependencies

    pip install -r requirements.txt

## :runner: Running

- Step 1: Draw trajectories:
    
    - Draw on a black canvas:

        ```
        python draw_curve.py --name traj_0 --height 256 --width 256 --output ./outputs

        python draw_curve.py --name traj_1 --height 256 --width 256 --output ./outputs
        ```` 

    - Draw on an image:

        ```
        python draw_curve.py --name rose --height 256 --width 256 --output ./outputs --input examples/rose.png
        ```

- Step 2: Trajectory to moving points/optical flow in a video

    - A single trajectory:

        ```
        python syn_traj_with_points.py \
        --inputs outputs/traj_0/ \
        --reverse 0 \
        --height 256 --width 256 --video_len 16 \
        --output ./outputs \
        --name traj_0 
        ```

    - A single trajectory (reverse):

        ```
        python syn_traj_with_points.py \
        --inputs outputs/traj_0/ \
        --reverse 1 \
        --height 256 --width 256 --video_len 16 \
        --output ./outputs \
        --name traj_0 
        ```

    - Multiple trajectories:

        ```
        python syn_traj_with_points.py \
        --inputs outputs/traj_0/ outputs/traj_1/ \
        --reverse 0 0 \
        --height 256 --width 256 --video_len 16 \
        --output ./outputs \
        --name traj_0 
        ```

## :e-mail: Contact

If you have any question, open an issue or email `wzhoux@connect.hku.hk`.