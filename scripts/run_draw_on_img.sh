output='./outputs'
input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_1.png'
name='left'
name='right'

input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_2.png'
name='left_down'

input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_3.png'
name='right_down'
name='right'
name='left'

input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_7.png'
name='left'
name='right'
name='v'

input='c:\\Users\\zhouxia.wang\\Downloads\\images\\rose_320x576.png'
name='left'
name='swing'

input='c:\\Users\\zhouxia.wang\\Downloads\\images\\sunflower_320x576.png'
name='left'
name='swing'
name='swing1'

# input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_4.png'
# name='up'
# name='down'

# input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_5.png'
# name='up'
# # name='down'

# input='c:\\Users\\zhouxia.wang\\Downloads\\images\\image_6.png'
# name='right'
# name='right_cloud'
# name='v'



height=320
width=576

python draw_curve.py \
    --height $height \
    --width $width \
    --output $output \
    --input $input \
    --name $name