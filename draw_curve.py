
import os
import sys

import cv2
import numpy as np

import argparse

# define mouse callback function to draw circle
def draw_curve(event, x, y, flags, param):
   global ix, iy, drawing, img
   if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
   elif event == cv2.EVENT_MOUSEMOVE:
      if drawing == True:
         cv2.circle(img, (x, y), 3,(0, 0, 255),-1)
         print(f'True ({x}, {y})')
         fout.write(f'{x}, {y}\n')
      elif event == cv2.EVENT_LBUTTONUP:
         drawing = False
         cv2.circle(img, (x, y), 3,(0, 0, 255),-1)
         print(f'False ({x}, {y})')

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', required=True, type=str, default='traj', help='Name of the output folder')
parser.add_argument('--height', required=False, type=int, default=256, help='Height of the image')
parser.add_argument('--width', required=False, type=int, default=256, help='Width of the image')
parser.add_argument('-i', '--input', required=False, type=str, default=None,  help='Path of input image')
parser.add_argument('-o', '--output', required=False, type=str, default='./outputs', help='Path of output image')

args = parser.parse_args()

height = args.height
width = args.width
name = args.name
input_path = args.input
output_path = args.output


# Create a background image
if input_path is not None and os.path.exists(input_path):
   img = cv2.imread(input_path)
   img = cv2.resize(img, (width, height))
   
   img_name = os.path.basename(input_path).split('.')[0]
   out_root = f'{output_path}/{img_name}_{name}'
   os.makedirs(out_root, exist_ok=True)
   
   output_file = f'{out_root}/trajectory.txt'
   fout = open(output_file, 'w')
   
   output_png = f'{out_root}/image.png'
   
else:
   img = np.zeros((height, width, 3), np.uint8)
   # Create output folder
   out_root = f'{output_path}/{name}'
   os.makedirs(out_root, exist_ok=True)

   output_file = f'{out_root}/trajectory.txt'
   fout = open(output_file, 'w')

   output_png = f'{out_root}/image.png'
   
# Create a window and bind the function to window
cv2.namedWindow("Curve Window")

# Connect the mouse button to our callback function
cv2.setMouseCallback("Curve Window", draw_curve)

# display the window
try:
   while True:
      cv2.imshow("Curve Window", img)
      if cv2.waitKey(10) == 10:
         break
except:
   fout.close()
   cv2.imwrite(f'{output_png}', img)
cv2.destroyAllWindows()

