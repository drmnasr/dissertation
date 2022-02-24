#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys

from segnet_utils import *
from helpers import *
from dwa_p import *

import cv2
from djitellopy import Tello
import time

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# create video output
#output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# create buffer manager
buffers = segmentationBuffers(net, opt)

#Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
tello.move_up(100)
frame_read = tello.get_frame_read()

#Initial Navigation state
pose = (467, 660, -90.0)
velocity = (0.0, 0.0)
goal = (400, 400)
angle = 0

mtxs = np.load('./warp.npy')
mtxs_inv = np.linalg.inv(mtxs)

# process frames
while True:

	img_input = frame_read.frame
	if img_input is not None:

		# Convert input to Cuda
		img_input = jetson.utils.cudaFromNumpy(img_input)

		# allocate buffers for this size image
		buffers.Alloc(img_input.shape, img_input.format)

		# process the segmentation network
		net.Process(img_input, ignore_class=opt.ignore_class)

		# Get the class masks
		class_mask = jetson.utils.cudaAllocMapped(width=960, height=720, format="gray8")
		class_mask_np = jetson.utils.cudaToNumpy(class_mask)
		net.Mask(class_mask, 960, 720)

		original = jetson.utils.cudaToNumpy(img_input)
		class_mask_np = class_mask_np[:, :, 0]

		# Color mask with floor only and segmented with all classes to be shown later
		mask = colorize(class_mask_np, get_label_colors(driveable = 2))
		segmented = colorize(class_mask_np, get_label_colors(driveable = -1))
		warped, warp_center, original_center = warp(mask)

		# Convert to Gray
		gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

		# Get Obstacles
		obstacles, hierarchy = cv2.findContours(gray,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		# Run DWA
		pose_, velocity, goal, angle_ = navigate(pose, velocity, goal, obstacles, warped, angle)

		commandX = (pose[1] - pose_[1]) * 10
		commandY = (pose[0] - pose_[0]) * 10

		# According to calibration, 1 pixel = 0.33 cm
		# Send new command too tello, adjust for scale cm/pixel in comparison to real coordinates from calibration
		tello.go_xyz_speed(commandX * 0.33, commandY * 0.33, 0, 15)

		pose = pose_
		angle = pose[2]

		# Draw Control point and Arbitrary Goal, so we can see
		cv2.circle(warped, (round(goal[0]), round(goal[1])), 15, (0, 255, 255), cv2.FILLED)
		cv2.circle(warped, (round(pose[0]), round(pose[1])), 15, (0, 255, 0), cv2.FILLED)

		cv2.imshow("output", warped)
		cv2.waitKey(1)

		# Syncronize
		jetson.utils.cudaDeviceSynchronize()
		
