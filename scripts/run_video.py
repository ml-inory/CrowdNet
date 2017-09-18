# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('/home/rzyang/caffe/build/install/python')
import cv2
import caffe
import argparse
import numpy as np

# parser settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='prototxt/CrowdNet_deploy.prototxt', type=str, help='model definition file, *.prototxt')
parser.add_argument('--weights', default='models/CrowdNet_full_20000.caffemodel', type=str, help='model weights, *.caffemodel')
parser.add_argument('--video', required=True, type=str, help='video file path')
parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')

# parse args
args = parser.parse_args()
model_def = args.model
model_weights = args.weights
video_path = args.video
use_gpu = args.use_gpu

# check existence
def check_file_exists(file_name):
    if not os.path.isfile(file_name):
        print '{} NOT exists.'.format(file_name)
        return False
    else:
        return True

if not (check_file_exists(model_def) and check_file_exists(model_weights) and check_file_exists(video_path)):
    sys.exit(1)

# caffe settings
## 1.set mode
if use_gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
## 2.load net
net = caffe.Net(model_def, model_weights, caffe.TEST)

# accept an image input and output #people and density map
def forward_img(net, img):
    ## 1.image preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 1)
    img = img.astype(np.float32)
    data_c, data_h, data_w = net.blobs['data'].data.shape[1:]
    img = cv2.resize(img, (data_w, data_h))
    if data_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array([img]).transpose(1,2,0)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    output_sum = output['sum'][0]
    output_dmap = net.blobs['dmap'].data[0,0]
    return (output_sum, output_dmap)

def get_heatmap(img):
    if img.max() != img.min():
        img = (img - img.min()) * 255 / (img.max() - img.min())
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print 'Cannot open video {}'.format(video_path)
    sys.exit(-1)

k = 0.004145
b = -7.057812
cv2.ocl.setUseOpenCL(False) # OpenCV bug
fgbg = cv2.createBackgroundSubtractorMOG2() # MOG model
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (640,360))
    fff_mask = np.zeros((360, 640, 3), dtype=np.uint8)
    '''
    # 1.Background subtraction
    fmask = fgbg.apply(frame)
    # 2.erode and dilate
    fmask = cv2.erode(fmask, kernel)
    fmask = cv2.dilate(fmask, kernel)
    # 3.find contour and append rect to fff_mask
    (_,cnts, _) = cv2.findContours(fmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:
            (x,y,w,h) = cv2.boundingRect(c)
            fff_mask[y:y+h, x:x+w, :] = 1
    '''
    fff_mask[100:, 100:500, :] = 1
    frame = frame * fff_mask
    output_sum, output_dmap = forward_img(net, frame)
    output_dmap = get_heatmap(output_dmap)
    output_sum = k * output_sum + b
    output_sum = max(0, output_sum)
    cv2.putText(frame, 'NUM: {}'.format(output_sum), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
#cv2.imshow('frame', frame)
    cv2.imshow('heatmap', frame)
    cv2.waitKey(1)


