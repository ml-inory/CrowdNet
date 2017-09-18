# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('/home/rzyang/caffe/build/install/python')
import cv2
import caffe
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# matplotlib settings
plt.ioff()
plt.switch_backend('agg')

# parser settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='model definition file, *.prototxt')
parser.add_argument('--weights', required=True, type=str, help='model weights, *.caffemodel')
parser.add_argument('--input_path', required=True, type=str, help='image file path, *.jpg/*.png')
parser.add_argument('--output_path', required=True, type=str, help='image file path, *.jpg/*.png')
parser.add_argument('--gt', type=str, default='./output.jpg', help='output image path, default to ,/output.jpg')
parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')

# check args
if len(sys.argv) < 4:
    parser.print_help()
    sys.exit(1)

# parse args
args = parser.parse_args()
model_def = args.model
model_weights = args.weights
input_path = args.input_path
output_path = args.output_path
gt = args.gt
use_gpu = args.use_gpu

# check existence
def check_file_exists(file_name):
    if not os.path.isfile(file_name):
        print '{} NOT exists.'.format(file_name)
        return False
    else:
        return True

if not (check_file_exists(model_def) and check_file_exists(model_weights)):
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
def forward_img(net, img_path):
    ## 1.image preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 1)
    img = cv2.imread(img_path).astype(np.float32)
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

gt_nums = []
pred_nums = []
with open(gt, 'r') as f:
    for line in f:
        line = line.strip()
        img_name, label = line.split(' ')
        label = int(label)
        img_path = os.path.join(input_path, img_name)
        output_sum, output_dmap = forward_img(net, img_path)
        
	gt_nums.append(label)
        pred_nums.append(output_sum)
	
	
	if output_dmap.max() != output_dmap.min():
    		output_dmap = (output_dmap - output_dmap.min()) * 255 / (output_dmap.max() - output_dmap.min())
	output_dmap = output_dmap.astype(np.uint8)
	output_dmap = cv2.applyColorMap(output_dmap, cv2.COLORMAP_JET)
	cv2.imwrite(os.path.join(output_path, img_name), output_dmap)
        print 'gt_num: {}  pred_num: {}'.format(label, output_sum)
print 'predict done.'

with open('gt_num.txt', 'w') as f:
    f.write(' '.join([str(x) for x in gt_nums]))
print 'save ground truth num to gt_num.txt'

with open('pred_num.txt', 'w') as f:
    f.write(' '.join([str(x) for x in pred_nums]))
print 'save predict num to pred_num.txt'


