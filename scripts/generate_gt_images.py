# -*- coding: utf-8 -*-
# function: 生成ground-truth图片
# Coded by 杨荣钊 on 2017/07/11

import os, sys
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py

def get_img_names(part, phase):
    img_root_path = os.path.join('part_' + part, phase + '_data', 'images')
    return [os.path.join(img_root_path, x) for x in os.listdir(img_root_path)]

def get_gt_names(part, phase):
    gt_root_path = os.path.join('part_' + part, phase + '_data', 'ground-truth')
    return [os.path.join(gt_root_path, x) for x in os.listdir(gt_root_path)]

def parse_mat(mat_name):
    mat = sio.loadmat(mat_name)
    location = mat['image_info'][0][0][0][0][0]
    num_people = mat['image_info'][0][0][0][0][1]
    return location, num_people[0][0]

def matlab_style_gauss2D(shape=(25,25), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def knn_distance(pt, pts, k=3):
    pt = np.array([pt])
    pts = np.array(pts)
    dis = np.sqrt(np.sum((pts - pt)**2, axis=1))
    foo = np.mean(dis[1:k+4]) / 100
    return foo

def split_gt(img_name, gt_name, slice_size=(256,256)):
    '''
        将location,num_people按原图split_num*split_num区域分开
        location: [x, y]
    '''
    img = cv2.imread(img_name)
    origin_h, origin_w = img.shape[:2]
    img = cv2.resize(img, slice_size)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    locations, num_people = parse_mat(gt_name)
    row_num, col_num = img.shape[:2]

    slice_col_num = slice_size[1]
    slice_row_num = slice_size[0]

    split_row_num = row_num / slice_row_num
    split_col_num = col_num / slice_col_num

    slice_imgs = {}

    for i in xrange(split_row_num):
        for j in xrange(split_col_num):
            src_x_range = (j*slice_col_num, (j+1)*slice_col_num)
            src_y_range = (i*slice_row_num, (i+1)*slice_row_num)
            #print 'src_x_range: ', src_x_range
            #print 'src_y_range: ', src_y_range
            gt_x_range = (src_x_range[0], src_x_range[1])
            gt_y_range = (src_y_range[0], src_y_range[1])
            #print 'gt_x_range: ', gt_x_range
            #print 'gt_y_range: ', gt_y_range

            slice_gt_key = 'slice_{}_{}_gt'.format(i+1, j+1)
            slice_src_key = 'slice_{}_{}_src'.format(i+1, j+1)
            slice_mask_key = 'slice_{}_{}_mask'.format(i+1, j+1)
            num_key = 'num_{}_{}'.format(i+1, j+1)

            slice_imgs[num_key] = 0
            slice_imgs[slice_gt_key] = np.zeros((slice_row_num, slice_col_num), dtype=np.float32)
            slice_imgs[slice_src_key] = img[src_y_range[0] : src_y_range[1], src_x_range[0] : src_x_range[1], ...]
            slice_imgs[slice_mask_key] = np.zeros((slice_row_num, slice_col_num), dtype=np.uint8)

            gt_row, gt_col = slice_imgs[slice_gt_key].shape[:2]

            gt_merge_img = np.zeros((slice_row_num, slice_col_num), dtype=np.float32)


            for location in locations:
                gt_img = np.zeros((slice_row_num, slice_col_num), dtype=np.float32)

                origin_x = int(location[0] * gt_col / origin_w)
                origin_y = int(location[1] * gt_row / origin_h)

                f_y = origin_y * 1.0 / slice_row_num

                # p_x = origin_x - gt_x_range[0]
                # p_y = origin_y - gt_y_range[0]
                p_x = origin_x
                p_y = origin_y

                if p_x >= 0 and p_x < gt_col and p_y >= 0 and p_y < gt_row :
                    # slice_imgs[slice_gt_key][p_y, p_x] = 255.0
                    gt_img[p_y, p_x] = 255.0
                    slice_imgs[slice_mask_key][p_y, p_x] = 255
                    slice_imgs[num_key] += 1

                # print 'num_people: ', slice_imgs[num_key]
                # filter with gaussian kernel, ksize=25, sigma=1.5
                ksize_h = 50 * f_y + 10
                ksize = (ksize_h, ksize_h)
                sigma = 5.0 * f_y + 2.0
                kernel = matlab_style_gauss2D(ksize, sigma)
                # kernel = (f_y**2 + 0.5) * matlab_style_gauss2D(ksize, sigma)
                # kernel_sum = kernel.sum()
                # if kernel_sum != 0:
                #     kernel /= kernel_sum
                gt_img = cv2.filter2D(gt_img, -1, kernel)

                gt_merge_img += gt_img

            # norm
            #if np.sum(gt_merge_img) > 0:
            #    gt_merge_img = gt_merge_img * (slice_imgs[num_key] * 1.0 / np.sum(gt_merge_img))

            slice_imgs[slice_gt_key] = gt_merge_img

    return slice_imgs, (split_row_num, split_col_num)

def main():
    part_list = ['B']
    phase_list = ['train', 'test']
    
    for part in part_list:
        for phase in phase_list:
            img_names = get_img_names(part, phase)
            gt_names = get_gt_names(part, phase)

            src_img_list_name = 'full_src_part_{}_{}.txt'.format(part, phase)
            #mask_img_list_name = 'mask_part_{}_{}.txt'.format(part, phase)
            label_img_h5_name = 'full_label_part_{}_{}.h5'.format(part, phase)

            # first time, delete file first
            os.system('rm {}'.format(src_img_list_name))
            #os.system('rm {}'.format(mask_img_list_name))
            os.system('rm {}'.format(label_img_h5_name))

            src_save_path = os.path.join('part_{}'.format(part), '{}_data'.format(phase), 'full_crop_images')
            if os.path.exists(src_save_path):
                os.system('rm -rf {}'.format(src_save_path))
            os.system('mkdir {}'.format(src_save_path))

            gt_save_path = os.path.join('part_{}'.format(part), '{}_data'.format(phase), 'full_gt_images')
            if os.path.exists(gt_save_path):
                os.system('rm -rf {}'.format(gt_save_path))
            os.system('mkdir {}'.format(gt_save_path))
            '''mask_save_path = os.path.join('part_{}'.format(part), '{}_data'.format(phase), 'mask_images')
            if not os.path.exists(mask_save_path):
                os.system('mkdir {}'.format(mask_save_path))
            '''

            h5_data = []
            h5_label = []
            #for i in xrange(3):
            for i in xrange(len(img_names)):
                img_name = img_names[i]
                gt_name = gt_names[i]
                #print 'img_name: {}'.format(img_name)
                #print 'gt_name: {}'.format(gt_name)

                slice_imgs, split_num = split_gt(img_name, gt_name, slice_size=(256,256))

                for j in xrange(1, split_num[0]+1):
                    for k in xrange(1, split_num[1]+1):
                        img_base_name = os.path.splitext(os.path.basename(img_name))[0]
                        gt_base_name = os.path.splitext(os.path.basename(gt_name))[0]
                        split_src_name = '{}_{}_{}.jpg'.format(img_base_name, j, k)
                        split_gt_name = '{}_{}_{}.jpg'.format(gt_base_name, j, k)
                        #slice_mask_name = '{}_{}_{}.jpg'.format(i+1, j, k)

                        src_key = 'slice_{}_{}_src'.format(j, k)
                        gt_key = 'slice_{}_{}_gt'.format(j, k)
                        #mask_key = 'slice_{}_{}_mask'.format(j, k)
                        num_key = 'num_{}_{}'.format(j, k)

                        # save images
                        slice_img_path = os.path.join(src_save_path, split_src_name)
                        cv2.imwrite(slice_img_path, slice_imgs[src_key])

                        gt_img = slice_imgs[gt_key].copy()
                        if gt_img.max() != gt_img.min():
                            gt_img = (gt_img - gt_img.min()) * 255 / (gt_img.max() - gt_img.min())
                        gt_img = cv2.applyColorMap(gt_img.astype(np.uint8), cv2.COLORMAP_JET)

                        gt_img_path = os.path.join(gt_save_path, split_gt_name)
                        cv2.imwrite(gt_img_path, gt_img)
                        #cv2.imwrite(os.path.join(mask_save_path, slice_mask_name), slice_imgs[mask_key])
                
                        # write txt
                        with open(src_img_list_name, 'a+') as f:
                            f.write('{} {}\r\n'.format(split_src_name, slice_imgs[num_key]))

                        h5_data.append([slice_imgs[gt_key].astype(np.float32)])
                        h5_label.append([slice_imgs[num_key]])
                        print 'Output {}'.format(slice_img_path)

            # write h5
            print 'write hdf5 file to {}'.format(label_img_h5_name)
            with h5py.File(label_img_h5_name, 'w') as f:
                f['gt_data'] = h5_data
                f['gt_label'] = h5_label
            
def validate():
    f = h5py.File('label_part_B_train.h5', 'r')
    # h5_data = f[u'gt_data']
    h5_label = f[u'gt_label']
    print h5_label[11:31]
    f.close()

if __name__ == '__main__':
    #validate()
    main()