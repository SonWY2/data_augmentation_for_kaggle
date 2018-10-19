# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 01:04:40 2018
@author: Administrator
"""

import cv2
import os
import numpy as np
import glob
import json
from pprint import pprint
import shutil
import Augmentor
from time import sleep
from datetime import datetime

from collections import OrderedDict
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="target image / json file location.")
parser.add_argument("save_path", type=str, help="non contrast adjusted image files will be stored with ground truth images.)
parser.add_argument("save_path_2", type=str, help="save path for hist equalization images")
parser.add_argument("output_size", type=int, help="output image size")
parser.add_argument("samples", type=int, help="how many images will you make")
args = parser.parse_args()
standard_data_path = args.input_path
standard_data_image_path = standard_data_path + "/images"
standard_data_json_path = standard_data_path + "/json"
non_contrast_augmentation_path = args.save_path
contrast_augmentation_path = args.save_path_2
output_size = args.output_size
samples_num = args.samples
'''


standard_data_path = "./00_standard_dataset"
standard_data_image_path = standard_data_path + "/images"
standard_data_json_path = standard_data_path + "/json"

non_contrast_augmentation_path = "./01_non_contrast_augmentation_data"
non_contrast_augmentation_image_path = non_contrast_augmentation_path + "/images"
non_contrast_augmentation_json_path = non_contrast_augmentation_path + "/json"

contrast_augmentation_path = "./02_contrast_augmentation_data"
contrast_augmentation_image_path = contrast_augmentation_path + "/images"
contrast_augmentation_json_path = contrast_augmentation_path + "/json"


output_size = 620
samples_num = 100

def path_check(path) :
    if os.path.exists(path) :
        shutil.rmtree(path)
    os.mkdir(path)
    return True

def image_downsampling(path, output_size=512) :
    absp = os.path.abspath(path)
    li = absp.split(os.sep) 
#    sm_img_path = os.path.join(*li[:-1]) + os.sep + "image_" + str(output_size) + "x" + str(output_size) # for ubuntu
    sm_img_path = ""
    for p in range(0, len(li) - 1) :
        sm_img_path += li[p] + os.sep
    sm_img_path += "image_" + str(output_size) + "x" + str(output_size)
    path_check(sm_img_path)
    
    size_dict = dict()
    image_list = glob.glob(path + "/*")
    for i in image_list :
        img = cv2.imread(i, 0)
        h, w = img.shape
        ds_img = cv2.resize(img, (output_size, output_size))
        cv2.imwrite(sm_img_path + os.sep + os.path.basename(i), ds_img)
        size_dict[os.path.basename(i).split(".")[0]] = (h, w)
    
    print("1) image down sampling done.")
    return size_dict, sm_img_path

def json_to_gt_image(path, size_dict, output_size=512) :
    absp = os.path.abspath(path)
    li = absp.split(os.sep) 
#    gt_path = os.path.join(*li[:-1]) + os.sep + "ground_truth_image" # for ubuntu
    '''
    # FIXME :
    '''
    gt_path = ""
    for p in range(0, len(li) - 1) :
        gt_path += li[p] + os.sep
    gt_path += "ground_truth_image_" + str(output_size) + "x" + str(output_size)
    path_check(gt_path)
    
    json_list = glob.glob(path + "/*")
    for j in json_list :
        with open(j) as json_file :
            data = json.load(json_file)
        
        box = data["boxes"][0]
        x1 = int(box['x1']); x2 = int(box['x2'])
        y1 = int(box['y1']); y2 = int(box['y2'])
        
        fname = os.path.basename(j).split(".")[0]
        h, w = size_dict[fname]
        
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        mask = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_AREA)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        cv2.imwrite(gt_path + "/" + fname + ".png", mask)
    
    print("2) gt image generated. done.")
    return gt_path
    
def image_augmentation(sm_img_path, gt_path, non_contrast_augmentation_path, samples_num = 10, hist_equal_mode=False) :
    def get_4_coordinates(img) :
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        _, contours, _ = cv2.findContours(img.copy(), 0, 1) # background : black. edge : white
        for cc in contours :
            ccx, ccy, w, h = cv2.boundingRect(cc)

        top_x, top_y = tuple(cc[cc[:, :, 1].argmin()][0])
        bottom_x, bottom_y = tuple(cc[cc[:, :, 1].argmax()][0])
        left_x, left_y = tuple(cc[cc[:, :, 0].argmin()][0])
        right_x, right_y = tuple(cc[cc[:, :, 0].argmax()][0])
        
        x_order = np.sort([top_x, bottom_x, left_x, right_x])
        y_order = np.sort([top_y, bottom_y, left_y, right_y])
        
        x1 = int((x_order[0] + x_order[1]) / 2)
        x2 = int((x_order[2] + x_order[3]) / 2)
        y1 = int((y_order[0] + y_order[1]) / 2)
        y2 = int((y_order[2] + y_order[3]) / 2)
        
        return (x1, y1, x2, y2)
    
    def coords_to_json(patient_id, coords) :
        x1, y1, x2, y2 = coords
        data = OrderedDict()
        data["id"] = os.path.basename(gt)
        coords_ = []
        boxes = OrderedDict()
        boxes["x1"] = float(x1)
        boxes["y1"] = float(y1)
        boxes["x2"] = float(x2)
        boxes["y2"] = float(y2)
        coords_.append(boxes)
        data["boxes"] = coords_
        return data
        
    path_check(non_contrast_augmentation_path)
    
    image_p = non_contrast_augmentation_path + os.sep + "images"
    path_check(image_p)

    json_p = non_contrast_augmentation_path + os.sep + "json"
    path_check(json_p)
    
    gt_p = non_contrast_augmentation_path + os.sep + "gt_temp"
    path_check(gt_p)
    
    
    today = datetime.today().strftime("%y%m%d")
    
    name_dict = dict()
    
    p = Augmentor.Pipeline(sm_img_path)
    p.ground_truth(gt_path)
    
    # elastic transfrom & rotate & horizontal flip
    p.flip_left_right(probability=0.4)
    p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=10)
    p.rotate(probability=1, max_left_rotation= 3 , max_right_rotation = 3)
    p.sample(samples_num)
    
    # augmented mask image move to temp folder
    target_folder = sm_img_path + os.sep + "output"
    t_filelist = glob.glob(target_folder + "/*")   
    f_idx = 0
    for _, f in enumerate(t_filelist) :
        if os.path.isdir(f) == True : continue
        if os.path.basename(f).find('groundtruth') != -1 :
            f_idx += 1
            f_name = os.path.basename(f).replace('_groundtruth_(1)_', '')
            f_name = f_name.replace(sm_img_path.split(os.sep)[-1], sm_img_path.split(os.sep)[-1] + "_original")
            new_name = today + "_" + ("%06d"%f_idx) + ".png"
            name_dict[f_name] = new_name
            shutil.move(f, gt_p + os.sep + new_name)
            
            
    gt_img_list = glob.glob(gt_p + "/*")
    for gt in gt_img_list :
        img = cv2.imread(gt, 0)
        coords = get_4_coordinates(img)
        json_data = coords_to_json(os.path.basename(gt), coords)
        with open(json_p + os.sep + os.path.basename(gt).split('.')[0] + ".json", 'w', encoding="utf-8") as make_file:
            json.dump(json_data, make_file,  indent="\t")
    shutil.rmtree(gt_p, ignore_errors=True)
    
    # augmented image move to images folder.
    # loop once again for mathicng name.
    t_filelist = glob.glob(target_folder + "/*")
    for _, f in enumerate(t_filelist) :
        if os.path.isdir(f) == True : continue
        new_name = name_dict[os.path.basename(f)]
        shutil.move(f, image_p + os.sep + new_name)
    shutil.rmtree(target_folder, ignore_errors=True)

    # hist equalization img
    if hist_equal_mode == True :
        img_list = glob.glob(image_p + "/*")

        path_check(contrast_augmentation_path)
        image_p = contrast_augmentation_path + os.sep + "images"
        path_check(image_p)
        
        for i in img_list :
            img = cv2.imread(i, 0)
            img = cv2.equalizeHist(img)
            cv2.imwrite(image_p + os.sep + os.path.basename(i), img)
        
        shutil.copytree(json_p, contrast_augmentation_path + os.sep + "json")
    

    print("3) image augmentation done.")
    return True

if __name__ == '__main__' :
    
    # 1) image downsampling
    size_dict, sm_img_path = image_downsampling(standard_data_image_path, output_size)
    
    # 2) json file to ground truth mask image
    gt_path = json_to_gt_image(standard_data_json_path, size_dict, output_size)
    
    # 3) image augmentation. (elastic distortion, rotation)
    image_augmentation(sm_img_path, gt_path, non_contrast_augmentation_path, samples_num, hist_equal_mode=True)