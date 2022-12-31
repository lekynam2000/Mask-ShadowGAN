#!/usr/bin/python3

import argparse
import sys
import os

from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import math
from statistics import mean

def compare(gt,est):
    result=dict()
    gt_img = Image.open(gt).convert('RGB')
    est_img = Image.open(est).convert('RGB') 
    result["rmse"] = math.sqrt(mse(gt_img,est_img))
    result["psnr"] = psnr(gt_img,est_img)
    result["ssim"] = ssim(gt_img,est_img,channel_axis=0)
    return result

def evaluate(gt_dir,est_dir,img_suf=".png"):
    gt_list = [os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(img_suf)]
    est_set = set([os.path.splitext(f)[0] for f in os.listdir(est_dir) if f.endswith(img_suf)])
    metrics_list={"rmse":[],"psnr":[],"ssim":[]}
    for idx, img_name in enumerate(gt_list):
        if img_name in est_set:
            gt = os.path.join(gt_dir,img_name+img_suf)
            est = os.path.join(est_dir,img_name+img_sur)
            metrics = compare(gt,est)
            for m in metrics:
                metrics_list[m].append(metrics[m]) 
        else:
            print(f"Error: image {img_name} is not generated")
    for metric in metrics_list:
        print(f"{metric}: {mean(metrics_list[metric])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='ground truth dir')
    parser.add_argument('--est_dir', type=str, help='generated images dir')
    parser.add_argument('--img_suf', type=str, default='.png', help='Image type')
    opt = parser.parse_args()
    evaluate(opt.gt_dir, opt.est_dir, opt.img_suf)