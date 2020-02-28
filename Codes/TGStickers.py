from libpy import Init
import numpy as np
import math
from scipy.optimize import minimize

import cv2

import sys
import os

def lik(para, inps):
    mu = para[0]
    sigma = para[1]
    if mu < 0 or mu > 255 or sigma <= 0:
        return 1000000
    outs = 0
    for i in range(0, len(inps)):
        try:
            outs += math.log(2 * math.pi * sigma) / 2 + (inps[i] - mu) ** 2 /(2 * sigma**2)
        except ValueError:
            continue
    #print(outs/len(inps))
    return outs/len(inps)


def main(dir_name, method = "full"):
    os.system("rm -rf ./OldOutput")
    os.system("mv ./Output ./OldOutput")
    os.system("mkdir ./Output")

    files,names = Init.GetSufixFile(dir_name, ["png", "jpg", "jpeg"])
    #Main algorithm loop
    for kase in range(0, len(files)):
        #Initial
        img = Init.ImageIO(file_dir = files[kase], img = [], io = "i", mode = "rgb", backend = "")
        
        #Resize Image
        
        (true_height, true_width, _) = np.shape(img)
        if true_height > true_width:
            height = 512
            width = int(512/true_height*true_width)
        else:
            width = 512
            height = int(512/true_width*true_height)
        size = (width, height)
        """
        size = (100, 100)
        """
        img = cv2.resize(img, size)

        if method == "resize":
            Init.ImageIO(file_dir = "./Output/" + names[kase] + ".png", img = img, io = "o", mode = "grey", backend = "")
            continue

        #Segmentation area finding
        segment_img = np.zeros((height, width))
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if img[i][j][3] != 0:
                    segment_img[i][j] = 255

        #Segmentation
        segment_img = cv2.Laplacian(segment_img, cv2.CV_64F)
        for i in range(0, len(segment_img)):
            for j in range(0, len(segment_img[i])):
                if segment_img[i][j] >= 1:
                    segment_img[i][j] = 255
                else:
                    segment_img[i][j] = 0

        #Segmentation connection area
        kernel_color = np.ones((4,4), np.float32)
        color_seg_img = cv2.filter2D(segment_img, -1, kernel_color)
        kernel_seg = np.ones((8,8), np.float32)
        exp_seg_img = cv2.filter2D(segment_img, -1, kernel_seg)
        kernel_smooth = np.ones((12,12), np.float32)
        exp_smooth_img = cv2.filter2D(segment_img, -1, kernel_smooth)
        
        #Pretreatment the expand boundary
        for i in range(0, len(exp_seg_img)):
            for j in range(0, len(exp_seg_img[i])):
                if exp_seg_img[i][j] >= 1:
                    exp_seg_img[i][j] = 255
                else:
                    exp_seg_img[i][j] = 0
                if exp_smooth_img[i][j] >= 1:
                    exp_smooth_img[i][j] = 255
                else:
                    exp_smooth_img[i][j] = 0
                if color_seg_img[i][j] >= 1:
                    color_seg_img[i][j] = 255
                else:
                    color_seg_img[i][j] = 0

        #Smooth the white boundary
        result_seg = cv2.GaussianBlur(exp_seg_img, (7, 7), sigmaX = 1)
        
        #Add white boundary to image
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if img[i][j][3] != 0:
                    if exp_seg_img[i][j] != 0:
                        img[i][j][3] = 255
                else:
                    img[i][j][0] = 255
                    img[i][j][1] = 255
                    img[i][j][2] = 255
                    img[i][j][3] = result_seg[i][j]

        #Get the image boundary and statistic
        img_smooth_R = np.zeros((height, width))
        img_smooth_G = np.zeros((height, width))
        img_smooth_B = np.zeros((height, width))

        for i in range(0, len(exp_smooth_img)):
            for j in range(0, len(exp_smooth_img[i])):
                if img[i][j][3] != 0 and exp_smooth_img[i][j] >= 1:
                    img_smooth_R[i][j] = img[i][j][0]
                    img_smooth_G[i][j] = img[i][j][1]
                    img_smooth_B[i][j] = img[i][j][2]

        
        #Delete outer boundary statistic
        R_table = np.array([0 for n in range (256)])
        G_table = np.array([0 for n in range (256)])
        B_table = np.array([0 for n in range (256)])
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if img[i][j][3] > 0:
                    R_table[int(img[i][j][0] + 0.5)] += 1
                    G_table[int(img[i][j][1] + 0.5)] += 1
                    B_table[int(img[i][j][2] + 0.5)] += 1
        R_table[255] = 0
        G_table[255] = 0
        B_table[255] = 0

        #RGB_sum = R_table.sum() + G_table.sum() + B_table.sum()
        #R_para = R_table.sum() / RGB_sum
        #G_para = G_table.sum() / RGB_sum
        #B_para = B_table.sum() / RGB_sum
        #print(R_para, G_para, B_para)
        #R_table = R_table / R_table.sum()
        #G_table = G_table / G_table.sum()
        #B_table = B_table / B_table.sum()

        #Fit the RGB center with MLE in 1-d GM
        #R_fit = minimize(lik, np.array([128,10]), R_table, method='L-BFGS-B').x[0]
        #G_fit = minimize(lik, np.array([128,10]), G_table, method='L-BFGS-B').x[0]
        #B_fit = minimize(lik, np.array([128,10]), B_table, method='L-BFGS-B').x[0]
        #R_fit = int(max(0, np.argmax(R_table) - 50 * R_para))
        #G_fit = int(max(0, np.argmax(G_table) - 50 * G_para))
        #B_fit = int(max(0, np.argmax(B_table) - 50 * B_para))
        R_fit = int(max(0, np.argmax(R_table) * 0.9))
        G_fit = int(max(0, np.argmax(G_table) * 0.9))
        B_fit = int(max(0, np.argmax(B_table) * 0.9))
        #R_fit = int(np.argmax(R_table)*0.8)
        #G_fit = int(np.argmax(G_table)*0.8)
        #B_fit = int(np.argmax(B_table)*0.8)
        #print(R_fit, G_fit, B_fit)
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if color_seg_img[i][j] >= 1:
                    img_smooth_R[i][j] = R_fit
                    img_smooth_G[i][j] = G_fit
                    img_smooth_B[i][j] = B_fit

        #Smooth image boundary
        img_smooth_R = cv2.GaussianBlur(img_smooth_R, (17, 17), sigmaX = 1)
        img_smooth_G = cv2.GaussianBlur(img_smooth_G, (17, 17), sigmaX = 1)
        img_smooth_B = cv2.GaussianBlur(img_smooth_B, (17, 17), sigmaX = 1)

        #After treatment
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if img[i][j][3] != 0:
                    if exp_seg_img[i][j] >= 1:
                        img[i][j][0] = img_smooth_R[i][j]
                        img[i][j][1] = img_smooth_G[i][j]
                        img[i][j][2] = img_smooth_B[i][j]
        Init.ImageIO(file_dir = "./Output/" + names[kase] + ".png", img = img, io = "o", mode = "grey", backend = "")


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Input parameter lost, please run the project with arg of location")
    else:
        if len(args) == 2:
            main(args[1])
        else:
            main(args[1], args[2])











