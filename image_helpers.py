import cv2
import numpy as np
import random

def convert_to_gray(img):
    return np.sum(img/3, axis=2, keepdims=True)

def augment_brightness(image):
    """
    Randomly adds brightness in all three (rgb) channels.
    :param image:
    :return:
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_guassian_blur(img):
    return cv2.GaussianBlur(img,(3,3),0)

def translate_image(img, trans_range_x, trans_range_y):
    """
    Randomly transforms image by moving pixes in x and y direction using given paramters.
    :param img:
    :param trans_range_x:
    :param trans_range_y:
    :return:
    """
    # Translation
    rows,cols,ch = img.shape
    tr_x = trans_range_x*np.random.uniform()-trans_range_x/2
    tr_y = trans_range_y*np.random.uniform()-trans_range_y/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    return cv2.warpAffine(img,Trans_M,(cols,rows))

def rotate_image(img, ang_range):
    """
    Randomly rotate the image
    :param img:
    :param ang_range:
    :return:
    """
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    return cv2.warpAffine(img,Rot_M,(cols,rows))

def transform_img(img, trans_range_x, trans_range_y):
    """
    Randomly transforms image
    1. Adding random guassian blue
    2. by moving pixes in x and y direction using given paramters.
    :param img:
    :return:
    """
    img = add_guassian_blur(img)
    img = translate_image(img, trans_range_x, trans_range_y)
    # if random.randint(0, 100) < 50:
    #     img = augment_brightness(img)
    return img
