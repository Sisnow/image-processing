import cv2
import numpy as np
import os
from PIL import Image


def run2(img, src_type, dst_type):
    cv2.imwrite("image/tmp_src.jpg", img)
    # 输入
    in_path = "image/tmp_src.jpg"
    # 输出
    out_path = "image/tmp_dst.jpg"
    # 要替换的背景颜色
    color = "deepskyblue"
    # 红：red、蓝：blue、黑：black、白：white

    # 去掉背景颜色
    os.system('backgroundremover -i "' + str(in_path) + '"  -o "image/cg.jpg"')
    # 加上背景颜色
    no_bg_image = Image.open("image/cg.jpg")
    x, y = no_bg_image.size
    new_image = Image.new('RGBA', no_bg_image.size, color=color)
    new_image.paste(no_bg_image, (0, 0, x, y), no_bg_image)
    new_image.save(out_path)
    res = cv2.imread(out_path)
    return res


def run(image, src_type, dst_type):
    # 图像缩放
    img = cv2.resize(image, None, fx=1, fy=1)
    rows, cols, channels = img.shape
    # 图片转换为灰度图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 图片的二值化处理
    if src_type == 'r':
        lower_red = np.array([0, 135, 135])
        upper_red = np.array([180, 245, 230])
        mask = cv2.inRange(hsv, lower_red, upper_red)
    elif src_type == 'b':
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([110, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    else:
        mask = cv2.inRange(hsv, np.array([0, 0, 221]), np.array([180, 30, 255]))
    # 腐蚀膨胀
    erode = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)
    dilate = cv2.dilate(erode, np.ones((10, 10), np.uint8), iterations=1)
    # 转变为
    if dst_type == 'w':
        for i in range(rows):
            for j in range(cols):
                if dilate[i, j] == 255:  # 像素点255表示白色
                    img[i, j] = (255, 255, 255)  # 此处替换颜色，为BGR通道，不是RGB
    elif dst_type == 'r':
        for i in range(rows):
            for j in range(cols):
                if dilate[i, j] == 255:  # 像素点255表示白色
                    img[i, j] = (0, 0, 255)  # 此处替换颜色，为BGR通道，不是RGB
    elif dst_type == 'b':
        for i in range(rows):
            for j in range(cols):
                if dilate[i, j] == 255:  # 像素点255表示白色
                    img[i, j] = (255, 0, 0)  # 此处替换颜色，为BGR通道，不是RGB
    return img
