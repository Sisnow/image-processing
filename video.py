import os

import cv2

from transfer import stylize

# 保存图片
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    image = cv2.resize(image, (512, 512))
    cv2.imwrite(address, image)


def run(original_path, style_type):
    # 获取初始和目标地址
    # original_path = "./original/hangzhou.mp4"
    post_path = "result.avi"

    videoCapture = cv2.VideoCapture(original_path)
    # 把视频转换为图片
    success, frame = videoCapture.read()
    i = 0
    while success:
        i = i + 1
        save_image(frame, './frames/\\', i)
        success, frame = videoCapture.read()

    path = './frames/'
    trans = './transfered/'
    img_list = os.listdir(path)
    get_key = lambda i: int(i.split('.')[0])
    img_sort = sorted(img_list, key=get_key)
    fps = 24
    size = (512, 512)

    video = cv2.VideoWriter(post_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # 选取风格
    if style_type == 1:
        style_path = "style.pth"
    elif style_type == 2:
        style_path = "style2.pth"
    # 进行风格转换并写入新视频
    for item in img_sort:
        if item.endswith('.jpg'):
            item2 = trans + item
            item = path + item
            stylize(item, item2, style_path)
            img = cv2.imread(item2)
            video.write(img)
    video.release()
    cv2.destroyAllWindows()


