"""
    @author  jasonYu
    @date    2017/6/14
    @version created
    @email   yuquanjie13@gmail.com
    @descrip capture image randomly from full size, eg. 2400 * 3200 to small size,
             eg.1200 * 1600
"""
import cv2
import cv2.cv as cv
import tools.get_data as get_data
import random as rd
from PIL import Image, ImageDraw
import numpy as np


def visiual_image(im_path, coord_toplef, coord_downrig):
    """
    visualize captured image, the size is 1000 * 1000
    :param im_path:
    :param coord_toplef: top left coordinates
    :param coord_downrig: right down coordinates
    :return: show image
    """
    im = cv2.imread(im_path)
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    draw.polygon([(coord_toplef[0], coord_toplef[1]), (coord_downrig[0], coord_toplef[1]),
                  (coord_downrig[0], coord_downrig[1]), (coord_toplef[0], coord_downrig[1])],
                 fill='blue', outline='red')
    img_draw = np.array(img_draw)
    im = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(im, (1000, 1000)))

def capture_image(imgs):
    """
    (TODO)
    :param imgs: a list, each elements is a dictionary
                 'imagePath'
                 'boxCoord'
                 'boxNum'
    :return:(TODO)
    """
    visiual = False
    for img in imgs:
        im = cv.LoadImage(img['imagePath'])
        im_size = cv2.imread(img['imagePath'])
        if im_size.shape[0] > 1000 and im_size.shape[1] > 1000:
            print img['imagePath']
            # choose a top-left corner
            height_remain = im_size.shape[0] - 1000  # 3200 - 1000
            weight_remain = im_size.shape[1] - 1000  # 2400 - 1000
            # for i in xrange(min(height_remain, weight_remain)):
            for i in xrange(50):
                # top_left = [rd.randint(0, height_remain), rd.randint(0, weight_remain)]
                top_left = [rd.randint(0, weight_remain), rd.randint(0, height_remain)]
                down_right = [top_left[0] + 1000, top_left[1] + 1000]
                if visiual:
                    visiual_image(img['imagePath'], top_left, down_right)
                cv.SetImageROI(im, (top_left[0], top_left[1], down_right[0], down_right[1]))
                print '/home/yuquanjie/Documents/deep-direct-regression/captured_data' + bytes(i)
                cv.SaveImage('/home/yuquanjie/Documents/deep-direct-regression/captured_data' + bytes(i), im)
                # cv.ShowImage('img', im)
                # cv.WaitKey(0)

if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2'
                                                 '/train/part1')
    capture_image(all_imgs)
    print type(all_imgs)
