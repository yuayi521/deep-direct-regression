from PIL import Image, ImageDraw
import cv2
import numpy as np


def draw_rectangle_image(im, top_lef, dow_rig):
    """
    accroding to top_left and down_right coordinates to draw a rectangle on a iamge
    :param im: cv2.imread() function's returned value
               numpy.ndarray
    :param top_lef: top left corner coordinates
    :param dow_rig: down right corner coordinates
    :return:
    """

    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    # calculate top_right and down_left corner coordinates
    top_rig = [dow_rig[0], top_lef[1]]
    dow_lef = [top_lef[0], dow_rig[1]]
    draw.polygon([(top_lef[0], top_lef[1]), (top_rig[0], top_rig[1]),
                  (dow_rig[0], dow_rig[1]), (dow_lef[0], dow_lef[1])],
                 fill='blue', outline='red')
    img_draw = np.array(img_draw)
    im = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(im, (1000, 1000)))
    cv2.waitKey(0)


def draw_text_on_image(img, point):
    """
    draw a point with text
    :param img:
    :param point:
    :return:
    """
    img_draw = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    text = "OOOOOO"
    draw.text((point[0], point[1]), text, "red")
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(img_draw, (1000, 1000)))
    cv2.waitKey(0)
