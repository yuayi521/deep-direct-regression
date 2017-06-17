from PIL import Image, ImageDraw
import cv2
import numpy as np


def draw_rectangle_image(im, roi):
    """
    draw a rectangel on image
    :param im: cv2.imread() function's returned value
               numpy.ndarray
    :param roi:2-dimension array,
               1st dimension is x-axis range, from start to end
               2nd dimension is y-axis range, from start to end
    :return:
    """
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    draw.polygon([(roi[0][0], roi[1][0]), (roi[0][1], roi[1][0]),
                  (roi[0][1], roi[1][1]), (roi[0][0], roi[1][1])],
                 fill='blue', outline='red')
    img_draw = np.array(img_draw)
    im = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(im, (1000, 1000)))
    cv2.waitKey(0)
