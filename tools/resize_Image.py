"""
    @author  jasonYu
    @date    2017/6/14
    @version created
    @email   yuquanjie13@gmail.com
    @descrip capture image randomly from full size, eg. 2400 * 3200 to small size,
             eg.1200 * 1600
"""
import cv2
import tools.get_data as get_data
import tools.mydraw as mydraw
import random as rd
import re
import string
from shapely.geometry import Polygon


def capture_image_random(imgs):
    """
    choose a point as top left coord randomly, then each aixs add  1000 pixel, generate
    down right coord, then get a 1000 * 1000 image
    :param imgs: a list, each elements is a dictionary
                 'imagePath'
                 'boxCoord'
                 'boxNum'
    :return:(TODO)
    """
    visiual = False
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        if im.shape[0] > 1000 and im.shape[1] > 1000:
            print img['imagePath']
            # choose a top-left corner
            height_remain = im.shape[0] - 1000  # 3200 - 1000
            weight_remain = im.shape[1] - 1000  # 2400 - 1000
            for i in xrange(50):
                weight_rand_idx = rd.randint(0, weight_remain)
                height_rand_idx = rd.randint(0, height_remain)
                # using regular expression to match image file name excluding .jpb
                pattern = re.compile(r'image_\d*')
                search = pattern.search(img['imagePath'])
                image_name = search.group()
                filename = '/home/yuquanjie/Documents/deep-direct-regression/captured_data/'\
                           + image_name + '_' + bytes(i) + '.jpg'
                # note : 1st dimension is height
                cv2.imwrite(filename, im[height_rand_idx: height_rand_idx + 1000,
                            weight_rand_idx: weight_rand_idx + 1000])
                if visiual:
                    roi_range = [[weight_rand_idx, weight_rand_idx + 1000],
                                 [height_rand_idx, height_rand_idx + 1000]]
                    mydraw.draw_rectangle_image(im, roi_range)


def get_captured_img_toplef_downrig_coord(im, c):
    """
    according text region center coordinate to capture a 1000 * 1000 image
    should consider some boundary cases
    :param im:
    :param c: cneter coordinate of text region
    :return:
    """
    vis = True
    if c[0] - 500 < 0 and c[1] - 500 < 0:
        top_left = [0, 0]
        down_rig = [1000, 1000]
    elif c[1] - 500 < 0 and (c[0] - 500 > 0 and c[0] + 500 < im.shape[1]):
        top_left = [c[0] - 500, 0]
        down_rig = [c[0] + 500, 1000]
    elif c[0] - 500 < 0 and (c[1] - 500 > 0 and c[1] + 500 < im.shape[0]):
        top_left = [0, c[1] - 500]
        down_rig = [1000, c[1] + 500]
    elif c[0] + 500 > im.shape[1] and c[1] + 500 > im.shape[0]:
        top_left = [im.shape[1] - 1000, 0]
        down_rig = [im.shape[1], 1000]
    elif c[0] + 500 > im.shape[1] and (c[1] - 500 > 0 and c[1] + 500 < im.shape[0]):
        top_left = [im.shape[1] - 1000, c[1] - 500]
        down_rig = [im.shape[1], c[1] + 500]
    elif c[0] - 500 < 0 and c[1] + 500 > im.shape[0]:
        top_left = [0, im.shape[0] - 1000]
        down_rig = [1000, im.shape[0]]
    elif c[1] + 500 > im.shape[0] and (c[0] - 500 > 0 and c[0] + 500 < im.shape[1]):
        top_left = [c[0] - 500, im.shape[0] - 1000]
        down_rig = [c[0] + 500, im.shape[0]]
    elif c[0] + 500 > im.shape[1] and c[1] + 500 > im.shape[0]:
        top_left = [im.shape[1] - 1000, im.shape[0] - 1000]
        down_rig = [im.shape[1], im.shape[0]]
    else:
        top_left = [c[0] - 500, c[1] - 500]
        down_rig = [c[0] + 500, c[1] + 500]

    if vis:
        mydraw.draw_rectangle_image(im, top_left, down_rig, "blue")
    return top_left, down_rig


def capture_image_from_textcenter(imgs):
    """
    according to text region center, generate  1000 * 1000 images
    :param imgs: a list(array), each elements is a dictionary
                 'imagePath' :
                 'boxCoord' :
                 'boxNum' :
    :return:
    """
    visiual = True
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        if im.shape[0] > 1000 and im.shape[1] > 1000:
            print img['imagePath']
            for coord in img['boxCoord']:
                # calculate center of text region
                text_center = [(string.atof(coord[0]) + string.atof(coord[4])) / 2,
                               (string.atof(coord[1]) + string.atof(coord[5])) / 2]
                if visiual:
                    mydraw.draw_text_on_image(im, text_center, "OOO")
                print coord
                print text_center
                [top_lef, dow_rig] = get_captured_img_toplef_downrig_coord(im, text_center)
                print top_lef, dow_rig
                print img['boxNum']
                for polygon in img['boxCoord']:
                    x1 = string.atof(polygon[0])
                    x2 = string.atof(polygon[2])
                    x3 = string.atof(polygon[4])
                    x4 = string.atof(polygon[6])

                    y1 = string.atof(polygon[1])
                    y2 = string.atof(polygon[3])
                    y3 = string.atof(polygon[5])
                    y4 = string.atof(polygon[7])
                    raw_img_poly = Polygon([(x1, y1), (x4, y4), (x3, y3), (x2, y2)])

                    top_rig = [dow_rig[0], top_lef[1]]
                    dow_lef = [top_lef[0], dow_rig[1]]
                    cap_img_poly = Polygon([(top_lef[0], top_lef[1]),
                                            (dow_lef[0], dow_lef[1]),
                                            (dow_rig[0], dow_rig[1]),
                                            (top_rig[0], top_rig[1])])

                    if raw_img_poly.intersects(cap_img_poly):
                        inter = raw_img_poly.intersection(cap_img_poly)
                        list_my = list(inter.exterior.coords)
                        mydraw.draw_rectangle_image(im, [list_my[0][0], list_my[0][1]],
                                                    [list_my[2][0], list_my[2][1]], "black")
                        print raw_img_poly
                        print cap_img_poly
                        print inter


if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2'
                                                 '/train/part1')
    capture_image_from_textcenter(all_imgs)
    print type(all_imgs)
