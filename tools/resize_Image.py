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
    :return: return top left and down right corner coordinates of captured image
    """
    vis = False
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
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        imgfilepath = img['imagePath']
        print img['imagePath']
        if im.shape[0] > 1000 and im.shape[1] > 1000:

            idx = 1
            for coord in img['boxCoord']:
                # calculate center of text region
                text_center = [(string.atof(coord[0]) + string.atof(coord[4])) / 2,
                               (string.atof(coord[1]) + string.atof(coord[5])) / 2]
                # get captured image's top-left and down right coordinates
                [top_lef, dow_rig] = get_captured_img_toplef_downrig_coord(im, text_center)
                # calculate top-right and down-left coordinates
                top_rig = [dow_rig[0], top_lef[1]]
                dow_lef = [top_lef[0], dow_rig[1]]
                # using shapely lib define a captured image Polygon object
                cap_img_poly = Polygon([(top_lef[0], top_lef[1]), (dow_lef[0], dow_lef[1]),
                                        (dow_rig[0], dow_rig[1]), (top_rig[0], top_rig[1])])
                # generate captured image file name
                pattern = re.compile(r'image_\d*')
                # search = pattern.search(img['imagePath'])
                search = pattern.search(imgfilepath)
                image_name = search.group()
                jpgname = '/home/yuquanjie/Documents/deep-direct-regression/captured_data/' \
                          + image_name + '_' + bytes(idx) + '.jpg'
                poly_point_is5 = True

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

                    if raw_img_poly.intersects(cap_img_poly):
                        txtname = '/home/yuquanjie/Documents/deep-direct-regression/captured_data/' \
                                  + image_name + '_' + bytes(idx) + '.txt'
                        # writting pattern is appending
                        txtwrite = open(txtname, 'a')
                        inter = raw_img_poly.intersection(cap_img_poly)
                        # insure the intersected quardrangle's aera is greater than 0
                        if inter.area == 0:
                            poly_point_is5 = False
                            break
                        list_inter = list(inter.exterior.coords)
                        x1, y1 = list_inter[0][0] - top_lef[0], list_inter[0][1] - top_lef[1]
                        x2, y2 = list_inter[3][0] - top_lef[0], list_inter[3][1] - top_lef[1]
                        x3, y3 = list_inter[2][0] - top_lef[0], list_inter[2][1] - top_lef[1]
                        x4, y4 = list_inter[1][0] - top_lef[0], list_inter[1][1] - top_lef[1]
                        # insure the top_left coordinates is on the right position
                        if x1 < x2 and y1 < y4 and x3 > x4 and y3 > y2:
                            poly_point_is5 = True
                        else:
                            poly_point_is5 = False
                            break
                        # insure the intersected poly is quardrangle
                        if len(list_inter) != 5:
                            poly_point_is5 = False
                            break

                        # list_inter[0] : top_left, list_inter[1]: down_left
                        # list_inter[2] : dow_righ, list_inter[3]: top_righ
                        strcoord = '{0},{1},{2},{3},{4},{5},{6},{7},\n'.format(bytes(list_inter[0][0] - top_lef[0]),
                                                                               bytes(list_inter[0][1] - top_lef[1]),
                                                                               bytes(list_inter[3][0] - top_lef[0]),
                                                                               bytes(list_inter[3][1] - top_lef[1]),
                                                                               bytes(list_inter[2][0] - top_lef[0]),
                                                                               bytes(list_inter[2][1] - top_lef[1]),
                                                                               bytes(list_inter[1][0] - top_lef[0]),
                                                                               bytes(list_inter[1][1] - top_lef[1]))
                        txtwrite.write(strcoord)
                txtwrite.close()
                # save image only when intersected polygon point number is 5 (4 + 1)
                # note : 1st dimension is height, 2nd dimension is width
                if poly_point_is5:
                    cv2.imwrite(jpgname, im[int(top_lef[1]): int(dow_rig[1]), int(top_lef[0]): int(dow_rig[0])])
                idx += 1


if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1')
    # all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/tmp')
    capture_image_from_textcenter(all_imgs)
