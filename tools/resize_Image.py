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


def capture_image_random(imgs, size=320, cropped_num=300):
    """
    choose a point as top left coord randomly, then each aixs add  320 pixel, generate
    down right coord, then get a 320 * 320 image
    :param imgs: a list, each elements is a dictionary
    :param size: cropped iamge size
    :param cropped_num: number of cropped iamge
    :return:(TODO)
    """
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        print img['imagePath']
        if im.shape[0] > size and im.shape[1] > size:
            # range of top_left_x and top_left_y
            height_remain = im.shape[0] - size  # 3200 - 320
            weight_remain = im.shape[1] - size  # 2400 - 320
            for i in xrange(cropped_num):
                top_left_x = rd.randint(0, weight_remain)
                top_left_y = rd.randint(0, height_remain)
                # get image file name excluding .jpg
                pattern = re.compile(r'image_\d*')
                search = pattern.search(img['imagePath'])
                image_name = search.group()
                jpgname = '/home/yuquanjie/Documents/cropped_image_optimise/' + image_name + '_' + bytes(i) + '.jpg'
                txtname = '/home/yuquanjie/Documents/cropped_image_optimise/' + image_name + '_' + bytes(i) + '.txt'
                cap_img_poly = Polygon([(top_left_x, top_left_y),
                                        (top_left_x, top_left_y + size),
                                        (top_left_x + size, top_left_y + size),
                                        (top_left_x + size, top_left_y)])
                write_resized_image = True
                # store intersected aeras' coordinates between captured image(320 * 320) and raw image's text region
                # if write_resized_image = True, then write this list in txt file, avoiding writting so many txt files
                strcoord_list = []
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
                        inter = raw_img_poly.intersection(cap_img_poly)
                        # insure the intersected quardrangle's aera is greater than 0
                        if inter.area == 0:
                            write_resized_image = False
                            break
                        # insure the text region's percentage should  greater than 10%
                        if inter.area < (size / 10) * (size / 10):
                            write_resized_image = False
                            break
                        # insure the text region's percentage should not greater than 88%
                        if inter.area > (size - 20) * (size - 20):
                            write_resized_image = False
                            break
                        list_inter = list(inter.exterior.coords)
                        x1, y1 = list_inter[0][0] - top_left_x, list_inter[0][1] - top_left_y
                        x2, y2 = list_inter[3][0] - top_left_x, list_inter[3][1] - top_left_y
                        x3, y3 = list_inter[2][0] - top_left_x, list_inter[2][1] - top_left_y
                        x4, y4 = list_inter[1][0] - top_left_x, list_inter[1][1] - top_left_y
                        # insure the top_left coordinates is on the top-left position
                        if x1 < x2 and y1 < y4 and x3 > x4 and y3 > y2:
                            write_resized_image = True
                        else:
                            write_resized_image = False
                            break
                        # insure the intersected poly is quardrangle
                        if len(list_inter) != 5:
                            write_resized_image = False
                            break
                        # list_inter[0] : top_left, list_inter[1]: down_left, list_inter[2] : dow_righj
                        strcoord = '{0},{1},{2},{3},{4},{5},{6},{7},\n'.format(x1, y1, x2, y2, x3, y3, x4, y4)
                        strcoord_list.append(strcoord)
                    else:
                        write_resized_image = False
                # note : 1st dimension is height, 2nd dimension is width
                if write_resized_image:
                    # print 'writing ... {0}'.format(jpgname)
                    # set writting pattern appending
                    txtwrite = open(txtname, 'a')
                    for strcoo in strcoord_list:
                        txtwrite.write(strcoo)
                    txtwrite.close()
                    cv2.imwrite(jpgname, im[top_left_y: top_left_y + size, top_left_x: top_left_x + size])


def get_resizedimage_toplef_downrig_coord(im, c, size=1000):
    """
    according text region center coordinate to capture a size(1000) * size(1000) image
    should consider some boundary cases
    :param im:
    :param c: cneter coordinate of text region
    :param size: resized image's size
    :return: return top left and down right corner coordinates of resized image
    """
    vis = False
    if c[0] - size/2 < 0 and c[1] - size/2 < 0:
        top_left = [0, 0]
        down_rig = [size, size]
    elif c[1] - size/2 < 0 and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        top_left = [c[0] - size/2, 0]
        down_rig = [c[0] + size/2, size]
    elif c[0] - size/2 < 0 and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        top_left = [0, c[1] - size/2]
        down_rig = [size, c[1] + size/2]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        top_left = [im.shape[1] - size, 0]
        down_rig = [im.shape[1], size]
    elif c[0] + size/2 > im.shape[1] and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        top_left = [im.shape[1] - size, c[1] - size/2]
        down_rig = [im.shape[1], c[1] + size/2]
    elif c[0] - size/2 < 0 and c[1] + size/2 > im.shape[0]:
        top_left = [0, im.shape[0] - 1000]
        down_rig = [1000, im.shape[0]]
    elif c[1] + size/2 > im.shape[0] and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        top_left = [c[0] - size/2, im.shape[0] - size]
        down_rig = [c[0] + size/2, im.shape[0]]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        top_left = [im.shape[1] - size, im.shape[0] - size]
        down_rig = [im.shape[1], im.shape[0]]
    else:
        top_left = [c[0] - size/2, c[1] - size/2]
        down_rig = [c[0] + size/2, c[1] + size/2]

    if vis:
        mydraw.draw_rectangle_image(im, top_left, down_rig, "blue")
    return top_left, down_rig


def resize_image_from_textcenter(imgs, size=320):
    """
    according to text region center, generate  320 * 320 images
    :param imgs: a list(array), each elements is a dictionary
                 'imagePath' :
                 'boxCoord' :
                 'boxNum' :
    :param size: resized size
    :return: no returning value, write image(320 * 320) and corresponding txt file on disk
    """
    for img in imgs:
        imgfilepath = img['imagePath']
        im = cv2.imread(imgfilepath)
        print imgfilepath
        if im.shape[0] > size and im.shape[1] > size:
            idx = 1
            for coord in img['boxCoord']:
                # resize raw iamge to 320 * 320, 5 resized image center
                # calculate center of text region, using top-left and down right coordinates
                text_center = [(string.atof(coord[0]) + string.atof(coord[4])) / 2,
                               (string.atof(coord[1]) + string.atof(coord[5])) / 2]
                text_top_left = [string.atof(coord[0]), string.atof(coord[1])]
                text_top_righ = [string.atof(coord[2]), string.atof(coord[3])]
                text_dow_righ = [string.atof(coord[4]), string.atof(coord[5])]
                text_dow_left = [string.atof(coord[6]), string.atof(coord[7])]

                # define a dict
                resized_image_center = {'cn': text_center, 'tf': text_top_left, 'tr': text_top_righ,
                                        'dr': text_dow_righ, 'dl': text_dow_left}
                # get resized image's top-left and down right coordinates
                for pos in resized_image_center:
                    [top_lef, dow_rig] = get_resizedimage_toplef_downrig_coord(im, resized_image_center[pos], size)
                    # calculate top-right and down-left coordinates
                    top_rig = [dow_rig[0], top_lef[1]]
                    dow_lef = [top_lef[0], dow_rig[1]]
                    # using shapely lib define a captured image Polygon object, is anti-clockwise
                    cap_img_poly = Polygon([(top_lef[0], top_lef[1]), (dow_lef[0], dow_lef[1]),
                                            (dow_rig[0], dow_rig[1]), (top_rig[0], top_rig[1])])
                    # generate captured image file name
                    pattern = re.compile(r'image_\d*')
                    # search = pattern.search(img['imagePath'])
                    search = pattern.search(imgfilepath)
                    image_name = search.group()
                    jpgname = '/home/yuquanjie/Documents/deep-direct-regression/resized_' + bytes(size) + '/' \
                              + image_name + '_' + bytes(idx) + '_' + pos + '.jpg'
                    # write all text file, but write image file only when write_resized_image is ture
                    write_resized_image = True

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
                            txtname = '/home/yuquanjie/Documents/deep-direct-regression/resized_' + bytes(size) + '/' \
                                      + image_name + '_' + bytes(idx) + '_' + pos + '.txt'
                            # set writting pattern appending
                            txtwrite = open(txtname, 'a')
                            inter = raw_img_poly.intersection(cap_img_poly)
                            # insure the intersected quardrangle's aera is greater than 0
                            if inter.area == 0:
                                write_resized_image = False
                                break
                            # insure the text region's percentage should  greater than 10%
                            if inter.area < (size / 10) * (size / 10):
                                write_resized_image = False
                                break
                            # insure the text region's percentage should not greater than 88%
                            if inter.area > (size - 20) * (size - 20):
                                write_resized_image = False
                                break
                            list_inter = list(inter.exterior.coords)
                            x1, y1 = list_inter[0][0] - top_lef[0], list_inter[0][1] - top_lef[1]
                            x2, y2 = list_inter[3][0] - top_lef[0], list_inter[3][1] - top_lef[1]
                            x3, y3 = list_inter[2][0] - top_lef[0], list_inter[2][1] - top_lef[1]
                            x4, y4 = list_inter[1][0] - top_lef[0], list_inter[1][1] - top_lef[1]
                            # insure the top_left coordinates is on the top-left position
                            if x1 < x2 and y1 < y4 and x3 > x4 and y3 > y2:
                                write_resized_image = True
                            else:
                                write_resized_image = False
                                break
                            # insure the intersected poly is quardrangle
                            if len(list_inter) != 5:
                                write_resized_image = False
                                break
                            # list_inter[0] : top_left, list_inter[1]: down_left, list_inter[2] : dow_righ
                            strcoord = '{0},{1},{2},{3},{4},{5},{6},{7},\n'.format(bytes(list_inter[0][0] - top_lef[0]),
                                                                                   bytes(list_inter[0][1] - top_lef[1]),
                                                                                   bytes(list_inter[3][0] - top_lef[0]),
                                                                                   bytes(list_inter[3][1] - top_lef[1]),
                                                                                   bytes(list_inter[2][0] - top_lef[0]),
                                                                                   bytes(list_inter[2][1] - top_lef[1]),
                                                                                   bytes(list_inter[1][0] - top_lef[0]),
                                                                                   bytes(list_inter[1][1] - top_lef[1]))
                            if write_resized_image:
                                txtwrite.write(strcoord)
                    txtwrite.close()
                    # note : 1st dimension is height, 2nd dimension is width
                    if write_resized_image:
                        # print 'writing ... {0}'.format(jpgname)
                        cv2.imwrite(jpgname, im[int(top_lef[1]): int(dow_rig[1]), int(top_lef[0]): int(dow_rig[0])])
                    idx += 1


if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1')
    # resize_image_from_textcenter(all_imgs, 320)
    capture_image_random(all_imgs, 320, 2200)
