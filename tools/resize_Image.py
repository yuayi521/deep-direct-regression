"""
    @author  jasonYu
    @date    2017/6/14
    @version created
    @email   yuquanjie13@gmail.com
    @descrip capture image randomly from full size, eg. 2400 * 3200 to small size,
             eg.1200 * 1600
"""
import numpy as np
import cv2
import tools.get_data as get_data
import tools.mydraw as mydraw
import random as rd
import os
import string
from shapely.geometry import Polygon


def intersect_cropped_rawtxtreg(crop_img_poly, raw_img_txtregion, tl_x, tl_y):
    """
    if the cropped image polygon intersected with raw text region
    1) stardard quardrange, has 4 point
    2) intersected aera / cropped image area is between 10% ~ 88%
    3) top-left in on the right position
    :param crop_img_poly: cropped image polygon(shapely.geometry.Polygon's object)
    :param raw_img_txtregion: raw image's text region
    :param tl_x: cropped image's top-left x coordinates on raw image
    :param tl_y:
    :return: true or false
    """
    writ_crop_img = True
    intersec_coord = []
    for polygon in raw_img_txtregion:
        x1, y1 = string.atof(polygon[0]), string.atof(polygon[1])
        x2, y2 = string.atof(polygon[2]), string.atof(polygon[3])
        x3, y3 = string.atof(polygon[4]), string.atof(polygon[5])
        x4, y4 = string.atof(polygon[6]), string.atof(polygon[7])
        raw_img_poly = Polygon([(x1, y1), (x4, y4), (x3, y3), (x2, y2)])
        if raw_img_poly.intersects(crop_img_poly):
            inter_poly = raw_img_poly.intersection(crop_img_poly)
            # the intersected quardrangle's aera is
            # 1) not equal to 0
            # 2) greater than 2.4%  1 / (320 * 320)
            # 3) smaller than 71% 72900 / (320 * 320)
            if inter_poly.area == 0 or inter_poly.area < 2500 or inter_poly.area > 72900:
                writ_crop_img = False
                break
            list_inter = list(inter_poly.exterior.coords)
            x1, y1 = list_inter[0][0] - tl_x, list_inter[0][1] - tl_y
            x2, y2 = list_inter[3][0] - tl_x, list_inter[3][1] - tl_y
            x3, y3 = list_inter[2][0] - tl_x, list_inter[2][1] - tl_y
            x4, y4 = list_inter[1][0] - tl_x, list_inter[1][1] - tl_y
            # insure the t_l coordinates is on the top-left position
            if x1 < x2 and y1 < y4 and x3 > x4 and y3 > y2:
                writ_crop_img = True
            else:
                writ_crop_img = False
                break
            # insure the intersected poly is quardrangle
            if len(list_inter) != 5:
                writ_crop_img = False
                break
            strcoord = '{0},{1},{2},{3},{4},{5},{6},{7},\n'.format(x1, y1, x2, y2, x3, y3, x4, y4)
            intersec_coord.append(strcoord)
    return writ_crop_img, intersec_coord


def capture_image_random(imgs, out_dir, size=320, cropped_num=300):
    """
    randomly choose a point as cropped image's top left coordinate, then each aixs add 320 pixel to generate
    down right coordinate, then get a 320 * 320 cropped image
    :param imgs: a list, each elements is a dictionary, including image's txt file path, and text region coordinates
    :param  out_dir: the directroy to story cropped image
    :param size: cropped iamge size
    :param cropped_num: the number of circulation to randomly crop iamge
    :return: no returning, just writing jpg and txt fix in out_dir
    """
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        print img['imagePath']
        if im.shape[0] < size or im.shape[1] < size:
            continue
        for i in xrange(cropped_num):
            tl_x = rd.randint(0, im.shape[1] - size)
            tl_y = rd.randint(0, im.shape[0] - size)
            basename = os.path.basename(img['imagePath'])
            image_name = basename.split('.')[0]
            jpgname = out_dir + '/' + image_name + '_' + bytes(i) + '.jpg'
            txtname = out_dir + '/' + image_name + '_' + bytes(i) + '.txt'
            crop_img = Polygon([(tl_x, tl_y), (tl_x, tl_y + size), (tl_x + size, tl_y + size), (tl_x + size, tl_y)])
            # use function to judge polygon crop_img whether intersect with raw image text region
            writ_crop_img, intersec_coord = intersect_cropped_rawtxtreg(crop_img, img['boxCoord'], tl_x, tl_y)
            if writ_crop_img:
                txtwrite = open(txtname, 'a')
                for coord in intersec_coord:
                    txtwrite.write(coord)
                txtwrite.close()
                cv2.imwrite(jpgname, im[tl_y: tl_y + size, tl_x: tl_x + size])


def get_croppedimg_tf_dr_coord(im, c, size=1000):
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
        t_l = [0, 0]
        d_r = [size, size]
    elif c[1] - size/2 < 0 and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        t_l = [c[0] - size/2, 0]
        d_r = [c[0] + size/2, size]
    elif c[0] - size/2 < 0 and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        t_l = [0, c[1] - size/2]
        d_r = [size, c[1] + size/2]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        t_l = [im.shape[1] - size, 0]
        d_r = [im.shape[1], size]
    elif c[0] + size/2 > im.shape[1] and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        t_l = [im.shape[1] - size, c[1] - size/2]
        d_r = [im.shape[1], c[1] + size/2]
    elif c[0] - size/2 < 0 and c[1] + size/2 > im.shape[0]:
        t_l = [0, im.shape[0] - 1000]
        d_r = [1000, im.shape[0]]
    elif c[1] + size/2 > im.shape[0] and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        t_l = [c[0] - size/2, im.shape[0] - size]
        d_r = [c[0] + size/2, im.shape[0]]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        t_l = [im.shape[1] - size, im.shape[0] - size]
        d_r = [im.shape[1], im.shape[0]]
    else:
        t_l = [c[0] - size/2, c[1] - size/2]
        d_r = [c[0] + size/2, c[1] + size/2]

    if vis:
        mydraw.draw_rectangle_image(im, t_l, d_r, "blue")
    return t_l, d_r


def get_croppedimg_tl_dr_coord_ver2(im, c, size=320):
    """
    crop image according c[0], c[1] and insure c[0], c[1] is the center of the cropped image,
    we ingore other cases
    :param im:
    :param c:
    :param size:
    :return: Return the cropped image's top-left and down-right coordinate
    """
    if c[0] - size/2 > 0 and c[1] - size/2 > 0 and c[0] + size/2 < im.shape[1] and c[1] + size/2 < im.shape[0]:
        t_l = [c[0] - size/2, c[1] - size/2]
        d_r = [c[0] + size/2, c[1] + size/2]
        return t_l, d_r
    else:
        return None, None


def cal_rotated_coord(co_line, t_m, cx, cy, w, h):
    """
    calculate text region's 4 corner's coordinates rotate (90, 180, 270)
    :param co_line:
    :param t_m: transfer matrix
    :param cx:
    :param cy:
    :param w:
    :param h:
    :return: A string which include 4 corner's rotated coordinates
    """
    coord = co_line.split(',')
    x1, y1 = string.atof(coord[0]), string.atof(coord[1])
    x2, y2 = string.atof(coord[2]), string.atof(coord[3])
    x3, y3 = string.atof(coord[4]), string.atof(coord[5])
    x4, y4 = string.atof(coord[6]), string.atof(coord[7])

    text_reg = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    rotated_text_reg = list(text_reg)
    for i, coord in enumerate(text_reg):
        # Grab the rotation components of the matrix)
        cos = np.abs(t_m[0, 0])
        sin = np.abs(t_m[0, 1])
        # compute the new bounding dimensions of the image
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        t_m[0, 2] += (nw / 2) - cx
        t_m[1, 2] += (nh / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0], coord[1], 1]
        # Perform the actual rotation and return the image
        calculated = np.dot(t_m, v)
        rotated_text_reg[i] = (calculated[0], calculated[1])
    strcoord = '{0},{1},{2},{3},{4},{5},{6},{7},\n'.format(rotated_text_reg[0][0], rotated_text_reg[0][1],
                                                           rotated_text_reg[1][0], rotated_text_reg[1][1],
                                                           rotated_text_reg[2][0], rotated_text_reg[2][1],
                                                           rotated_text_reg[3][0], rotated_text_reg[3][1])
    return strcoord


def rotate_img(cropped_img, outdir, base_name, idx, inter_co):

    angle_list = [90, 180, 270]
    for angle in angle_list:
        jpgname_90 = outdir + '/' + base_name + '_' + bytes(idx) + '_' + bytes(angle) + '.jpg'
        txtname_90 = outdir + '/' + base_name + '_' + bytes(idx) + '_' + bytes(angle) + '.txt'
        (h, w) = (cropped_img.shape[1], cropped_img.shape[0])
        center = (w / 2, h / 2)
        tran_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated90 = cv2.warpAffine(cropped_img, tran_matrix, (w, h))
        rotated90_coord = []
        for co_line in inter_co:
            rotated_line = cal_rotated_coord(co_line, tran_matrix, center[0], center[1], w, h)
            rotated90_coord.append(rotated_line)
        txtwrite = open(txtname_90, 'a')
        for ite in rotated90_coord:
            txtwrite.write(ite)
        txtwrite.close()

        cv2.imwrite(jpgname_90, rotated90)


def crop_image_from_textcenter(imgs, outdir, size=320):
    """
    according to text region center, crop raw image to generate 320 * 320 image, and save croppe image's text region
    :param imgs: a list(array), each elements is a dictionary
    :param outdir: output directory name
    :param size: cropped image's size
    :return: no returning value, write image(320 * 320) and corresponding txt file on disk
    """
    for img in imgs:
        idx = 1
        im = cv2.imread(img['imagePath'])
        print img['imagePath']
        if im.shape[0] < size and im.shape[1] < size:
            continue
        for coord in img['boxCoord']:
            # calculate the text region's center coordinate
            text_cen = [(string.atof(coord[0]) + string.atof(coord[4])) / 2,
                        (string.atof(coord[1]) + string.atof(coord[5])) / 2]
            # get cropped image's top-left and down-right coordinate
            [t_l, d_r] = get_croppedimg_tl_dr_coord_ver2(im, text_cen, size)
            if t_l is None or d_r is None:
                continue
            # calculate cropped image's top-right and down-left coordinates
            t_r = [d_r[0], t_l[1]]
            d_l = [t_l[0], d_r[1]]
            # using shapely lib define a cropped image's polygon
            crop_img = Polygon([(t_l[0], t_l[1]), (d_l[0], d_l[1]), (d_r[0], d_r[1]), (t_r[0], t_r[1])])
            base_name = img['imagePath'].split('/')[-1].split('.')[0]
            jpgname = outdir + '/' + base_name + '_' + bytes(idx) + '.jpg'
            txtname = outdir + '/' + base_name + '_' + bytes(idx) + '.txt'
            # use function to judge cropped image whether intersect with raw image's text regions
            # generate cropped image's txt file
            write_bool, inter_co = intersect_cropped_rawtxtreg(crop_img, img['boxCoord'], t_l[0], t_l[1])
            if write_bool:
                txtwrite = open(txtname, 'a')
                for co in inter_co:
                    txtwrite.write(co)
                txtwrite.close()
                cropped_img = im[int(t_l[1]): int(t_l[1]) + size, int(t_l[0]): int(t_l[0]) + size]
                cv2.imwrite(jpgname, cropped_img)
                rotate = True
                if rotate:
                    rotate_img(cropped_img, outdir, base_name, idx, inter_co)
                idx += 1


if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/Dataset/icdar/icdar')
    crop_image_from_textcenter(all_imgs, '/home/yuquanjie/Documents/Dataset/icdar/crop_center_rotated', 320)

