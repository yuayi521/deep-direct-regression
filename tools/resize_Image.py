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
import os
import string
from shapely.geometry import Polygon


def intersect_cropped_rawtxtreg(crop_img, rawtext_coord, tl_x, tl_y):
    """
    if the cropped image polygon intersected with raw text region
    1) stardard quardrange, has 4 point
    2) intersected aera / cropped image area is between 10% ~ 88%
    3) top-left in on the right position
    :param crop_img: cropped image polygon
    :param rawtext_coord:
    :param tl_x: cropped image top-left x coordinates on raw image
    :param tl_y:
    :return: true or false
    """
    writ_crop_img = True
    intersec_coord = []
    for polygon in rawtext_coord:
        x1, y1 = string.atof(polygon[0]), string.atof(polygon[1])
        x2, y2 = string.atof(polygon[2]), string.atof(polygon[3])
        x3, y3 = string.atof(polygon[4]), string.atof(polygon[5])
        x4, y4 = string.atof(polygon[6]), string.atof(polygon[7])
        raw_img_poly = Polygon([(x1, y1), (x4, y4), (x3, y3), (x2, y2)])
        if raw_img_poly.intersects(crop_img):
            inter = raw_img_poly.intersection(crop_img)
            # the intersected quardrangle's aera is
            # 1) not equal to 0
            # 2) greater than 0.00001%  1 / (320 * 320)
            # 3) smaller than 71% 72900 / (320 * 320)
            if inter.area == 0 or inter.area < 1 or inter.area > 72900:
                writ_crop_img = False
                break
            list_inter = list(inter.exterior.coords)
            x1, y1 = list_inter[0][0] - tl_x, list_inter[0][1] - tl_y
            x2, y2 = list_inter[3][0] - tl_x, list_inter[3][1] - tl_y
            x3, y3 = list_inter[2][0] - tl_x, list_inter[2][1] - tl_y
            x4, y4 = list_inter[1][0] - tl_x, list_inter[1][1] - tl_y
            # insure the t_ft coordinates is on the top-left position
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
        # else:
        #     writ_crop_img = False
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
        t_ft = [0, 0]
        down_rig = [size, size]
    elif c[1] - size/2 < 0 and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        t_ft = [c[0] - size/2, 0]
        down_rig = [c[0] + size/2, size]
    elif c[0] - size/2 < 0 and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        t_ft = [0, c[1] - size/2]
        down_rig = [size, c[1] + size/2]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        t_ft = [im.shape[1] - size, 0]
        down_rig = [im.shape[1], size]
    elif c[0] + size/2 > im.shape[1] and (c[1] - size/2 > 0 and c[1] + size/2 < im.shape[0]):
        t_ft = [im.shape[1] - size, c[1] - size/2]
        down_rig = [im.shape[1], c[1] + size/2]
    elif c[0] - size/2 < 0 and c[1] + size/2 > im.shape[0]:
        t_ft = [0, im.shape[0] - 1000]
        down_rig = [1000, im.shape[0]]
    elif c[1] + size/2 > im.shape[0] and (c[0] - size/2 > 0 and c[0] + size/2 < im.shape[1]):
        t_ft = [c[0] - size/2, im.shape[0] - size]
        down_rig = [c[0] + size/2, im.shape[0]]
    elif c[0] + size/2 > im.shape[1] and c[1] + size/2 > im.shape[0]:
        t_ft = [im.shape[1] - size, im.shape[0] - size]
        down_rig = [im.shape[1], im.shape[0]]
    else:
        t_ft = [c[0] - size/2, c[1] - size/2]
        down_rig = [c[0] + size/2, c[1] + size/2]

    if vis:
        mydraw.draw_rectangle_image(im, t_ft, down_rig, "blue")
    return t_ft, down_rig


def get_croppedimg_tf_dr_coord_onecase(im, c, size=320):
    """
    crop image according c[0], c[1] and insure c[0], c[1] is the center of the cropped image,
    :param im:
    :param c:
    :param size:
    :return: Return the cropped image's top-left and down-right coordinate
    """
    if c[0] - size/2 > 0 and c[1] - size/2 > 0 and c[0] + size/2 < im.shape[1] and c[1] + size/2 < im.shape[0]:
        t_ft = [c[0] - size/2, c[1] - size/2]
        down_rig = [c[0] + size/2, c[1] + size/2]
        return t_ft, down_rig
    else:
        return None, None


def resize_image_from_textcenter(imgs, outdir, size=320):
    """
    according to text region center, generate  320 * 320 images
    :param imgs: a list(array), each elements is a dictionary
    :param outdir: output directory name
    :param size: resized size
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
            [t_f, d_r] = get_croppedimg_tf_dr_coord_onecase(im, text_cen, size)
            if t_f is None or d_r is None:
                continue
            # calculate cropped image's top-right and down-left coordinates
            t_r = [d_r[0], t_f[1]]
            d_l = [t_f[0], d_r[1]]
            # using shapely lib define a cropped image's polygon
            crop_img = Polygon([(t_f[0], t_f[1]), (d_l[0], d_l[1]), (d_r[0], d_r[1]), (t_r[0], t_r[1])])
            base_name = img['imagePath'].split('/')[-1].split('.')[0]
            jpgname = outdir + '/' + base_name + '_' + bytes(idx) + '.jpg'
            txtname = outdir + '/' + base_name + '_' + bytes(idx) + '.txt'
            # use function to judge polygon crop_img whether intersect with raw image text regions
            write_b, inter_co = intersect_cropped_rawtxtreg(crop_img, img['boxCoord'], t_f[0], t_f[1])
            if write_b:
                txtwrite = open(txtname, 'a')
                for co in inter_co:
                    txtwrite.write(co)
                txtwrite.close()
                cv2.imwrite(jpgname, im[int(t_f[1]): int(t_f[1]) + size, int(t_f[0]): int(t_f[0]) + size])
                idx += 1


if __name__ == '__main__':
    # all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/shumei_train/shum')
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017_dataset/val')
    resize_image_from_textcenter(all_imgs, '/home/yuquanjie/Documents/icdar2017_crop_center_test', 320)
    # capture_image_random(all_imgs, '/home/yuquanjie/Documents/shumei_crop', 320, 2200 / 2)
    # capture_image_random(all_imgs, '/home/yuquanjie/Documents/shumei_crop', 320, 2200 / 10)

