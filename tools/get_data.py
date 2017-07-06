import os
import string
import numpy as np
import cv2
import re
import glob
from PIL import Image, ImageDraw, ImageFont
fnt = ImageFont.truetype('/home/yuquanjie/Download/FreeMono.ttf', size=35)


def save_groudtruth(im, coords, filename):
    """
    print text region on image and save image
    :param im: numpy.ndarray
    :param coords: coordinates of text region
    :param filename: image file path
    :return: save image on a directory
    """
    print 'Saving ground truth ......{0}'.format(filename)
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    for coord in coords:
        draw.polygon([(float(coord[0]), float(coord[1])), (float(coord[2]), float(coord[3])),
                      (float(coord[4]), float(coord[5])), (float(coord[6]), float(coord[7]))],
                     outline="red", fill="blue")
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    bname_excludepoint = filename.split('/')[-1].split('.')[0]
    image_path = '/home/yuquanjie/Documents/deep-direct-regression/result/' + bname_excludepoint + '_gt.jpg'
    cv2.imwrite(image_path, img_draw[0: img_draw.shape[0], 0: img_draw.shape[1]])


def get_zone(text_reg):
    """
    split text region to get positive zone and gray zone(not care region)
    should judge quardrange's short side
    :param text_reg:
    :return: A tuple (gray_zone, posi_zone)
    """
    posi_zone = []
    gray_zone = []
    for txt in text_reg:
        x1, y1, x2, y2 = txt[0], txt[1], txt[2], txt[3]
        x3, y3, x4, y4 = txt[4], txt[5], txt[6], txt[7]
        line_1_2_len = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        line_1_4_len = np.sqrt(np.square(x1 - x4) + np.square(y1 - y4))
        if line_1_2_len <= line_1_4_len:
            # short side is line_1_2
            mid_point_1_2 = [(x1 + x2) / 2, (y1 + y2) / 2]
            mid_point_m_1 = [(x1 + mid_point_1_2[0]) / 2, (y1 + mid_point_1_2[1]) / 2]
            mid_point_m_2 = [(x2 + mid_point_1_2[0]) / 2, (y2 + mid_point_1_2[1]) / 2]

            mid_point_3_4 = [(x3 + x4) / 2, (y3 + y4) / 2]
            mid_point_m_3 = [(x3 + mid_point_3_4[0]) / 2, (y3 + mid_point_3_4[1]) / 2]
            mid_point_m_4 = [(x4 + mid_point_3_4[0]) / 2, (y4 + mid_point_3_4[1]) / 2]

            gray_zone.append([x1, y1, mid_point_m_1[0], mid_point_m_1[1], mid_point_m_4[0], mid_point_m_4[1], x4, y4])
            gray_zone.append([mid_point_m_2[0], mid_point_m_2[1], x2, y2, x3, y3, mid_point_m_3[0], mid_point_m_3[1]])
            posi_zone.append([mid_point_m_1[0], mid_point_m_1[1], mid_point_m_2[0], mid_point_m_2[1],
                              mid_point_m_3[0], mid_point_m_3[1], mid_point_m_4[0], mid_point_m_4[1]])
        else:
            # short side is line_1_4
            mid_point_1_4 = [(x1 + x4) / 2, (y1 + y4) / 2]
            mid_point_m_1 = [(x1 + mid_point_1_4[0]) / 2, (y1 + mid_point_1_4[1]) / 2]
            mid_point_m_4 = [(x4 + mid_point_1_4[0]) / 2, (y4 + mid_point_1_4[1]) / 2]

            mid_point_2_3 = [(x2 + x3) / 2, (y2 + y3) / 2]
            mid_point_m_2 = [(x2 + mid_point_2_3[0]) / 2, (y2 + mid_point_2_3[1]) / 2]
            mid_point_m_3 = [(x3 + mid_point_2_3[0]) / 2, (y3 + mid_point_2_3[1]) / 2]
            gray_zone.append([x1, y1, x2, y2, mid_point_m_2[0], mid_point_m_2[1], mid_point_m_1[0], mid_point_m_1[1]])
            gray_zone.append([mid_point_m_4[0], mid_point_m_4[1], mid_point_m_3[0], mid_point_m_3[1], x3, y3, x4, y4])
            posi_zone.append([mid_point_m_1[0], mid_point_m_1[1], mid_point_m_2[0], mid_point_m_2[1],
                              mid_point_m_3[0], mid_point_m_3[1], mid_point_m_4[0], mid_point_m_4[1]])

    return gray_zone, posi_zone


def visualize(im, coords, filename):
    """

    :param im:
    :param coords:
    :param filename:
    :return:
    """
    print filename
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for coord in coords:
        draw = ImageDraw.Draw(img_draw)
        draw.polygon([(float(coord[0]), float(coord[1])), (float(coord[2]), float(coord[3])),
                      (float(coord[4]), float(coord[5])), (float(coord[6]), float(coord[7]))],
                     outline="red", fill="blue")
        draw.text([float(coord[0]), float(coord[1])], "1", font=fnt)
        draw.text([float(coord[2]), float(coord[3])], "2", font=fnt)
        draw.text([float(coord[4]), float(coord[5])], "3", font=fnt)
        draw.text([float(coord[6]), float(coord[7])], "4", font=fnt)
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(img_draw, (800, 800)))
    cv2.waitKey(0)


def get_raw_data(input_path, save_gt=False):
    """
    process txt file
    getting all text region's coordinates and number of text region
    :param input_path: a directory has text and image 2 directory
    :param save_gt: save gt in result/ directory
    :return: 1) a list, each element is a dictionary
             'imagePath'
             'boxCoord'
             'boxNum'
             2) number of the txt that have processed
    """
    # define variable for returning
    all_txts = []  # a list, each element is a dictionary
    coords = []  # a list, storing a image's all text region's coordinates which is clockwise
    num_txt = 0
    visual = False
    print('Parsing txt files')
    # txt_directory = os.path.join(input_path, 'text')
    # all_txt_files = [os.path.join(txt_directory, s) for s in os.listdir(txt_directory)]
    txtfiles = input_path + '/*.txt'
    all_txt_files = glob.glob(txtfiles)
    box_num = 0
    for txt in all_txt_files:
        with open(txt, 'r') as f:
            num_txt += 1
            for line in f:
                box_num += 1
                line_split = line.strip().split(',')
                # clockwise
                (x1, y1, x2, y2) = line_split[0:4]
                (x3, y3, x4, y4) = line_split[4:8]
                coords.append((x1, y1, x2, y2, x3, y3, x4, y4))
            txtfilepath = txt
            # using regular expression, get image file path
            # pattern = re.compile('text')
            # img_file_path = pattern.sub('image', txt)
            pattern = re.compile('txt')
            img_file_path = pattern.sub('jpg', txtfilepath)
            txt_data = {'imagePath': img_file_path, 'boxCoord': coords, 'boxNum': box_num}
            box_num = 0
            coords = []
            # image file wheater corresponding to text file, and image file is not empty then add
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path):
                all_txts.append(txt_data)
            # -----------------------visualizing-----------------------------------------
            # draw text region on image and save image
            # print text region on image for comparing gt and predicted results
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path) and save_gt:
                save_groudtruth(cv2.imread(img_file_path), txt_data['boxCoord'], img_file_path)

            # draw text region on image and show image
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path) and visual:
                visualize(cv2.imread(img_file_path), txt_data['boxCoord'], img_file_path)
                # -----------------------visualizing-----------------------------------------
    return all_txts, num_txt


if __name__ == '__main__':
    get_raw_data('/home/yuquanjie/Documents/icdar2017_crop_center', True)

    test_get_zone_function = False
    if test_get_zone_function:
        txts = glob.glob('/home/yuquanjie/Documents/icdar2017_crop_center/*.txt')
        for txtname in txts:
            text_region = []
            print txtname
            # jpgname = '/home/yuquanjie/Documents/icdar2017_crop_center/image_11_2.jpg'
            jpgname = '/home/yuquanjie/Documents/icdar2017_crop_center/' + txtname.split('/')[-1].split('.')[0] + '.jpg'
            print jpgname
            with open(txtname, 'r') as f:
                for line in f:
                    line_split = line.strip().split(',')
                    (x_1, y_1, x_2, y_2) = line_split[0:4]
                    (x_3, y_3, x_4, y_4) = line_split[4:8]
                    text_region.append([string.atof(x_1), string.atof(y_1), string.atof(x_2), string.atof(y_2),
                                        string.atof(x_3), string.atof(y_3), string.atof(x_4), string.atof(y_4)])
            gray_z, pos_zone = get_zone(text_region)
            img = cv2.imread(jpgname)
            #
            cv2.imshow('img', img)
            cv2.waitKey(0)
            # raw text region
            for reg in text_region:
                counters = np.array([[int(reg[0]), int(reg[1])], [int(reg[2]), int(reg[3])],
                                     [int(reg[4]), int(reg[5])], [int(reg[6]), int(reg[7])]])
                cv2.fillPoly(img, pts=[counters], color=(0, 0, 255))
                # cv2.fillPoly(img, pts=[counters], color=(255, 0, 0))
            cv2.imshow('img', img)
            cv2.waitKey(0)
            # gray_zone
            for reg in gray_z:
                counters = np.array([[int(reg[0]), int(reg[1])], [int(reg[6]), int(reg[7])],
                                     [int(reg[4]), int(reg[5])], [int(reg[2]), int(reg[3])]])
                cv2.fillPoly(img, pts=[counters], color=(255, 0, 0))
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # positive zone
            for reg in pos_zone:
                counters = np.array([[int(reg[0]), int(reg[1])], [int(reg[6]), int(reg[7])],
                                     [int(reg[4]), int(reg[5])], [int(reg[2]), int(reg[3])]])
                cv2.fillPoly(img, pts=[counters], color=(0, 0, 0))
            cv2.imshow('img', img)
            cv2.waitKey(0)


