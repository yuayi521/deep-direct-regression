import os
import string
import numpy as np
import cv2
import re
import glob
import h5py
import tools.point_check as point_check
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


def test_get_zone():
    """

    :return:
    """
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


def image_generator_not_random(list_of_files, crop_size=320, scale=1):
    """
    a python generator, traversal all txt files and jpg files
    :param list_of_files: list, storing all jpg file path
    :param crop_size: cropped image size
    :param scale: normalization parameters
    :return: A list [numpy array, text region list]
    """
    while True:
        text_region = []
        for jpgname in list_of_files:
            print jpgname
            # jpgname = np.random.choice(list_of_files)
            img = cv2.imread(jpgname)
            pattern = re.compile('jpg')
            txtname = pattern.sub('txt', jpgname)
            if not os.path.isfile(txtname):
                continue
            cropped_image = img
            with open(txtname, 'r') as f:
                for line in f:
                    line_split = line.strip().split(',')
                    print line_split
                    # clockwise
                    (x1, y1, x2, y2) = line_split[0:4]
                    (x3, y3, x4, y4) = line_split[4:8]
                    text_region.append([string.atof(x1), string.atof(y1), string.atof(x2), string.atof(y2),
                                        string.atof(x3), string.atof(y3), string.atof(x4), string.atof(y4)])
            if cropped_image is None or text_region is None or \
                    cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
                continue
            yield [scale * cropped_image, text_region]


def image_output_pair(path, scale):
    """

    :param path:
    :param scale:
    :return:
    """
    all_images, num = get_raw_data(path)
    # print len(all_images)
    # for img, txtreg in images:
    for image in all_images:
        # print image['imagePath']
        img = cv2.imread(image['imagePath'])
        # 0) according to image path generate text_region is a list which element is aslo a list whose elemnt is a float
        txtreg = []
        patt = re.compile('jpg')
        txtpath = patt.sub('txt', image['imagePath'])
        with open(txtpath, 'r') as f:
            for line in f:
                line_split = line.strip().split(',')
                # print line_split
                # clockwise
                (x1, y1, x2, y2) = line_split[0:4]
                (x3, y3, x4, y4) = line_split[4:8]
                txtreg.append([string.atof(x1), string.atof(y1), string.atof(x2), string.atof(y2),
                               string.atof(x3), string.atof(y3), string.atof(x4), string.atof(y4)])
        # 1) generate imput data, input data is (320, 320, 3)
        # img *= scale
        # 2) generate clsssification data
        # split text region into gray_zone and posi_zone
        gray_zone, posi_zone = get_zone(txtreg)
        # x-axis and y-axis reduced scale
        reduced_x, reduced_y = float(img.shape[1]) / 80.0, float(img.shape[0]) / 80.0
        mask_label = np.ones((80, 80))
        # y_class_label = -1 * np.ones((80, 80))  # negative lable is -1
        y_class_label = np.zeros((80, 80))  # negative lable is 0
        for ix in xrange(y_class_label.shape[0]):
            for jy in xrange(y_class_label.shape[1]):
                for posi in posi_zone:
                    x1, x2 = posi[0] / reduced_x, posi[2] / reduced_x
                    x3, x4 = posi[4] / reduced_x, posi[6] / reduced_x
                    y1, y2 = posi[1] / reduced_y, posi[3] / reduced_y
                    y3, y4 = posi[5] / reduced_y, posi[7] / reduced_y
                    posi_poly = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    if point_check.point_in_polygon(ix, jy, posi_poly):
                        y_class_label[ix][jy] = 1
                for gray in gray_zone:
                    x1, x2 = gray[0] / reduced_x, gray[2] / reduced_x
                    x3, x4 = gray[4] / reduced_x, gray[6] / reduced_x
                    y1, y2 = gray[1] / reduced_y, gray[3] / reduced_y
                    y3, y4 = gray[5] / reduced_y, gray[7] / reduced_y
                    gray_poly = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    if point_check.point_in_polygon(ix, jy, gray_poly):
                        mask_label[ix][jy] = 0
        # calculate ones's locations before expand the dimension of y_class_label
        one_locs = np.where(y_class_label > 0)
        y_class_label = np.expand_dims(y_class_label, axis=-1)
        mask_label = np.expand_dims(mask_label, axis=-1)

        # 3) generate regression data
        y_regr_lable = np.zeros((80, 80, 8))
        # visit all text pixel
        for idx in xrange(len(one_locs[0])):
            # judge text pixel belong to which box
            for polygon in txtreg:
                x1, x2 = polygon[0] / reduced_x, polygon[2] / reduced_x
                x3, x4 = polygon[4] / reduced_x, polygon[6] / reduced_x
                y1, y2 = polygon[1] / reduced_y, polygon[3] / reduced_y
                y3, y4 = polygon[5] / reduced_y, polygon[7] / reduced_y
                # 80 * 80 image's quardrangle
                quard = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                ix = one_locs[0][idx]
                jy = one_locs[1][idx]
                # (ix, jy) pixel belong to quardragle quard
                if point_check.point_in_polygon(ix, jy, quard):
                    top_left_x, top_left_y = quard[0][0], quard[0][1]
                    top_righ_x, top_righ_y = quard[1][0], quard[1][1]
                    dow_righ_x, dow_righ_y = quard[2][0], quard[2][1]
                    dow_left_x, dow_left_y = quard[3][0], quard[3][1]

                    y_regr_lable[ix][jy][0] = top_left_x * 4 - ix * 4
                    y_regr_lable[ix][jy][1] = top_left_y * 4 - jy * 4
                    y_regr_lable[ix][jy][2] = top_righ_x * 4 - ix * 4
                    y_regr_lable[ix][jy][3] = top_righ_y * 4 - jy * 4
                    y_regr_lable[ix][jy][4] = dow_righ_x * 4 - ix * 4
                    y_regr_lable[ix][jy][5] = dow_righ_y * 4 - jy * 4
                    y_regr_lable[ix][jy][6] = dow_left_x * 4 - ix * 4
                    y_regr_lable[ix][jy][7] = dow_left_y * 4 - jy * 4
        y_regr_cls_mask_label = np.concatenate((y_regr_lable, y_class_label, mask_label), axis=-1)
        y_cls_mask_label = np.concatenate((y_class_label, mask_label), axis=-1)
        yield (scale * img, y_cls_mask_label, y_regr_cls_mask_label)


def gene_h5_train_file(data_path):
    """
    read training txt and image file, then generate y_cls_label, y_regr_label, mask_label, and write these label
    into h5 file
    :param data_path: jpg and txt file path
    :return: No return value, just write h5 file on disk
    """
    img = []
    y_cls_mask = []
    y_reg_cls_mask = []
    os.chdir(data_path)
    jpgfiles = glob.glob('*.jpg')
    idx = 1
    # the position of generator objector is very important
    gene_obj = image_output_pair(data_path, 1/255.0)
    while True:
        if idx == len(jpgfiles):
            break
        print '\t{0}/{1}'.format(idx, len(jpgfiles))
        # the position of generator objector is very important
        # gene_obj = image_output_pair(data_path, 1/255.0)
        img_it, y_cls_mask_it, y_reg_cls_mask_it = gene_obj.next()
        img.append(img_it)
        y_cls_mask.append(y_cls_mask_it)
        y_reg_cls_mask.append(y_reg_cls_mask_it)
        idx += 1

    # img => (320, 320, 3)
    # after np.stack => (19041, 320, 320, 3)
    img_input = np.stack(img, axis=0)
    y_cls = np.stack(y_cls_mask, axis=0)
    y_reg = np.stack(y_reg_cls_mask, axis=0)
    print 'input data shape is {0}'.format(img_input.shape)
    print 'y_cls data shape is {0}'.format(y_cls.shape)
    print 'y_reg data shape is {0}'.format(y_reg.shape)
    
    # wirte data
    file_write = h5py.File('train.h5', 'w')
    file_write.create_dataset('X_train', data=img_input)
    file_write.create_dataset('Y_train_cls', data=y_cls)
    file_write.create_dataset('Y_train_merge', data=y_reg)
    file_write.close()


def test_gene_h5_train_file():
    """

    :return:
    """
    gene_h5_train_file('/home/yuquanjie/Documents/icdar2017_crop_center')

if __name__ == '__main__':
    # get_raw_data('/home/yuquanjie/Documents/icdar2017_crop_center_test', True)

    test_get_zone_function = False
    if test_get_zone_function:
        test_get_zone()

    test_gene_h5_train_file_b = True
    if test_gene_h5_train_file_b:
        test_gene_h5_train_file()



