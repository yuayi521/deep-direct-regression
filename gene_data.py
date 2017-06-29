'''
    @author  jasonYu
    @date    2017/6/3
    @version created
    @email   yuquanjie13@gmail.com
'''

import string
import re
import numpy as np
import cv2
import os
import h5py
from PIL import Image, ImageDraw, ImageFont
import tools.get_data as get_data
import tools.point_check as point_check
import sys
import time
fnt = ImageFont.truetype('/home/yuquanjie/Download/FreeMono.ttf', size=35)

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

'''
1. generate intput data
    1.1 input image resizing, from (2400,3200) to (320,320)
    1.2 (TODO) zero-center by mean pixel??
    1.3 preprocess image
2. generate output data
    2.1 generate classification output data
    2.2 generate regression output data
'''


def get_train_data(all_img):
    visulise = False
    num_text_pixel = 0.0
    num_not_text_pixel = 0.0
    while True:
        for img_data in all_img:
            imagename = img_data['imagePath']
            pattern = re.compile('image/')
            txtname = pattern.sub('text/', imagename)  # ..../image/image_1000.jpg  => ..../text/image_1000.jpg
            pattern = re.compile('jpg')
            txtname = pattern.sub('txt', txtname)  # ..../text/image_1000.jpg  => ..../text/image_1000.txt
            # insure the jpg file is not empty, and insure the .txt and .jpg file both exist
            if os.path.isfile(imagename) and os.path.isfile(txtname) and os.path.getsize(imagename):
                # print imagename
                img = cv2.imread(imagename)
                # 1)generate input data, input image, from (1000,1000) to (320,320)
                img_320 = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
                # input image dimension is (1, 320, 320, 3)
                img_320 = np.expand_dims(img_320, axis=0)
                # 2)generate classification output data
                # x-axis and y-axis reduced scale
                reduced_x = float(img.shape[1]) / 80.0
                reduced_y = float(img.shape[0]) / 80.0
                # negative lable is -1
                y_class_lable = -1 * np.ones((80, 80))
                # y_class_lable = np.zeros((80, 80))
                for ix in xrange(y_class_lable.shape[0]):
                    for jy in xrange(y_class_lable.shape[1]):
                        for polygon in img_data['boxCoord']:
                            x1 = string.atof(polygon[0]) / reduced_x
                            x2 = string.atof(polygon[2]) / reduced_x
                            x3 = string.atof(polygon[4]) / reduced_x
                            x4 = string.atof(polygon[6]) / reduced_x

                            y1 = string.atof(polygon[1]) / reduced_y
                            y2 = string.atof(polygon[3]) / reduced_y
                            y3 = string.atof(polygon[5]) / reduced_y
                            y4 = string.atof(polygon[7]) / reduced_y

                            polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                            if point_check.point_in_polygon(ix, jy, polygon):
                                y_class_lable[ix][jy] = 1
                if visulise:
                    img_vis = cv2.imread(imagename)
                    img_80 = cv2.resize(img_vis, (80, 80), interpolation=cv2.INTER_CUBIC)
                    img_draw = Image.fromarray(cv2.cvtColor(img_80, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_draw)
                    for coord in img_data['boxCoord']:
                        draw.polygon([(float(coord[0]) / reduced_x, float(coord[1]) / reduced_y),
                                      (float(coord[2]) / reduced_x, float(coord[3]) / reduced_y),
                                      (float(coord[4]) / reduced_x, float(coord[5]) / reduced_y),
                                      (float(coord[6]) / reduced_x, float(coord[7]) / reduced_y)],
                                     outline="red", fill="blue")
                    img_draw = np.array(img_draw)
                    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                    cv2.imshow('img', img_draw)
                    # cv2.waitKey(0)

                # output classificaiton label is (1, 80, 80, 1)
                # calculate ones's locations before expand the dimension of y_class_label
                one_locs = np.where(y_class_lable > 0)

                y_class_lable = np.expand_dims(y_class_lable, axis=0)
                y_class_lable = np.expand_dims(y_class_lable, axis=3)

                # statistic number of text region and non-text region
                num_text_pixel += len(one_locs[0])
                num_not_text_pixel += 6400 - len(one_locs[0])
                # print '(text pixel / 6400) : {0:.2f}%'.format(len(one_locs[0]) / 6400.0 * 100)

                # 3)generate regression output data
                y_regr_lable = np.zeros((80, 80, 8))
                # visit all text pixel
                for idx in xrange(len(one_locs[0])):
                    # judge text pixel belong to which box
                    for polygon in img_data['boxCoord']:
                        x1 = string.atof(polygon[0]) / reduced_x
                        x2 = string.atof(polygon[2]) / reduced_x
                        x3 = string.atof(polygon[4]) / reduced_x
                        x4 = string.atof(polygon[6]) / reduced_x

                        y1 = string.atof(polygon[1]) / reduced_y
                        y2 = string.atof(polygon[3]) / reduced_y
                        y3 = string.atof(polygon[5]) / reduced_y
                        y4 = string.atof(polygon[7]) / reduced_y
                        # 80 * 80  size image's quardrangle
                        quard = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                        ix = one_locs[0][idx]
                        jy = one_locs[1][idx]
                        # (ix, jy) pixel belong to quardragle quard
                        if point_check.point_in_polygon(ix, jy, quard):
                            top_left_x = quard[0][0]
                            top_left_y = quard[0][1]
                            top_righ_x = quard[1][0]
                            top_righ_y = quard[1][1]

                            dow_righ_x = quard[2][0]
                            dow_righ_y = quard[2][1]
                            dow_left_x = quard[3][0]
                            dow_left_y = quard[3][1]

                            y_regr_lable[ix][jy][0] = top_left_x * 4 - ix * 4
                            y_regr_lable[ix][jy][1] = top_left_y * 4 - jy * 4
                            y_regr_lable[ix][jy][2] = top_righ_x * 4 - ix * 4
                            y_regr_lable[ix][jy][3] = top_righ_y * 4 - jy * 4

                            y_regr_lable[ix][jy][4] = dow_righ_x * 4 - ix * 4
                            y_regr_lable[ix][jy][5] = dow_righ_y * 4 - jy * 4
                            y_regr_lable[ix][jy][6] = dow_left_x * 4 - ix * 4
                            y_regr_lable[ix][jy][7] = dow_left_y * 4 - jy * 4
                if visulise:
                    img_regr = cv2.imread(imagename)
                    img_regr = cv2.resize(img_regr, (320, 320), interpolation=cv2.INTER_CUBIC)
                    img_draw = Image.fromarray(cv2.cvtColor(img_regr, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_draw)
                    for idx in xrange(len(one_locs[0])):
                        ix = one_locs[0][idx]
                        jy = one_locs[1][idx]
                        draw.polygon([(y_regr_lable[ix][jy][0] + ix * 4, y_regr_lable[ix][jy][1] + jy * 4),
                                      (y_regr_lable[ix][jy][2] + ix * 4, y_regr_lable[ix][jy][3] + jy * 4),
                                      (y_regr_lable[ix][jy][4] + ix * 4, y_regr_lable[ix][jy][5] + jy * 4),
                                      (y_regr_lable[ix][jy][6] + ix * 4, y_regr_lable[ix][jy][7] + jy * 4)],
                                     outline="black")
                    img_draw = np.array(img_draw)
                    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                    cv2.imshow('img', cv2.resize(img_draw, (800, 800)))
                    cv2.waitKey(0)

                y_regr_lable = np.expand_dims(y_regr_lable, axis=0)
                yield [np.copy(img_320), np.copy(y_class_lable), np.copy(y_regr_lable),
                       img_data, num_text_pixel, num_not_text_pixel]
            else:
                continue


if __name__ == '__main__':
    n_text_pixel = 0.0
    n_not_text_pixel = 0.0
    raw_data_path = sys.argv[1]
    # all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/cropped_image320/5')
    all_imgs, numFileTxt = get_data.get_raw_data(raw_data_path)
    # object of python generator
    data_gen_train = get_train_data(all_imgs)
    X_train_list = []
    Y_train_cls_list = []
    Y_train_mer_list = []
    start_time = time.time()
    print 'start time is {0}'.format(start_time)
    for i in range(3000):
        if i % 100 == 0:
            print 'has precessed {0}'.format(i)
        X_train_it, Y_train_cls_it, Y_train_reg_it, Z, num_text_pixel_it, num_not_text_pixel_it = data_gen_train.next()
        X_train_list.append(X_train_it)
        Y_train_cls_list.append(Y_train_cls_it)
        Y_train_mer_list.append(np.concatenate([Y_train_reg_it, Y_train_cls_it], axis=3))
        n_text_pixel += num_text_pixel_it
        n_not_text_pixel += num_not_text_pixel_it

    X_train = np.stack(X_train_list, axis=0)
    Y_train_cls = np.stack(Y_train_cls_list, axis=0)
    Y_train_mer = np.stack(Y_train_mer_list, axis=0)
    Y = [Y_train_cls, Y_train_mer]

    print 'input training data shape is {0}'.format(X_train.shape)
    print 'output training data shape is {0}'.format(Y_train_mer.shape)
    # print the percentage between text region and all region
    print 'text region percentage is {0:.2f}%'.format(n_text_pixel / (n_not_text_pixel + n_text_pixel) * 100)

    # wirte data
    # file_write = h5py.File('train_5.h5', 'w')
    h5filename = sys.argv[2]
    file_write = h5py.File(h5filename, 'w')
    file_write.create_dataset('X_train', data=X_train)
    file_write.create_dataset('Y_train_cls', data=Y_train_cls)
    file_write.create_dataset('Y_train_merge', data=Y_train_mer)
    file_write.close()
    print 'processing 0.2 million training data time is {0}'.format(time.time() - start_time)
