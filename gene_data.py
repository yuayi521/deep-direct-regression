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
from PIL import Image, ImageDraw
import tools.get_data as get_data
import tools.point_check as point_check

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


def get_train_data(all_imgs):
    visulise = False
    num_text = 0.0
    num_no_text = 0.0
    while True:
        for img_data in all_imgs:
            # image file wheater corresponding to text fle
            annot = img_data['imagePath']
            strinfo = re.compile('image/')
            annot = strinfo.sub('text/', annot)
            strinfo = re.compile('jpg')
            txtfilename = strinfo.sub('txt', annot)

            # insure the jpg file is not empty, image_1024_11.jpg is empty
            if os.path.isfile(img_data['imagePath']) and os.path.isfile(txtfilename) and \
                    os.path.getsize(img_data['imagePath']):
                print img_data['imagePath']
                img = cv2.imread(img_data['imagePath'])
                width = img.shape[0]  # 2400
                height = img.shape[1]  # 3200

                # 1)generate input data
                # 1.1)input image, from (2400,3200) to (320,320)
                img_320 = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)

                # 2)generate output data
                # 2.1)generate classification output data
                divi_x = float(height) / 80.0
                divi_y = float(width) / 80.0
                # y_class_lable = -1 * np.ones((80,80)).astype(np.float32)
                y_class_lable = -1 * np.ones((80, 80))
                # y_class_lable = np.zeros((80, 80))
                for ix in xrange(y_class_lable.shape[0]):
                    for jy in xrange(y_class_lable.shape[1]):
                        for polygon in img_data['boxCoord']:
                            x1 = string.atof(polygon[0]) / divi_x
                            x2 = string.atof(polygon[2]) / divi_x
                            x3 = string.atof(polygon[4]) / divi_x
                            x4 = string.atof(polygon[6]) / divi_x

                            y1 = string.atof(polygon[1]) / divi_y
                            y2 = string.atof(polygon[3]) / divi_y
                            y3 = string.atof(polygon[5]) / divi_y
                            y4 = string.atof(polygon[7]) / divi_y

                            polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                            if point_check.point_in_polygon(ix, jy, polygon):
                                y_class_lable[ix][jy] = 1
                                # else:
                                #    y_class_lable[ix][jy] = 0

                if visulise:
                    if True:
                        img = cv2.imread(img_data['imagePath'])
                        img_80 = cv2.resize(img, (80, 80), interpolation=cv2.INTER_CUBIC)
                        img_draw = Image.fromarray(cv2.cvtColor(img_80, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_draw)
                        for coord in img_data['boxCoord']:
                            print 'detail'
                            print float(coord[0]) / divi_x, float(coord[1]) / divi_y
                            print float(coord[2]) / divi_x, float(coord[3]) / divi_y
                            print float(coord[4]) / divi_x, float(coord[5]) / divi_y
                            print float(coord[6]) / divi_x, float(coord[7]) / divi_y
                            print 'detail'
                            draw.polygon([(float(coord[0]) / divi_x, float(coord[1]) / divi_y),
                                          (float(coord[2]) / divi_x, float(coord[3]) / divi_y),
                                          (float(coord[4]) / divi_x, float(coord[5]) / divi_y),
                                          (float(coord[6]) / divi_x, float(coord[7]) / divi_y)],
                                         outline="red", fill="blue")
                        img_draw = np.array(img_draw)
                        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                        one_locs = np.where(y_class_lable > 0)
                        print one_locs
                        print len(one_locs[0])
                        print img_data['imagePath']
                        print img_data['boxNum']
                        cv2.imshow('img', img_draw)
                        cv2.waitKey(0)

                img_320 = np.expand_dims(img_320, axis=0)
                y_class_lable = np.expand_dims(y_class_lable, axis=0)
                y_class_lable = np.expand_dims(y_class_lable, axis=3)

                # statistic number of text region and non-text region
                one_locs = np.where(y_class_lable > 0)
                # print len(one_locs[0])
                num_text += len(one_locs[0])
                num_no_text += 6400 - len(one_locs[0])
                print len(one_locs[0]) / (6400.0 - len(one_locs[0]))
                # 2.2)generate regression output data
                # y_regr_lable = np.zeros((80,80,8)).astype(np.float32)
                y_regr_lable = np.zeros((80, 80, 8))
                for i in xrange(len(one_locs[0])):
                    # get quadrilateral vertex 4 corrdinates
                    for polygon in img_data['boxCoord']:
                        x1 = string.atof(polygon[0]) / divi_x
                        x2 = string.atof(polygon[2]) / divi_x
                        x3 = string.atof(polygon[4]) / divi_x
                        x4 = string.atof(polygon[6]) / divi_x

                        y1 = string.atof(polygon[1]) / divi_y
                        y2 = string.atof(polygon[3]) / divi_y
                        y3 = string.atof(polygon[5]) / divi_y
                        y4 = string.atof(polygon[7]) / divi_y

                        poly = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                        ix = one_locs[0][i]
                        jy = one_locs[1][i]
                        if point_check.point_in_polygon(ix, jy, poly):
                            left_top_x = poly[0][0]
                            left_top_y = poly[0][1]
                            righ_top_x = poly[1][0]
                            righ_top_y = poly[1][1]

                            righ_dow_x = poly[2][0]
                            righ_dow_y = poly[2][1]
                            left_dow_x = poly[3][0]
                            left_dow_y = poly[3][1]

                            y_regr_lable[ix][jy][0] = left_top_x * 4 - ix * 4
                            y_regr_lable[ix][jy][1] = left_top_y * 4 - jy * 4
                            y_regr_lable[ix][jy][2] = righ_top_x * 4 - ix * 4
                            y_regr_lable[ix][jy][3] = righ_top_y * 4 - jy * 4

                            y_regr_lable[ix][jy][4] = righ_dow_x * 4 - ix * 4
                            y_regr_lable[ix][jy][5] = righ_dow_y * 4 - jy * 4
                            y_regr_lable[ix][jy][6] = left_dow_x * 4 - ix * 4
                            y_regr_lable[ix][jy][7] = left_dow_y * 4 - jy * 4
                if visulise:
                    print y_regr_lable[59][75]
                    print y_regr_lable[59][76]
                    print y_regr_lable[59][77]
                y_regr_lable = np.expand_dims(y_regr_lable, axis=0)
                yield np.copy(img_320), np.copy(y_class_lable), np.copy(y_regr_lable), img_data, num_text, num_no_text
            else:
                continue


if __name__ == '__main__':
    num_text = 0.0
    num_no_text = 0.0
    # all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1')
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/deep-direct-regression/captured_data')
    data_gen_train = get_train_data(all_imgs)
    X_train, Y_train_cls, Y_train_regr, Z, num_text_it, num_no_text_it = data_gen_train.next()
    num_text += num_text_it
    num_no_text += num_no_text_it
    Y_train_merge = np.concatenate([Y_train_regr, Y_train_cls], axis=3)
    for i in range(1500):
        X_train_iter, Y_train_cls_iter, Y_train_regr_iter, Z, num_text_it, num_no_text_it = data_gen_train.next()

        num_text += num_text_it
        num_no_text += num_no_text_it

        X_train = np.concatenate([X_train, X_train_iter], axis=0)
        Y_train_cls = np.concatenate([Y_train_cls, Y_train_cls_iter], axis=0)

        Y_train_merge_iter = np.concatenate([Y_train_regr_iter, Y_train_cls_iter], axis=3)
        Y_train_merge = np.concatenate([Y_train_merge, Y_train_merge_iter], axis=0)
    Y = [Y_train_cls, Y_train_merge]
    print X_train.shape
    print Y_train_cls.shape
    print Y_train_merge.shape
    # print text region divide non-text region
    print 'text divide non-text region is:'
    print num_text / num_no_text
    # wirte data
    file_read = h5py.File('train_dataset-1500.h5', 'w')
    file_read.create_dataset('X_train', data=X_train)
    file_read.create_dataset('Y_train_cls', data=Y_train_cls)
    file_read.create_dataset('Y_train_merge', data=Y_train_merge)
    file_read.close()
