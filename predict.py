"""
    @author  jasonYu
    @date    2017/6/3
    @version created
    @email   yuquanjie13@gmail.com
"""
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras import optimizers
from keras.layers import Input
from keras.models import Model, load_model
import tensorflow as tf
import string
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
import tools.get_data as get_data
import tools.point_check as point_check
import re

HUBER_DELTA = 1.0
gpu_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def my_hinge(y_true, y_pred):
    """ Compute hinge loss for classification

    :param y_true: Ground truth for category,
                    negative is -1, positive is 1
                    tensor shape (?, 80, 80, 1)
    :param y_pred: Predicted category
                    tensor shape (?, 80, 80, 1)
    :return: hinge loss, tensor shape (1, )
    """
    sub_1 = tf.sign(0.5 - y_true)
    sub_2 = y_pred - y_true
    my_hinge_loss = tf.reduce_mean(tf.square(tf.maximum(0.0, sub_1 * sub_2)))
    return my_hinge_loss


def new_smooth(y_true, y_pred):
    """
    Compute regression loss, loss / batch_size
    :param y_true: ground truth of regression and classification
                    tensor shape (batch_size, 80, 80, 9)
                    (:, :, :, 0:8) is regression label
                    (:, :, :, 8) is classification label
    :param y_pred: predicted value of regression
    :return: every pixel loss, average loss of 8 feature map
             tensor shape(batch_size, 80, 80)
    """
    # expand dimension of y_true, from (batch_size, 80, 80, 9) to (batch_size, 80, 80, 16)
    sub = tf.expand_dims(y_true[:, :, :, 8], axis=3)
    for i in xrange(7):
        y_true = tf.concat([y_true, sub], axis=3)
    abs_val = tf.abs(y_true[:, :, :, 0:8] - y_pred)
    smooth = tf.where(tf.greater(1.0, abs_val),
                      0.5 * abs_val ** 2,
                      abs_val - 0.5)
    loss = tf.where(tf.greater(y_true[:, :, :, 8:16], 0),
                    smooth,
                    0.0 * smooth)
    # loss = tf.where(tf.greater(y_true[:, :, :, 8:16], 0),
    #               y_true[:, :, :, 8:16],
    #               0 * y_true[:, :, :, 8:16]) * smooth
    loss = tf.reduce_mean(loss, axis=-1)
    # loss_batch = loss / tf.to_float(tf.shape(y_true)[0])
    return loss


def multi_task(input_tensor=None, trainable=False):
    img_input = BatchNormalization()(input_tensor)

    # conv_1
    conv1_1 = Convolution2D(32, (5, 5), strides=(1, 1), padding='same',
                            activation='relu', name='conv1_1')(img_input)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # conv_2
    conv2_1 = Convolution2D(64, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv2_1')(pool1)
    conv2_2 = Convolution2D(64, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_2)

    # conv_3
    conv3_1 = Convolution2D(128, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv3_1')(pool2)
    conv3_2 = Convolution2D(128, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv3_2')(conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)
    pool3_for_fuse = Convolution2D(128, (1, 1), strides=(1, 1), padding='same',
                                   activation='relu', name='pool3_for_fuse')(pool3)

    # conv_4
    conv4_1 = Convolution2D(256, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv4_1')(pool3)
    conv4_2 = Convolution2D(256, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv4_2')(conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_2)
    pool4_for_fuse = Convolution2D(128, (1, 1), strides=(1, 1), padding='same',
                                   activation='relu', name='pool4_for_fuse')(pool4)

    # conv_5
    conv5_1 = Convolution2D(512, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv5_1')(pool4)
    conv5_2 = Convolution2D(512, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv5_2')(conv5_1)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(conv5_2)
    pool5_for_fuse = Convolution2D(128, (1, 1), strides=(1, 1), padding='same',
                                   activation='relu', name='pool5_for_fuse')(pool5)

    # conv_6
    conv6_1 = Convolution2D(512, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv6_1')(pool5)
    conv6_2 = Convolution2D(512, (3, 3), strides=(1, 1), padding='same',
                            activation='relu', name='conv6_2')(conv6_1)
    pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='pool6')(conv6_2)

    #
    conv7_1 = Convolution2D(128, (1, 1), strides=(1, 1), padding='same',
                            activation='relu', name='conv7_1')(pool6)

    upscore2 = Conv2DTranspose(filters=128, kernel_size=(2, 2),
                               strides=(2, 2), padding='valid', use_bias=False,
                               name='upscore2')(conv7_1)

    fuse_pool5 = add([upscore2, pool5_for_fuse])
    upscore4 = Conv2DTranspose(filters=128, kernel_size=(2, 2),
                               strides=(2, 2), padding='valid', use_bias=False,
                               name='upscore4')(fuse_pool5)
    fuse_pool4 = add([upscore4, pool4_for_fuse])

    upscore8 = Conv2DTranspose(filters=128, kernel_size=(2, 2),
                               strides=(2, 2), padding='valid', use_bias=False,
                               name='upscore8')(fuse_pool4)
    fuse_pool3 = add([upscore8, pool3_for_fuse])

    upscore16 = Conv2DTranspose(filters=128, kernel_size=(2, 2),
                                strides=(2, 2), padding='valid', use_bias=False,
                                name='upscore16')(fuse_pool3)
    ##########################################################################
    ####### shared layer
    ##########################################################################
    x_clas = Convolution2D(1, (1, 1), strides=(1, 1), padding='same', name='out_class')(upscore16)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu')(upscore16)
    x = Convolution2D(8, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
    x_regr = Lambda(lambda t: 800 * t - 400)(x)
    return [x_clas, x_regr, x]


def get_train_data(all_img):
    visulise = False
    num_text = 0.0
    num_no_text = 0.0
    while True:
        for img_data in all_img:
            print img_data['imagePath']
            # image file wheater corresponding to text fle
            annot = img_data['imagePath']
            strinfo = re.compile('image/')
            annot = strinfo.sub('text/', annot)
            strinfo = re.compile('jpg')
            annot = strinfo.sub('txt', annot)

            if os.path.isfile(img_data['imagePath']) and os.path.isfile(annot):
                img = cv2.imread(img_data['imagePath'])
                width = img.shape[0]  # 3200, shape 0 is height
                height = img.shape[1]  # 2400, shape 1 is width

                # 1)generate input data
                # 1.1)input image, from (2400,3200) to (320,320)
                img_320 = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)

                # 2)generate output data
                # 2.1)generate classification output data
                divi_x = float(height) / 80.0
                divi_y = float(width) / 80.0
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
                    if img_data['imagePath'] == '/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/' \
                                                'part1/image/image_0.jpg':
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
                # print len(one_locs[0]) / (6400.0 - len(one_locs[0]))
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
                if visulise and img_data['imagePath'] == '/home/yuquanjie/Documents/icdar2017rctw_train_v1.2' \
                                                         '/train/part1/image/image_100.jpg':
                    print y_regr_lable[59][75]
                    print y_regr_lable[59][76]
                    print y_regr_lable[59][77]
                y_regr_lable = np.expand_dims(y_regr_lable, axis=0)
                # img is raw image, size is 2400 * 3200
                yield np.copy(img_320), np.copy(y_class_lable), np.copy(y_regr_lable), np.copy(img), img_data
            else:
                continue


if __name__ == '__main__':
    # define Input
    img_input = Input((320, 320, 3))
    # define network
    multi = multi_task(img_input, trainable=True)
    # multask_model = Model(img_input, multi[0:2])
    multask_model = Model(img_input, multi[0:2])
    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9)
    # compile model
    # multask_model.compile(loss=[my_hinge, smoothL1], optimizer=sgd)
    multask_model.compile(loss=[my_hinge, new_smooth], optimizer=sgd)
    # train data, test model using train data
    # all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1')
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017_test')
    data_gen_train = get_train_data(all_imgs)
    while True:
        X, Y_cls, Y_regr, raw_img, img_data = data_gen_train.next()
        # load model
        final_model = load_model('model/2017-07-04-14-30-loss-decrease-92-0.19.hdf5',
                                 custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})
        # predict
        predict_all = final_model.predict_on_batch(1/255.0 * X)
        # 1) classification result
        predict_cls = predict_all[0]
        # reduce first and last dimension
        predict_cls = np.sum(predict_cls, axis=-1)
        predict_cls = np.sum(predict_cls, axis=0)
        # one_locs type is tuple, negative lable is -1
        one_locs = np.where(predict_cls > 0)
        # coord type is list
        # firstly, each pixel of 80 * 80 feature map multiply 4, get pixel classification on 320 * 320
        coord = [one_locs[0] * 4, one_locs[1] * 4]
        # seconly, each pixel of 320 * 320 multiply reduced scale, get pixel classification on 1000 * 1000
        # raw image's size (1000 * 1000) / (320 * 320), 1st dimensiom represent width reduced scale
        reduced_scale = [raw_img.shape[1] / 320.0, raw_img.shape[0] / 320.0]
        # coord represent the text's coordinates on raw image (1000 * 1000)
        coord = [coord[0] * reduced_scale[0], coord[1] * reduced_scale[1]]
        # 2) regression result
        predict_regr = predict_all[1]
        # reduce dimension from (1, 80, 80, 8) to (80, 80, 8)
        predict_regr = np.sum(predict_regr, axis=0)
        # x, y represent each text pixel's 8 coordiantes , non-text pixel doesn't have this
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        # predict_regr is 3 dimension, 1st and 2nd dimension range is 0-79, should use one_locs not coords
        # predict_regr[][][] * reduced_scale to get the coordinates on the raw image (1000 * 1000)
        for idx in xrange(len(one_locs[0])):
            x1.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][0] * reduced_scale[0])
            y1.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][1] * reduced_scale[1])
            x2.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][2] * reduced_scale[0])
            y2.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][3] * reduced_scale[1])
            x3.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][4] * reduced_scale[0])
            y3.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][5] * reduced_scale[1])
            x4.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][6] * reduced_scale[0])
            y4.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][7] * reduced_scale[1])

        img = cv2.imread(img_data['imagePath'])
        img_draw = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_draw)
        for i in xrange(len(one_locs[0])):
            # draw predicted text region in pixel level on raw image(1000 * 1000), and save image
            # use the coordinates is on the raw image(1000 * 1000)
            draw.text(([coord[0][i], coord[1][i]]), "O", "red")
            # draw regression parameters on iamge
            top_left = [coord[0][i] + x1[i], coord[1][i] + y1[i]]
            top_righ = [coord[0][i] + x2[i], coord[1][i] + y2[i]]
            dow_righ = [coord[0][i] + x3[i], coord[1][i] + y3[i]]
            dow_left = [coord[0][i] + x4[i], coord[1][i] + y4[i]]
            draw.polygon([(top_left[0], top_left[1]), (top_righ[0], top_righ[1]),
                          (dow_righ[0], dow_righ[1]), (dow_left[0], dow_left[1])], outline="black")
        img_draw = np.array(img_draw)
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

        # get image name excluding directory path using regular expression
        image_name = img_data['imagePath'].split('/')[-1]
        image_path = '/home/yuquanjie/Documents/deep-direct-regression/result/' + image_name
        cv2.imwrite(image_path, img_draw[0: img_draw.shape[0], 0: img_draw.shape[1]])




