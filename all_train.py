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
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Input
from keras.models import Model, load_model
from keras.preprocessing.image import list_pictures
from tools.get_data import get_zone
import tools.point_check as point_check
import cv2
import string
import numpy as np
import os
import re
import h5py
import tensorflow as tf
import datetime


def tf_count(t, val):
    """
    https://stackoverflow.com/questions/36530944/how-to-get-the-count-of-an-element-in-a-tensor
    :param t:
    :param val:
    :return:
    """
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count


def read_multi_h5file(filelist):
    """
    read multi h5 file
    :param filelist:
    :return: network input X and output Y
    """
    read = h5py.File(filelist[0], 'r')
    x_train = read['X_train'][:]
    y_1_cls = read['Y_train_cls'][:]
    y_2_mer = read['Y_train_merge'][:]
    read.close()

    for idx in range(1, len(filelist)):
        read = h5py.File(filelist[idx], 'r')
        x_ite = read['X_train'][:]
        y_1_cls_ite = read['Y_train_cls'][:]
        y_2_mer_ite = read['Y_train_merge'][:]
        read.close()
        x_train = np.concatenate((x_train, x_ite))
        y_1_cls = np.concatenate((y_1_cls, y_1_cls_ite))
        y_2_mer = np.concatenate((y_2_mer, y_2_mer_ite))

    y_train = [y_1_cls, y_2_mer]
    return x_train, y_train


def l2(y_true, y_pred):
    """
    L2 loss, not divide batch size
    :param y_true: Ground truth for category, negative is 0, positive is 1
                   tensor shape (?, 80, 80, 2)
                   (?, 80, 80, 0): classification label
                   (?, 80, 80, 1): mask label,
                                   0 represent margin between pisitive and negative region, not contribute to loss
                                   1 represent positive and negative region
    :param y_pred:
    :return: A tensor (1, ) total loss of a batch / all contributed pixel
    """
    # extract mask label
    mask_label = tf.expand_dims(y_true[:, :, :, 1], axis=-1)
    # count the number of 1 in mask_label tensor, number of contributed pixels(for each output feature map in batch)
    num_contributed_pixel = tf_count(mask_label, 1)
    # extract classification label
    clas_label = tf.expand_dims(y_true[:, :, :, 0], axis=-1)
    # int32 to flot 32
    num_contributed_pixel = tf.cast(num_contributed_pixel, tf.float32)

    loss = tf.reduce_sum(tf.multiply(mask_label, tf.square(clas_label - y_pred))) / num_contributed_pixel
    # divide batch_size
    # loss = loss / tf.to_float(tf.shape(y_true)[0])
    return loss


def my_hinge(y_true, y_pred):
    """
    Compute hinge loss for classification, return batch loss, not divide batch_size
    :param y_true: Ground truth for category, negative is 0, positive is 1
                   tensor shape (?, 80, 80, 2)
                   (?, 80, 80, 0): classification label
                   (?, 80, 80, 1): mask label,
                                   0 represent margin between pisitive and negative region, not contribute tot loss
                                   1 represent positive and negative region
    :param y_pred:
    :return: tensor shape (1, ), batch total loss / contirbuted pixels
    """
    # extract mask label
    mask_label = tf.expand_dims(y_true[:, :, :, 1], axis=-1)
    # count the number of 1 in mask_label tensor, the number of contributed pixels
    num_contributed_pixel = tf_count(mask_label, 1)
    # extract classification label
    clas_label = tf.expand_dims(y_true[:, :, :, 0], axis=-1)
    # int32 to flot 32
    num_contributed_pixel = tf.cast(num_contributed_pixel, tf.float32)

    exper_1 = tf.sign(0.5 - clas_label)
    exper_2 = y_pred - clas_label
    loss_mask = tf.multiply(mask_label, tf.square(tf.maximum(0.0, exper_1 * exper_2)))

    # sum over all axis, and reduce all dimensions
    loss = tf.reduce_sum(loss_mask) / num_contributed_pixel
    # divide batch_size
    # loss = loss / tf.to_float(tf.shape(y_true)[0])
    return loss


def new_smooth(y_true, y_pred):
    """
    Compute regression loss
    :param y_true: ground truth of regression and classification
                   tensor shape (batch_size, 80, 80, 10)
                   (:, :, :, 0:8) is regression label
                   (:, :, :, 8) is classification label
                   (:, :, :, 9) is mask label
    :param y_pred:
    :return: every pixel loss, average loss of 8 feature map
             tensor shape(batch_size, 80, 80)
    """
    # extract classification label and mask label
    cls_label = tf.expand_dims(y_true[:, :, :, 8], axis=-1)
    mask_label = tf.expand_dims(y_true[:, :, :, 9], axis=-1)
    num_contibuted_pixel = tf.cast(tf_count(mask_label, 1), tf.float32)
    expanded_mask_label = tf.expand_dims(y_true[:, :, :, 9], axis=-1)
    # expand dimension of y_true, from (batch_size, 80, 80, 9) to (batch_size, 80, 80, 16)
    for i in xrange(7):
        y_true = tf.concat([y_true, cls_label], axis=-1)
    # expand dimension of mask label to make it equal to y_pred
    for i in xrange(7):
        expanded_mask_label = tf.concat([expanded_mask_label, mask_label], axis=-1)

    abs_val = tf.abs(y_true[:, :, :, 0:8] - y_pred)
    smooth = tf.where(tf.greater(1.0, abs_val),
                      0.5 * abs_val ** 2,
                      abs_val - 0.5)
    loss = tf.where(tf.greater(y_true[:, :, :, 8:16], 0),
                    smooth,
                    0.0 * smooth)
    loss = tf.multiply(loss, expanded_mask_label)
    # firstly, for a  pixel (x_i, y_i), summing 8 channel's loss, then calculating average loss
    loss = tf.reduce_mean(loss, axis=-1)
    # secondly, sum all dimension loss, then divied number of contributed pixel
    loss = tf.reduce_sum(loss) / num_contibuted_pixel
    # thirdly, divide batch_size
    # loss = loss / tf.to_float(tf.shape(y_true)[0])
    # lambda_loc = 0.01
    lambda_loc = 1
    return lambda_loc * loss


def multi_task(input_tensor):
    im_input = BatchNormalization()(input_tensor)

    # conv_1
    conv1_1 = Convolution2D(32, (5, 5), strides=(1, 1), padding='same',
                            activation='relu', name='conv1_1')(im_input)
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
    # shared layer
    ##########################################################################
    x_clas = Convolution2D(1, (1, 1), strides=(1, 1), padding='same', name='cls')(upscore16)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu')(upscore16)
    x = Convolution2D(8, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
    x_regr = Lambda(lambda t: 800 * t - 400)(x)
    return [x_clas, x_regr, x]
    # merged = np.concatenate((x_regr, x_clas), axis=-1)
    # return merged


def image_generator(list_of_files, crop_size=320, scale=1):
    """
    a python generator, read image's text region from txt file
    :param list_of_files: list, storing all jpg file path
    :param crop_size: cropped image size
    :param scale: normalization parameters
    :return: A list [numpy array, text region list]
    """
    while True:
        text_region = []
        jpgname = np.random.choice(list_of_files)
        img = cv2.imread(jpgname)
        pattern = re.compile('jpg')
        txtname = pattern.sub('txt', jpgname)
        if not os.path.isfile(txtname):
            continue
        cropped_image = img
        with open(txtname, 'r') as f:
            for line in f:
                line_split = line.strip().split(',')
                # clockwise
                (x1, y1, x2, y2) = line_split[0:4]
                (x3, y3, x4, y4) = line_split[4:8]
                text_region.append([string.atof(x1), string.atof(y1), string.atof(x2), string.atof(y2),
                                    string.atof(x3), string.atof(y3), string.atof(x4), string.atof(y4)])
        if cropped_image is None or text_region is None or \
                cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
            continue
        yield [scale * cropped_image, text_region]


def image_output_pair(images):
    """

    :param images:
    :return:
    """
    for img, txtreg in images:
        # 1) generate imput data, input data is (320, 320, 3)

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
        yield (img, y_cls_mask_label, y_regr_cls_mask_label)


def group_by_batch(dataset, batch_size):
    """

    :param dataset:
    :param batch_size:
    :return:
    """
    while True:
        img, y_cls_mask_label, y_regr_cls_mask_label = zip(*[dataset.next() for i in xrange(batch_size)])
        batch = (np.stack(img), [np.stack(y_cls_mask_label), np.stack(y_regr_cls_mask_label)])
        yield batch


def load_dataset(directory, crop_size=320, batch_size=32):
    files = list_pictures(directory, 'jpg')
    generator = image_generator(files, crop_size, scale=1/255.0)
    generator = image_output_pair(generator)
    generator = group_by_batch(generator, batch_size)
    return generator


if __name__ == '__main__':
    gpu_id = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # define Input
    img_input = Input((320, 320, 3))
    # define network
    multi = multi_task(img_input)
    multask_model = Model(img_input, multi[0:2])
    # multask_model = Model(img_input, multi)
    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9)
    # parallel, use 4 GPU(TODO)
    # compile model
    # hinge loss
    # multask_model.compile(loss=[my_hinge, new_smooth], optimizer=sgd)
    # L2 loss
    multask_model.compile(loss=[l2, new_smooth], optimizer=sgd)
    # resume training
    multask_model = load_model('model/2017-07-06-18-04-loss-decrease-41-1.46.hdf5',
                               custom_objects={'my_hinge': l2, 'new_smooth': new_smooth})
    use_generator = False
    if use_generator:
        # use python generator to generate training data
        train_set = load_dataset('/home/yuquanjie/Documents/icdar2017_crop_center', 320, 64)
        val_set = load_dataset('/home/yuquanjie/Documents/icdar2017_crop_center_test', 320, 32)
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        filepath = "model/" + date_time + "-loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        # fit model
        model_info = multask_model.fit_generator(train_set, steps_per_epoch=100, epochs=10000, callbacks=callbacks_list,
                                                 validation_data=val_set, validation_steps=10, initial_epoch=0)
    else:
        print 'reading data from h5 file .....'
        filenamelist = ['dataset/train.h5']
        X, Y = read_multi_h5file(filenamelist)
        print 'traning data, input shape is {0}, output classifiction shape is {1}, regression shape is {2}'. \
            format(X.shape, Y[0].shape, Y[1].shape)
        # get date and time
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        filepath = "model/" + date_time + "-loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        # fit model
        model_info = loss_class = multask_model.fit(X, Y, batch_size=64, epochs=10000, shuffle=True,
                                                    callbacks=callbacks_list, verbose=1, validation_split=0.1)
