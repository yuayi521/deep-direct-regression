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
from shapely.geometry import Polygon
from keras.preprocessing.image import list_pictures
import tools.point_check as point_check
import cv2
import string
import numpy as np
import random as rd
import os
import re
import h5py
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

HUBER_DELTA = 1.0
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


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


"""
def weighted_hinge(y_true, y_pred):
    Compute weighted hinge loss for classification

    :param y_true: Ground truth for category,
                    negative is -1, positive is 1
                    tensor shape (?, 80, 80, 1)
    :param y_pred: Predicted category
                    tensor shape (?, 80, 80, 1)
    :return: hinge loss, tensor shape (1, )

    reduce_dim_y_true = tf.reduce_sum(y_true, axis=-1)
    reduce_dim_y_pred = tf.reduce_sum(y_pred, axis=-1)

    zero = tf.constant(0, dtype=tf.int32)
    where_not_zero = tf.not_equal(reduce_dim_y_true, zero)
    where_zero = tf.equal(reduce_dim_y_true, zero)

    # indices_pos, tensor shape(batch_size, 80, 80), indicating positive indices
    indices_pos = tf.where(where_not_zero)
    # indices_pos, tensor shape(batch_size, 80, 80), indicating negative indices
    indices_neg = tf.where(where_zero)

    num_pos = tf.to_float(tf.shape(indices_pos)[0])
    num_neg = tf.to_float(tf.shape(indices_neg)[0])

    factor = num_pos / num_neg
    pos_ratio = 1 - factor
    neg_ratio = 1 - pos_ratio

    loss_pos =
    loss_neg =
    weighted_hinge_loss = pos_ratio * loss_pos + neg_ratio * loss_neg
    return weighted_hinge_loss
"""


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


def smooth_l1(y_true, y_pred):
    """
    Compute regresstion loss, loss didn't not divide batch_size
    :param y_true:
    :param y_pred:
    :return:
    """
    sub = tf.expand_dims(y_true[:, :, :, 8], axis=3)
    for i in xrange(7):
        y_true = tf.concat([y_true, sub], axis=3)
    abs_val = tf.abs(y_true[:, :, :, 0:8] - y_pred)
    smooth = tf.where(tf.greater(1.0, abs_val),
                      0.5 * abs_val ** 2,
                      abs_val - 0.5)
    if False:
        loss = tf.where(tf.greater(y_true[:, :, :, 8:16], 0),
                        y_true[:, :, :, 8:16],
                        0 * y_true[:, :, :, 8:16]) * smooth
    else:
        loss = tf.where(tf.greater(y_true[:, :, :, 8:16], 0),
                        y_true[:, :, :, 8:16],
                        0 * y_true[:, :, :, 8:16]) * smooth
    loss = tf.reduce_mean(loss, axis=-1)
    return loss


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


def plot_model_history(model_history):
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    # summarize history for accuracy --- classification
    axs[0][0].plot(range(1, len(model_history.history['cls_acc']) + 1), model_history.history['cls_acc'])
    axs[0][0].plot(range(1, len(model_history.history['val_cls_acc']) + 1), model_history.history['val_cls_acc'])
    axs[0][0].set_title('Model Accuracy Classification')
    axs[0][0].set_ylabel('Accuracy')
    axs[0][0].set_xlabel('Epoch')
    axs[0][0].set_xticks(np.arange(1, len(model_history.history['cls_acc']) + 1), len(model_history.history['cls_acc']) / 10)
    axs[0][0].legend(['train', 'val'], loc='best')

    # summarize history for accuracy --- Regression
    axs[0][1].plot(range(1, len(model_history.history['lambda_1_acc']) + 1), model_history.history['lambda_1_acc'])
    axs[0][1].plot(range(1, len(model_history.history['val_lambda_1_acc']) + 1), model_history.history['val_lambda_1_acc'])
    axs[0][1].set_title('Model Accuracy Regression')
    axs[0][1].set_ylabel('Accuracy')
    axs[0][1].set_xlabel('Epoch')
    axs[0][1].set_xticks(np.arange(1, len(model_history.history['lambda_1_acc']) + 1), len(model_history.history['lambda_1_acc']) / 10)
    axs[0][1].legend(['train', 'val'], loc='best')

    # summarize history for loss --- classification
    axs[1][0].plot(range(1, len(model_history.history['cls_loss']) + 1), model_history.history['cls_loss'])
    axs[1][0].plot(range(1, len(model_history.history['val_cls_loss']) + 1), model_history.history['val_cls_loss'])
    axs[1][0].set_title('Model Loss Classification')
    axs[1][0].set_ylabel('Loss')
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_xticks(np.arange(1, len(model_history.history['cls_loss']) + 1), len(model_history.history['cls_loss']) / 10)
    axs[1][0].legend(['train', 'val'], loc='best')

    # summarize history for loss --- Regression
    axs[1][1].plot(range(1, len(model_history.history['lambda_1_loss']) + 1), model_history.history['lambda_1_loss'])
    axs[1][1].plot(range(1, len(model_history.history['val_lambda_1_loss']) + 1), model_history.history['val_lambda_1_loss'])
    axs[1][1].set_title('Model Loss Regression')
    axs[1][1].set_ylabel('Loss')
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].set_xticks(np.arange(1, len(model_history.history['lambda_1_loss']) + 1), len(model_history.history['lambda_1_loss']) / 10)
    axs[1][1].legend(['train', 'val'], loc='best')

    plt.show()


def read_txts(txtpath):
    """

    :param txtpath:
    :return: a list containing all text region clockwise coordinates which sotred in a tuple
    """
    coords = []
    with open(txtpath, 'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            # clockwise
            (x1, y1, x2, y2) = line_split[0:4]
            (x3, y3, x4, y4) = line_split[4:8]
            coords.append((x1, y1, x2, y2, x3, y3, x4, y4))
    return coords


def random_crop(image, txts, crop_size=320):
    """

    :param image:
    :param txts:
    :param crop_size:
    :return:
    """
    ret_cropped_img = True
    height = image.shape[1]
    width = image.shape[0]
    strcoord_list = []
    x, y = 0, 0
    if width < crop_size or height < crop_size:
        return None, None
    # once find a cropped image which has intersections with raw image's text region, then jump out, and return cropped
    # image and its text region coordinates
    for idx in xrange(2000):
        strcoord_list = []
        x = rd.randint(0, width - crop_size)
        y = rd.randint(0, height - crop_size)
        cropped_img_poly = Polygon([(x, y), (x, y + crop_size),
                                    (x + crop_size, y + crop_size), (x + crop_size, y)])
        for txt in txts:
            x1, x2 = string.atof(txt[0]), string.atof(txt[2])
            x3, x4 = string.atof(txt[4]), string.atof(txt[6])
            y1, y2 = string.atof(txt[1]), string.atof(txt[3])
            y3, y4 = string.atof(txt[5]), string.atof(txt[7])
            rawimg_txt_poly = Polygon([(x1, y1), (x4, y4), (x3, y3), (x2, y2)])
            if rawimg_txt_poly.intersects(cropped_img_poly):
                inter = rawimg_txt_poly.intersection(cropped_img_poly)
                # insure the intersected quardrangle's aera is greater than 0
                if inter.area == 0:
                    ret_cropped_img = False
                    break
                # insure the text region's percentage should  greater than 10%
                if inter.area < (crop_size / 10) * (crop_size / 10):
                    ret_cropped_img = False
                    break
                # insure the text region's percentage should not greater than 88%
                if inter.area > (crop_size - 20) * (crop_size - 20):
                    ret_cropped_img = False
                    break
                list_inter = list(inter.exterior.coords)
                x1, y1 = list_inter[0][0] - x, list_inter[0][1] - y
                x2, y2 = list_inter[3][0] - x, list_inter[3][1] - y
                x3, y3 = list_inter[2][0] - x, list_inter[2][1] - y
                x4, y4 = list_inter[1][0] - x, list_inter[1][1] - y
                # insure the top_left coordinates is on the top-left position
                if x1 < x2 and y1 < y4 and x3 > x4 and y3 > y2:
                    ret_cropped_img = True
                else:
                    ret_cropped_img = False
                    break
                    # insure the intersected poly is quardrangle
                if len(list_inter) != 5:
                    ret_cropped_img = False
                    break
                strcoord_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
            else:
                ret_cropped_img = False
                break
        if ret_cropped_img:
            break
    # ret_cropped_img is True, represent cropped iamge correctly, and the cropped image has
    # text region, the percentage range form 10 % to 88%
    if ret_cropped_img:
        return image[y:(y + crop_size), x:(x + crop_size), :], strcoord_list
    else:
        return None, None


def image_generator(list_of_files, crop_size=320, scale=1):
    """
    this is a python generator, return array format image
    :param list_of_files: list, storing all jpg file path
    :param crop_size: cropped image size
    :param scale: normalization parameters
    :return: return array format image
    """
    while True:
        filename = np.random.choice(list_of_files)
        img = cv2.imread(filename)
        pattern = re.compile('jpg')
        txtpath = pattern.sub('txt', filename)
        if os.path.isfile(txtpath):
            txts = read_txts(txtpath)
        else:
            continue
        cropped_image, text_region = random_crop(img, txts, crop_size)
        if cropped_image is None or text_region is None or \
                cropped_image.shape[0] != crop_size or cropped_image.shape[1] != crop_size:
            continue
        # save middle result
        if False:
            pattern = re.compile(r'image_\d*')
            search = pattern.search(filename)
            image_name = search.group()
            jpgname = '/home/yuquanjie/Documents/visual/' + image_name + '_' + bytes(10) + '.jpg'
            txtname = '/home/yuquanjie/Documents/visual/' + image_name + '_' + bytes(10) + '.txt'
            # print 'writing ... {0}'.format(jpgname)
            cv2.imwrite(jpgname, cropped_image)
            txtwrite = open(txtname, 'a')
            for txt in text_region:
                for it in txt:
                    txtwrite.write(bytes(it) + ',')
                txtwrite.write('\n')
            txtwrite.close()
        # save middle result
        yield [scale * cropped_image, text_region]


def image_output_pair(images):
    """

    :param images:
    :return:
    """
    for img, txtreg in images:
        # 1) generate imput data, input data is (320, 320, 3)
        # 2) generate clsssification data
        # x-axis and y-axis reduced scale
        reduced_x = float(img.shape[1]) / 80.0
        reduced_y = float(img.shape[0]) / 80.0
        y_class_label = -1 * np.ones((80, 80))  # negative lable is -1
        for ix in xrange(y_class_label.shape[0]):
            for jy in xrange(y_class_label.shape[1]):
                for polygon in txtreg:
                    x1, x2 = polygon[0] / reduced_x, polygon[2] / reduced_x
                    x3, x4 = polygon[4] / reduced_x, polygon[6] / reduced_x
                    y1, y2 = polygon[1] / reduced_y, polygon[3] / reduced_y
                    y3, y4 = polygon[5] / reduced_y, polygon[7] / reduced_y
                    polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    if point_check.point_in_polygon(ix, jy, polygon):
                        y_class_label[ix][jy] = 1
        # output classificaiton label is (80, 80, 1)
        # calculate ones's locations before expand the dimension of y_class_label
        one_locs = np.where(y_class_label > 0)
        y_class_label = np.expand_dims(y_class_label, axis=-1)
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
                # 80 * 80  size image's quardrangle
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
        y_merge_label = np.concatenate((y_regr_lable, y_class_label), axis=-1)
        yield (img, y_class_label, y_merge_label)


def group_by_batch(dataset, batch_size):
    """

    :param dataset:
    :param batch_size:
    :return:
    """
    while True:
        img, y_class_label, y_merge_label = zip(*[dataset.next() for i in xrange(batch_size)])
        batch = (np.stack(img), [np.stack(y_class_label), np.stack(y_merge_label)])
        yield batch


def load_dataset(directory, crop_size=320, batch_size=32):
    files = list_pictures(directory, 'jpg')
    generator = image_generator(files, crop_size, scale=1/255.0)
    generator = image_output_pair(generator)
    generator = group_by_batch(generator, batch_size)
    return generator


if __name__ == '__main__':
    # define Input
    img_input = Input((320, 320, 3))
    # define network
    multi = multi_task(img_input)
    multask_model = Model(img_input, multi[0:2])
    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9)
    # compile model
    # multask_model.compile(loss=[my_hinge, smooth_l1], optimizer=sgd)
    multask_model.compile(loss=[my_hinge, new_smooth], optimizer=sgd, metrics=['acc'])
    # resume training
    # multask_model = load_model('model/2017-06-23-17-14-loss-decrease-1827-0.65.hdf5',
    #                            custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})

    # use python generator to generate training data
    train_set = load_dataset('/home/yuquanjie/Documents/icdar2017_dataset/train', 320, 64)
    # val_set = load_dataset('/home/yuquanjie/Documents/icdar2017_dataset/val', 320, 8)
    # get date and time
    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    filepath = "model/" + date_time + "-loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit model
    model_info = multask_model.fit_generator(train_set, steps_per_epoch=1000, epochs=1000,
                                             callbacks=callbacks_list)
    # plot_model_history(model_info)
