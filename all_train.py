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
import sys
import numpy as np
import os
import h5py
import tensorflow as tf
import datetime

sys.path.append('/home/yuquanjie/Documents/deep-direct-regression/tools')
HUBER_DELTA = 1.0
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def read_multi_h5file(filenamelist):
    """
    read multi h5 file
    :param filenamelist:
    :return: network input X and output Y
    """
    read = h5py.File(filenamelist[0], 'r')
    x_train = read['X_train'][:]
    y_1_cls = read['Y_train_cls'][:]
    y_2_mer = read['Y_train_merge'][:]
    read.close()

    for idx in range(1, len(filenamelist)):
        read = h5py.File(filenamelist[idx], 'r')
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
    x_clas = Convolution2D(1, (1, 1), strides=(1, 1), padding='same', name='out_class')(upscore16)
    x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu')(upscore16)
    x = Convolution2D(8, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
    x_regr = Lambda(lambda t: 800 * t - 400)(x)
    return [x_clas, x_regr, x]


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
    multask_model.compile(loss=[my_hinge, new_smooth], optimizer=sgd)

    # resume training
    # model.save_weights() use load_weights()
    # multask_model.load_weights('model/2017-06-23-17-14-loss-decrease-1827-0.65.hdf5')
    multask_model = load_model('model/2017-06-23-17-14-loss-decrease-1827-0.65.hdf5',
                               custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})

    # read training data from h5 file
    print 'reading data from h5 file .....'
    filenamelist = ['dataset/train_size320_1_.h5', 'dataset/train_size320_2_.h5', 'dataset/train_size320_3_.h5']
    # filenamelist = ['dataset/train_size320_num1500.h5']
    X, Y = read_multi_h5file(filenamelist)
    print 'traning data, input shape is {0}, output classifiction shape is {1}, regression shape is {2}'.\
        format(X.shape, Y[0].shape, Y[1].shape)
    """"
    # read validation data from h5 file
    file_read = h5py.File('dataset/train_size320_num1500.h5', 'r')
    X_val = file_read['X_train'][:]
    Y_val_1 = file_read['Y_train_cls'][:]
    Y_val_2 = file_read['Y_train_merge'][:]
    Y_val = [Y_val_1, Y_val_2]
    file_read.close()
    print 'validation data, input shape is {0}, output shape classification shape is {1}, regression ' \
          'shape is {2}'.format(X_val.shape, Y_val[0].shape, Y[1].shape)
    """
    # saved model file path and name
    # get date and time
    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    filepath = "model/" + date_time + "-loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
    # fit model
    use_val_data = False
    if use_val_data:
        # add validation data
        # loss_class = multask_model.fit(X, Y, batch_size=8, epochs=5000, shuffle=True, callbacks=callbacks_list,
        #                              validation_data=(X_val, Y_val), verbose=1)
        print '1'
    else:
        # not use validation data for faster speed
        loss_class = multask_model.fit(X, Y, batch_size=64, epochs=5000, shuffle=True,
                                       callbacks=callbacks_list, verbose=1)
