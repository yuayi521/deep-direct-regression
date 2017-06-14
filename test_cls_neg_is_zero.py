'''
    @author  jasonYu
    @date    2017/6/3
    @version created
    @email   yuquanjie13@gmail.com
'''
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Input
from keras.models import Model
import sys
import os
import h5py
import tensorflow as tf

sys.path.append('/home/yuquanjie/Documents/deep-direct-regression/tools')
HUBER_DELTA = 1.0
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)

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

def weighted_hinge(y_true, y_pred):
    """ Compute weighted hinge loss for classification

    :param y_true: Ground truth for category,
                    negative is -1, positive is 1
                    tensor shape (?, 80, 80, 1)
    :param y_pred: Predicted category
                    tensor shape (?, 80, 80, 1)
    :return: hinge loss, tensor shape (1, )
    """

    num_pos = tf.reduce_mean(y_true)
    num_neg = 6400.0 - num_pos

    #loss_pos =
    #loss_neg
    #return my_hinge_loss

def smoothL1(y_true, y_pred):
    #print y_true
    import tensorflow as tf
    #1. slice
    #conTmp = tf.slice(y_true, [0, 0, 0, 8],[1, 80, 80, 1])
    #2. concatenate
    #tmp = tf.expand_dims(y_true[:, :, :, 8], 3)  page 27 helped by hl
    tmp = tf.expand_dims(y_true[:, :, :, 8], 3)
    #print tmp
    y_true = tf.concat([y_true, tmp], 3)
    y_true = tf.concat([y_true, tmp], 3)
    y_true = tf.concat([y_true, tmp], 3)

    y_true = tf.concat([y_true, tmp], 3)
    y_true = tf.concat([y_true, tmp], 3)
    y_true = tf.concat([y_true, tmp], 3)
    y_true = tf.concat([y_true, tmp], 3)
    #print y_true
    x = K.abs(y_true[:, :, :, 0:8] - y_pred)
    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        x = tf.where(tf.greater(HUBER_DELTA, x),
                     0.5 * x ** 2,
                     HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        x = tf.where(tf.greater(y_true[:, :, :,8:16], 0),
                     y_true[:, :, :,8:16],
                     0 * y_true[:, :, :,8:16]) * x
        #return  K.sum(x)
        return  K.mean(x, axis = -1)

def multi_task(input_tensor=None, trainable=False):
    img_input = BatchNormalization()(input_tensor)

    #conv_1
    conv1_1 = Convolution2D(32, (5, 5), strides=(1,1), padding='same',
                            activation='relu', name='conv1_1')(img_input)
    pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_1)

    #conv_2
    conv2_1 = Convolution2D(64, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv2_1')(pool1)
    conv2_2 = Convolution2D(64, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)

    #conv_3
    conv3_1 = Convolution2D(128, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv3_1')(pool2)
    conv3_2 = Convolution2D(128, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv3_2')(conv3_1)
    pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_2)
    pool3_for_fuse = Convolution2D(128, (1, 1), strides=(1,1), padding='same',
                                   activation='relu', name='pool3_for_fuse')(pool3)

    #conv_4
    conv4_1 = Convolution2D(256, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv4_1')(pool3)
    conv4_2 = Convolution2D(256, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv4_2')(conv4_1)
    pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_2)
    pool4_for_fuse = Convolution2D(128, (1, 1), strides=(1,1), padding='same',
                                   activation='relu', name='pool4_for_fuse')(pool4)

    #conv_5
    conv5_1 = Convolution2D(512, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv5_1')(pool4)
    conv5_2 = Convolution2D(512, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv5_2')(conv5_1)
    pool5 = MaxPooling2D((2,2), strides=(2,2), name='pool5')(conv5_2)
    pool5_for_fuse = Convolution2D(128, (1, 1), strides=(1,1), padding='same',
                                   activation='relu', name='pool5_for_fuse')(pool5)

    #conv_6
    conv6_1 = Convolution2D(512, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv6_1')(pool5)
    conv6_2 = Convolution2D(512, (3, 3), strides=(1,1), padding='same',
                            activation='relu', name='conv6_2')(conv6_1)
    pool6 = MaxPooling2D((2,2), strides=(2,2), name='pool6')(conv6_2)

    #
    conv7_1 = Convolution2D(128, (1, 1), strides=(1,1), padding='same',
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
    x_clas = Convolution2D(1, (1, 1), strides=(1,1), padding='same' ,name='out_class')(upscore16)
    x = Convolution2D(128, (1, 1), strides=(1,1), padding='same', activation='relu')(upscore16)
    x = Convolution2D(8, (1, 1), strides=(1,1), padding='same', activation='sigmoid')(x)
    x_regr = Lambda(lambda t: 800 * t - 400)(x)
    return [x_clas, x_regr, x]

if __name__ == '__main__':
    # define Input
    img_input = Input((320,320,3))
    # define network
    multi = multi_task(img_input,trainable=True)
    #multask_model = Model(img_input, multi[0:2])
    multask_model = Model(img_input, multi[0])
    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9)
    # compile model
    #multask_model.compile(loss=[my_hinge, smoothL1], optimizer=sgd)
    multask_model.compile(loss=[my_hinge], optimizer=sgd)
    # read training data from h5 file
    file = h5py.File('dataset/train_dataset-1500-negIsZero.h5','r')
    X = file['X_train'][:]
    Y_1 = file['Y_train_cls'][:]
    Y_2 = file['Y_train_merge'][:]
    file.close()
    Y = [Y_1, Y_2]
    print 'traning data shape ------'
    print X.shape
    print Y_1.shape
    print Y_2.shape
    # read validation data from h5 file
    file = h5py.File('dataset/val_dataset-1000.h5', 'r')
    print 'validation data shape -----'
    X_val = file['X_train'][:]
    Y_val_1 = file['Y_train_cls'][:]
    Y_val_2 = file['Y_train_merge'][:]
    Y_val = [Y_val_1, Y_val_2]
    print X_val.shape
    print Y_val_1.shape
    print Y_val_2.shape
    file.close()
    # saved model file path and name
    filepath = "model/loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True,
                                 mode='min')
    callbacks_list = [checkpoint]
    # fit model
    #loss_class = multask_model.fit(X, Y_1, batch_size=32, epochs=5000, callbacks = callbacks_list, validation_data=(X_val, Y_val_1), verbose=1)
    loss_class = multask_model.fit(X, Y_1, batch_size=32, epochs=5000, callbacks = callbacks_list, verbose=1)

