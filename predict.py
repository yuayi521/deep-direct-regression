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
import glob
import tools.point_check as point_check
import tools.nms as nms
import tools.draw_loss as draw_loss


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


def multi_task(input_tensor=None):
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


def get_pred_img(all_img):
    """
    get image data for predicting
    :param all_img: a lsit containing all image data whcih need to predict
    :return: image's numpy array
             image's path
    """
    while True:
        for img_path in all_img:
            print img_path
            if os.path.isfile(img_path):
                im_arr = cv2.imread(img_path)
                im_arr
                img_path_dict = {'imagePath': img_path}
                yield np.copy(im_arr), img_path_dict


if __name__ == '__main__':
    gpu_id = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # define Input
    img_input = Input((320, 320, 3))
    # define network
    multi = multi_task(img_input)
    # multask_model = Model(img_input, multi[0:2])
    multask_model = Model(img_input, multi[0:2])
    # define optimizer
    sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9)
    # compile model
    # multask_model.compile(loss=[my_hinge, smoothL1], optimizer=sgd)
    multask_model.compile(loss=[my_hinge, new_smooth], optimizer=sgd)
    # train data, test model using train data
    all_imgs = glob.glob('/home/yuquanjie/Documents/shumei_crop_center_test/' + '*.jpg')
    # python generator
    data_gen_pred = get_pred_img(all_imgs)
    while True:
        X, img_data = data_gen_pred.next()
        # load model
        final_model = load_model('model/2017-07-12-19-03-loss-decrease-65-1.05.hdf5',
                                 custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})
        # predict
        X = np.expand_dims(X, axis=0)
        predict_all = final_model.predict_on_batch(1/255.0 * X)
        # 1) classification result
        predict_cls = predict_all[0]
        # reduce dimension from (1, 80, 80, 1) to (80, 80)
        predict_cls = np.sum(predict_cls, axis=-1)
        predict_cls = np.sum(predict_cls, axis=0)
        # the pixel of text region on 80 * 80 feature map
        one_locs = np.where(predict_cls > 0.7)
        # the pixel of text region on 320 * 320 raw image
        coord = [one_locs[0] * 4, one_locs[1] * 4]

        # 2) regression result
        predict_regr = predict_all[1]
        # reduce dimension from (1, 80, 80, 8) to (80, 80, 8)
        predict_regr = np.sum(predict_regr, axis=0)
        # x1-4, y1-4 represent pixel(320 * 320 raw image) of text region's 8 corner coordinates
        x1, y1, x2, y2, x3, y3, x4, y4, score = [], [], [], [], [], [], [], [], []
        # predict_regr[][][], 1st and 2nd are 80 * 80 feature's coordinates
        # for each text region(cls > 0.7) on 80 * 80 feature map, calculate it's 8 corner coordinates
        for idx in xrange(len(one_locs[0])):
            x1.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][0])
            y1.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][1])
            x2.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][2])
            y2.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][3])
            x3.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][4])
            y3.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][5])
            x4.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][6])
            y4.append(predict_regr[one_locs[0][idx]][one_locs[1][idx]][7])
            use_aver_score = False
            if not use_aver_score:
                score.append(predict_cls[one_locs[0][idx]][one_locs[1][idx]])
            else:
                feat_map_poly = [(predict_regr[one_locs[0][idx]][one_locs[1][idx]][0] / 4,
                                  predict_regr[one_locs[0][idx]][one_locs[1][idx]][1] / 4),
                                 (predict_regr[one_locs[0][idx]][one_locs[1][idx]][2] / 4,
                                  predict_regr[one_locs[0][idx]][one_locs[1][idx]][3] / 4),
                                 (predict_regr[one_locs[0][idx]][one_locs[1][idx]][4] / 4,
                                  predict_regr[one_locs[0][idx]][one_locs[1][idx]][5] / 4),
                                 (predict_regr[one_locs[0][idx]][one_locs[1][idx]][6] / 4,
                                  predict_regr[one_locs[0][idx]][one_locs[1][idx]][7] / 4)]
                acc_score = 0.0
                num_score = 0
                for ix in xrange(80):
                    for jy in xrange(80):
                        if point_check.point_in_polygon(ix, jy, feat_map_poly):
                            num_score += 1
                            acc_score += predict_cls[ix][jy]
                score.append(acc_score / num_score)

        # nms
        # dets store all predicted text region pixel's 8 corner coord and score
        dets = []
        for idx in xrange(len(x1)):
            dets.append([x1[idx], y1[idx], x2[idx], y2[idx],
                         x3[idx], y3[idx], x4[idx], y4[idx], score[idx]])
        thresh = 0.3
        # using nms for all bbox, the remaining bbox's index
        idx_after_nms = []
        if len(x1) > 0:
            idx_after_nms = nms.poly_nms(np.array(dets), thresh)
        else:
            print 'no predicted text region pixel on {0}'.format(img_data['imagePath'])

        img = cv2.imread(img_data['imagePath'])
        img_draw = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_draw)
        use_nms = True
        if use_nms:
            for i in idx_after_nms:
                draw.polygon([(dets[i][0] + coord[0][i], dets[i][1] + coord[1][i]),
                              (dets[i][2] + coord[0][i], dets[i][3] + coord[1][i]),
                              (dets[i][4] + coord[0][i], dets[i][5] + coord[1][i]),
                              (dets[i][6] + coord[0][i], dets[i][7] + coord[1][i])], outline="blue")
        else:
            for i in xrange(len(one_locs[0])):
                # draw predicted text region on raw image(320 * 320)
                # use the coordinates on the raw image(320 * 320)
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
        image_name = img_data['imagePath'].split('/')[-1].split('.')[0]
        image_path = '/home/yuquanjie/Documents/deep-direct-regression/result/' + image_name + '' + '.jpg'
        # show on heatmap
        # cv2.imwrite(image_path, img_draw[0: img_draw.shape[0], 0: img_draw.shape[1]])

        # only draw classification result
        img = cv2.imread(img_data['imagePath'])
        img_cls = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_cls)
        for i in xrange(len(one_locs[0])):
            # draw predicted text region on raw image(320 * 320)
            # use the coordinates on the raw image(320 * 320)
            draw.text(([coord[0][i], coord[1][i]]), "O", "red")
        img_cls = np.array(img_cls)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_RGB2BGR)

        # show classification heat map
        fig_name = '/home/yuquanjie/Documents/deep-direct-regression/result/' + image_name + '_heatmap' + '.jpg'
        draw_loss.heatmap_cls(predict_cls, img_data, img_cls, img_draw, fig_name)


