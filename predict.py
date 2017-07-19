"""
    @author  jasonYu
    @date    2017/6/3
    @version created
    @email   yuquanjie13@gmail.com
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import tools.point_check as point_check
import tools.nms as nms
import tools.draw_loss as draw_loss
from PIL import Image, ImageDraw
from keras.models import model_from_json
from keras.models import load_model
from quiver_engine.server import launch


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
                img_path_dict = {'imagePath': img_path}
                yield np.copy(im_arr), img_path_dict


if __name__ == '__main__':
    gpu_id = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    """
    # load create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    multitask_model = model_from_json(loaded_model_json)

    # load weights
    # multitask_model.load_weights('model/2017-07-13-18-34-loss-decrease-23-5.19-saved-weights.hdf5')

    model_disk = load_model('model/2017-07-09-14-53-loss-decrease-171-0.89.hdf5',
                            custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})
    weights_disk = model_disk.get_weights()
    multitask_model.set_weights(weights_disk)
    """

    # multitask_model = load_model('model/2017-07-09-14-53-loss-decrease-171-0.89.hdf5',
    #                            custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})

    multitask_model = load_model('model/2017-07-17-18-43-epoch-16-loss-6.78-saved-all-model.hdf5',
                                 custom_objects={'my_hinge': my_hinge, 'new_smooth': new_smooth})
    launch(multitask_model, input_folder='./img', port=5000)

    # all_imgs = glob.glob('/home/yuquanjie/Documents/shumei_crop_center_test/' + '*.jpg')
    all_imgs = glob.glob('/home/yuquanjie/Documents/icdar2017_crop_center/' + '*.jpg')
    # python generator
    data_gen_pred = get_pred_img(all_imgs)
    while True:
        X, img_data = data_gen_pred.next()
        # predict
        X = np.expand_dims(X, axis=0)
        predict_all = multitask_model.predict_on_batch(1/255.0 * X)
        # 1) classification result
        predict_cls = predict_all[0]
        # reduce dimension from (1, 80, 80, 1) to (80, 80)
        predict_cls = np.sum(predict_cls, axis=-1)
        predict_cls = np.sum(predict_cls, axis=0)
        # the pixel of text region on 80 * 80 feature map
        one_locs = np.where(predict_cls > 0.6)
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


