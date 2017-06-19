"""
    @author  jasonYu
    @date    2017/6/14
    @version created
    @email   yuquanjie13@gmail.com
    @descrip capture image randomly from full size, eg. 2400 * 3200 to small size,
             eg.1200 * 1600
"""
import cv2
import tools.get_data as get_data
import tools.mydraw as mydraw
import random as rd
import re
import string


def capture_image_random(imgs):
    """
    choose a point as top left coord randomly, then each aixs add  1000 pixel, generate
    down right coord, then get a 1000 * 1000 image
    :param imgs: a list, each elements is a dictionary
                 'imagePath'
                 'boxCoord'
                 'boxNum'
    :return:(TODO)
    """
    visiual = False
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        if im.shape[0] > 1000 and im.shape[1] > 1000:
            print img['imagePath']
            # choose a top-left corner
            height_remain = im.shape[0] - 1000  # 3200 - 1000
            weight_remain = im.shape[1] - 1000  # 2400 - 1000
            for i in xrange(50):
                weight_rand_idx = rd.randint(0, weight_remain)
                height_rand_idx = rd.randint(0, height_remain)
                # using regular expression to match image file name excluding .jpb
                pattern = re.compile(r'image_\d*')
                search = pattern.search(img['imagePath'])
                image_name = search.group()
                filename = '/home/yuquanjie/Documents/deep-direct-regression/captured_data/'\
                           + image_name + '_' + bytes(i) + '.jpg'
                # note : 1st dimension is height
                cv2.imwrite(filename, im[height_rand_idx: height_rand_idx + 1000,
                            weight_rand_idx: weight_rand_idx + 1000])
                if visiual:
                    roi_range = [[weight_rand_idx, weight_rand_idx + 1000],
                                 [height_rand_idx, height_rand_idx + 1000]]
                    mydraw.draw_rectangle_image(im, roi_range)


def get_captured_img_toplef_downrig_coord(im, c):
    """
    according text region center coordinate to capture a 1000 * 1000 image
    should consider some boundary cases
    :param im:
    :param c: cneter coordinate of text region
    :return:
    """
    if c[0] - 500 < 0 and c[1] - 500 < 0:
        print '1'
        top_left = [0, 0]
        down_rig = [1000, 1000]
    elif c[1] - 500 < 0 and (c[0] - 500 > 0 and c[0] + 500 < im.shape[1]):
        print '2'
        top_left = [c[0] - 500, 0]
        down_rig = [c[0] + 500, 1000]
    elif c[0] - 500 < 0 and (c[1] - 500 > 0 and c[1]  + 500 < im.shape[0]):
        print '3'
        top_left = [0, c[1] - 500]
        down_rig = [1000, c[1] + 500]
    elif c[0] + 500 > im.shape[1] and c[1] + 500 > im.shape[0]:
        print '4'
        top_left = [im.shape[1] - 1000, 0]
        down_rig = [im.shape[1], 1000]
    elif c[0] + 500 > im.shape[1] and (c[1] - 500 > 0 and c[1] + 500 < im.shape[0]):
        print '5'
        top_left = [im.shape[1] - 1000, c[1] - 500]
        down_rig = [im.shape[1], c[1] + 500]
    elif c[0] - 500 < 0 and c[1] + 500 > im.shape[0]:
        print '6'
        top_left = [0, im.shape[0] - 1000]
        down_rig = [1000, im.shape[0]]
    elif c[1] + 500 > im.shape[0] and (c[0] - 500 > 0 and c[0] + 500 < im.shape[1]):
        print '7'
        top_left = [c[0] - 500, im.shap[0] - 1000]
        down_rig = [c[0] + 500, im.shape[0]]
    elif c[0] + 500 > im.shape[1] and c[1] + 500 > im.shape[0]:
        print '8'
        top_left = [im.shape[1] - 1000, im.shape[0] - 1000]
        down_rig = [im.shape[1], im.shape[0]]
    else:
        print '9'
        top_left = [c[0] - 500, c[1] - 500]
        down_rig = [c[0] + 500, c[1] + 500]

    print top_left
    print down_rig
    mydraw.draw_rectangle_image(im, top_left, down_rig)
    return top_left, down_rig


def capture_image_from_textcenter(imgs):
    """
    according to text region center, generate  1000 * 1000 images
    :param imgs: a list, each elements is a dictionary
                 'imagePath' :
                 'boxCoord' :
                 'boxNum' :
    :return:
    """
    visiual = True
    for img in imgs:
        im = cv2.imread(img['imagePath'])
        if im.shape[0] > 1000 and im.shape[1] > 1000:
            print img['imagePath']
            for coord in img['boxCoord']:
                # calculate center of text region
                text_center = [(string.atof(coord[0]) + string.atof(coord[4])) / 2,
                               (string.atof(coord[1]) + string.atof(coord[5])) / 2]
                if visiual:
                    mydraw.draw_text_on_image(im, text_center)
                print coord
                print text_center
                height_weight_range = get_captured_img_toplef_downrig_coord(im, text_center)

if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2'
                                                 '/train/part1')
    capture_image_from_textcenter(all_imgs)
    print type(all_imgs)
