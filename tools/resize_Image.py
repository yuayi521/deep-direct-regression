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


def get_h_w_range(img, center):
    """
    according text region center to capture a 1000 * 1000 image
    should consider boundary case
    :param img:
    :param center:
    :return:
    """
    # near to top left corner
    if center[0] - 500 < 0 and center[1] - 500 < 0:
        print 'near to top left corner'
        top_left = [0, 0]
        down_rig = [1000, 1000]
    # near to down left corner
    elif center[1] - 500 > 0 and center[0] - 500 < 0:
        print 'near to down left corner'
        top_left = [0, img.shape[0] - 1000]
        down_rig = [1000, img.shape[0]]
    # near to top right
    elif center[0] + 500 > img.shape[1] and center[1] + 500 < img.shape[0]:
        print 'near to top right'
        top_left = [img.shape[1] - 1000, 0]
        down_rig = [img.shape[1], 1000]
    # near to down right
    elif center[0] + 500 > img.shape[1] and center[1] + 500 > img.shape[0]:
        print 'near to down right'
        top_left = [img.shape[1] - 1000, img.shape[0] - 1000]
        down_rig = [img.shape[1], img.shape[0]]
    else:
        print 'yes'
        top_left = [center[0] - 500, center[1] - 500]
        down_rig = [center[0] + 500, center[1] + 500]

    print top_left
    print down_rig
    mydraw.draw_rectangle_image(img, top_left, down_rig)
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
                height_weight_range = get_h_w_range(im, text_center)

if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2'
                                                 '/train/part1')
    capture_image_from_textcenter(all_imgs)
    print type(all_imgs)
