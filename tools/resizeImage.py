"""
    @author  jasonYu
    @date    2017/6/14
    @version created
    @email   yuquanjie13@gmail.com
    @descrip capture image randomly from full size, eg. 2400 * 3200 to small size,
             eg.1200 * 1600
"""
import cv2
import cv2.cv as cv
import tools.get_data as get_data
import random as rd

def capture_image(imgs):
    """
    (TODO)
    :param imgs: a list, each elements is a dictionary
                 'imagePath'
                 'boxCoord'
                 'boxNum'
    :return:(TODO)
    """
    for img in imgs:
        im = cv.LoadImage(img['imagePath'])
        im_size = cv2.imread(img['imagePath'])
        if im_size.shape[0] > 1000 and im_size.shape[1] > 1000:
            print img['imagePath']
            # choose a top-left corner
            height_remain = im_size.shape[0] - 1000
            weight_remain = im_size.shape[1] - 1000

            for i in xrange(min(height_remain, weight_remain)):
                top_left = [rd.randint(0, height_remain), rd.randint(0, weight_remain)]
                cv.SetImageROI(im, (top_left[0], top_left[1],
                                    top_left[0] + 1000, top_left[0] + 1000))
                cv.ShowImage('img', im)
                cv.WaitKey(0)

if __name__ == '__main__':
    all_imgs, numFileTxt = get_data.get_raw_data('/home/yuquanjie/Documents/icdar2017rctw_train_v1.2'
                                                 '/train/part1')
    capture_image(all_imgs)
    print type(all_imgs)
