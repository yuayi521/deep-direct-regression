import os
import numpy as np
import cv2
import re
import glob
from PIL import Image, ImageDraw, ImageFont
fnt = ImageFont.truetype('/home/yuquanjie/Download/FreeMono.ttf', size=35)


def save_groudtruth(im, coords, filename):
    """
    print text region on image and save image
    :param im: numpy.ndarray
    :param coords: coordinates of text region
    :param filename: image file path
    :return: save image on a directory
    """
    print 'Saving ground truth ......{0}'.format(filename)
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    for coord in coords:
        draw.polygon([(float(coord[0]), float(coord[1])), (float(coord[2]), float(coord[3])),
                      (float(coord[4]), float(coord[5])), (float(coord[6]), float(coord[7]))],
                     outline="red", fill="blue")
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    # get image name using regular expression
    pattern = re.compile(r'image_\d*_\d*')
    # pattern = re.compile(r'\d*_\d*')
    search = pattern.search(filename)
    image_name = search.group()
    image_name += '_'
    image_name += ".jpg"
    image_path = '/home/yuquanjie/Documents/deep-direct-regression/result/' + image_name
    cv2.imwrite(image_path, img_draw[0: img_draw.shape[0], 0: img_draw.shape[1]])


def visualize(im, coords, filename):
    """

    :param im:
    :param coords:
    :param filename:
    :return:
    """
    print filename
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for coord in coords:
        draw = ImageDraw.Draw(img_draw)
        draw.polygon([(float(coord[0]), float(coord[1])), (float(coord[2]), float(coord[3])),
                      (float(coord[4]), float(coord[5])), (float(coord[6]), float(coord[7]))],
                     outline="red", fill="blue")
        draw.text([float(coord[0]), float(coord[1])], "1", font=fnt)
        draw.text([float(coord[2]), float(coord[3])], "2", font=fnt)
        draw.text([float(coord[4]), float(coord[5])], "3", font=fnt)
        draw.text([float(coord[6]), float(coord[7])], "4", font=fnt)
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', cv2.resize(img_draw, (800, 800)))
    cv2.waitKey(0)


def get_raw_data(input_path, save_gt=False):
    """
    process txt file
    getting all text region's coordinates and number of text region
    :param input_path: a directory has text and image 2 directory
    :param save_gt: save gt in result/ directory
    :return: 1) a list, each element is a dictionary
             'imagePath'
             'boxCoord'
             'boxNum'
             2) number of the txt that have processed
    """
    # define variable for returning
    all_txts = []  # a list, each element is a dictionary
    coords = []  # a list, storing a image's all text region's coordinates which is clockwise
    num_txt = 0
    visual = False
    print('Parsing txt files')
    # txt_directory = os.path.join(input_path, 'text')
    # all_txt_files = [os.path.join(txt_directory, s) for s in os.listdir(txt_directory)]
    txtfiles = input_path + '/*.txt'
    all_txt_files = glob.glob(txtfiles)
    box_num = 0
    for txt in all_txt_files:
        with open(txt, 'r') as f:
            num_txt += 1
            for line in f:
                box_num += 1
                line_split = line.strip().split(',')
                # clockwise
                (x1, y1, x2, y2) = line_split[0:4]
                (x3, y3, x4, y4) = line_split[4:8]
                coords.append((x1, y1, x2, y2, x3, y3, x4, y4))
            txtfilepath = txt
            # using regular expression, get image file path
            # pattern = re.compile('text')
            # img_file_path = pattern.sub('image', txt)
            pattern = re.compile('txt')
            img_file_path = pattern.sub('jpg', txtfilepath)
            txt_data = {'imagePath': img_file_path, 'boxCoord': coords, 'boxNum': box_num}
            box_num = 0
            coords = []
            # image file wheater corresponding to text file, and image file is not empty then add
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path):
                all_txts.append(txt_data)
            # -----------------------visualizing-----------------------------------------
            # draw text region on image and save image
            # print text region on image for comparing gt and predicted results
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path) and save_gt:
                save_groudtruth(cv2.imread(img_file_path), txt_data['boxCoord'], img_file_path)

            # draw text region on image and show image
            if os.path.isfile(img_file_path) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(img_file_path) and visual:
                visualize(cv2.imread(img_file_path), txt_data['boxCoord'], img_file_path)
            # -----------------------visualizing-----------------------------------------
    return all_txts, num_txt

if __name__ == '__main__':
    get_raw_data('/home/yuquanjie/Documents/icdar2017_cropped320', True)
