import os
import numpy as np
import cv2
import re
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
    img_draw = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_draw)
    for coord in coords:
        draw.polygon([(float(coord[0]), float(coord[1])), (float(coord[2]), float(coord[3])),
                      (float(coord[4]), float(coord[5])), (float(coord[6]), float(coord[7]))],
                     outline="red", fill="blue")
    img_draw = np.array(img_draw)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    # get image name using regular expression
    # pattern = re.compile(r'image_\d*_\d*\.jpg')
    pattern = re.compile(r'image_\d*_\d*')
    search = pattern.search(filename)
    image_name = search.group()
    image_name += '_'
    image_name += ".jpg"
    image_path = '/home/yuquanjie/Documents/deep-direct-regression/result/' + image_name
    cv2.imwrite(image_path, img_draw[0: img_draw.shape[0], 0: img_draw.shape[1]])


def get_raw_data(input_path):
    """
    process image and txt file, for each iamge file, reading corresponding txt file,
    getting all text region's coordinates and number of text region
    :param input_path: a directory has text and image 2 directory
    :return: 1) a list, each element is a dictionary
             'imagePath'
             'boxCoord'
             'boxNum'
             2) number of the txt and image file have processed
    """
    # define variable for returning
    all_imgs = []  # a list, each element is a dictionary
    coords = []  # a list, storing a image's all text region's coordinates which is wise clock
    num_file_txt = 0
    visulise = False
    save_gt = True
    print('Parsing annotation files')
    annot_path = os.path.join(input_path, 'text')
    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    box_num = 0
    for annot in annots:
        with open(annot, 'r') as f:
            num_file_txt += 1
            for line in f:
                box_num += 1
                line_split = line.strip().split(',')
                # each line of txt file is clockwise
                (x1, y1, x2, y2) = line_split[0:4]
                (x3, y3, x4, y4) = line_split[4:8]
                # anticlockwise
                # (x1, y1, x4, y4) = line_split[0:4]
                # (x3, y3, x2, y2) = line_split[4:8]
                coords.append((x1, y1, x2, y2, x3, y3, x4, y4))
            txtfilepath = annot
            # using regular expression, get image file path
            strinfo = re.compile('text')
            annot = strinfo.sub('image', annot)
            strinfo = re.compile('txt')
            imgfilepath = strinfo.sub('jpg', annot)
            annotation_data = {'imagePath': imgfilepath, 'boxCoord': coords, 'boxNum': box_num}
            box_num = 0
            # image file wheater corresponding to text file, and image file is not empty then add
            if os.path.isfile(imgfilepath) and os.path.isfile(txtfilepath) and os.path.getsize(imgfilepath):
                all_imgs.append(annotation_data)

            # print text region on image for comparing gt and predicted results
            if save_gt and os.path.isfile(imgfilepath) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(imgfilepath):
                print imgfilepath
                save_groudtruth(cv2.imread(imgfilepath), coords, imgfilepath)

            # visulising
            if visulise and os.path.isfile(imgfilepath) and os.path.isfile(txtfilepath) \
                    and os.path.getsize(imgfilepath):
                print imgfilepath
                for coord in coords:
                    img = cv2.imread(imgfilepath)
                    img_draw = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
            # it is very important to empty coords
            coords = []
    return all_imgs, num_file_txt

if __name__ == '__main__':
    get_raw_data('/home/yuquanjie/Documents/deep-direct-regression/captured_data')
