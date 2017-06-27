"""
    @author  jasonYu
    @date    2017-06-27
    @version created
    @email   yuquanjie13@gmail.com
    @descrip resized_320 directory has 400k image and txt file, create a directory train_idx,
             every 4k image and txt file
"""
import glob
import os
import re

if __name__ == '__main__':
    jpgfiles = glob.glob(r'/home/yuquanjie/Documents/deep-direct-regression/resized_320/*.jpg')
    idx = 1
    dir_idx = 1
    workdir = '/home/yuquanjie/Documents/deep-direct-regression/resized_320/'
    to_move_list_jpg = []
    to_move_list_txt = []

    for jpg in jpgfiles:
        # get txt fiel
        pattern = re.compile('jpg')
        txt = pattern.sub('txt', jpg)
        to_move_list_jpg.append(jpg)
        to_move_list_txt.append(txt)
        if idx % 4000 == 0:
            print 'writing directory {0}'.format('train_' + bytes(dir_idx))
            imag_dir = workdir + 'train_' + bytes(dir_idx) + '/image'
            text_dir = workdir + 'train_' + bytes(dir_idx) + '/text'
            os.makedirs(imag_dir)
            os.makedirs(text_dir)
            for i in range(len(to_move_list_jpg)):
                shell_commd = 'cp ' + to_move_list_jpg[i] + ' ' + imag_dir
                print shell_commd
                os.system(shell_commd)
                shell_commd = 'cp ' + to_move_list_txt[i] + ' ' + text_dir
                os.system(shell_commd)
            # emtpy variable
            to_move_list_jpg = []
            to_move_list_txt = []

            dir_idx += 1
        idx += 1
    print 'jpg file number is {0}'.format(idx)
