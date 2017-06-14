
# coding: utf-8

# In[16]:


'''
    @author  jasonYu
    @date    2017/6/3
    @version created
    @email   yuquanjie13@gmail.com
'''
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import sys
import string
import re
sys.path.append('/home/yuquanjie/Documents/deep-direct-regression/tools')
from point_check import point_in_polygon

'''
1. generate intput data
    1.1 input image resizing, from (2400,3200) to (320,320)
    1.2 (TODO) zero-center by mean pixel??
    1.3 proprecess image
2. generate output data
    2.1 generate classification output data
    2.2 (TODO) generate regression output data
'''
def get_train_data(all_imgs):
    visulise = False
    while True:
        for img_data in all_imgs:      
            print img_data['imagePath']
            # image file wheater corresponding to text fle
            annot = img_data['imagePath']
            strinfo = re.compile('image/')
            annot = strinfo.sub('text/',annot)
            strinfo = re.compile('jpg')
            annot = strinfo.sub('txt',annot)
            
            if os.path.isfile(img_data['imagePath']) and os.path.isfile(annot):
                img = cv2.imread(img_data['imagePath'])
                width = img.shape[0] #2400
                height = img.shape[1] #3200

                ## 1)generate input data
                ### 1.1)input image, from (2400,3200) to (320,320)
                img_320 = cv2.resize(img,(320,320),interpolation=cv2.INTER_CUBIC)

                ## 2)generate output data
                ### 2.1)generate classification output data
                divi_x = float(height) / 80.0
                divi_y = float(width) / 80.0
                y_class_lable = -1 * np.ones((80,80)) 
                for ix in xrange(y_class_lable.shape[0]):
                    for jy in xrange(y_class_lable.shape[1]):
                        for polygon in img_data['boxCoord']:
                            x1 = string.atof(polygon[0]) / divi_x
                            x2 = string.atof(polygon[2]) / divi_x
                            x3 = string.atof(polygon[4]) / divi_x
                            x4 = string.atof(polygon[6]) / divi_x

                            y1 = string.atof(polygon[1]) / divi_y
                            y2 = string.atof(polygon[3]) / divi_y
                            y3 = string.atof(polygon[5]) / divi_y
                            y4 = string.atof(polygon[7]) / divi_y

                            polygon = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                            if point_in_polygon(ix,jy,polygon):
                                y_class_lable[ix][jy] = 1
                            #else:
                            #    y_class_lable[ix][jy] = 0   

                if visulise:
                    if img_data['imagePath'] == '/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1/image/image_100.jpg' :
                        img = cv2.imread(img_data['imagePath'])
                        img_80 = cv2.resize(img,(80,80),interpolation=cv2.INTER_CUBIC)
                        img_draw = Image.fromarray(cv2.cvtColor(img_80,cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_draw)
                        for coord in img_data['boxCoord']:
                            print 'detail'
                            print float(coord[0]) / divi_x, float(coord[1]) / divi_y
                            print float(coord[2]) / divi_x, float(coord[3]) / divi_y
                            print float(coord[4]) / divi_x, float(coord[5]) / divi_y
                            print float(coord[6]) / divi_x, float(coord[7]) / divi_y
                            print 'detail'
                            draw.polygon([(float(coord[0]) / divi_x, float(coord[1]) / divi_y), 
                                          (float(coord[2]) / divi_x, float(coord[3]) / divi_y),
                                          (float(coord[4]) / divi_x, float(coord[5]) / divi_y), 
                                          (float(coord[6]) / divi_x, float(coord[7]) / divi_y)],
                                         outline = "red",fill="blue")
                        img_draw = np.array(img_draw)
                        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                        one_locs = np.where(y_class_lable > 0)
                        print one_locs
                        print len(one_locs[0])
                        print img_data['imagePath']
                        print img_data['boxNum']
                        cv2.imshow('img',img_draw)
                        cv2.waitKey(0)        
                
                img_320 = np.expand_dims(img_320,axis = 0)
                one_locs = np.where(y_class_lable > 0)
                y_class_lable = np.expand_dims(y_class_lable,axis = 0)
                y_class_lable = np.expand_dims(y_class_lable,axis = 3)
                #yield np.copy(img_320), np.copy(y_class_lable), img_data
                
                ### 2.2)(TODO) generate regression output data
                y_regr_lable = np.zeros((80,80,8))                   
                for i in xrange(len(one_locs[0])):
                    # get quadrilateral vertex 4 corrdinates
                    for polygon in img_data['boxCoord']:
                        x1 = string.atof(polygon[0]) / divi_x
                        x2 = string.atof(polygon[2]) / divi_x
                        x3 = string.atof(polygon[4]) / divi_x
                        x4 = string.atof(polygon[6]) / divi_x

                        y1 = string.atof(polygon[1]) / divi_y
                        y2 = string.atof(polygon[3]) / divi_y
                        y3 = string.atof(polygon[5]) / divi_y
                        y4 = string.atof(polygon[7]) / divi_y

                        poly = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                        ix = one_locs[0][i]
                        jy = one_locs[1][i]
                        if point_in_polygon(ix,jy,poly):
                            left_top_x = poly[0][0]
                            left_top_y = poly[0][1]
                            righ_top_x = poly[1][0]
                            righ_top_y = poly[1][1]

                            righ_dow_x = poly[2][0]
                            righ_dow_y = poly[2][1] 
                            left_dow_x = poly[3][0]
                            left_dow_y = poly[3][1]

                            y_regr_lable[ix][jy][0] = left_top_x * 4 - ix * 4
                            y_regr_lable[ix][jy][1] = left_top_y * 4 - jy * 4
                            y_regr_lable[ix][jy][2] = righ_top_x * 4 - ix * 4
                            y_regr_lable[ix][jy][3] = righ_top_y * 4 - jy * 4

                            y_regr_lable[ix][jy][4] = righ_dow_x * 4 - ix * 4
                            y_regr_lable[ix][jy][5] = righ_dow_y * 4 - jy * 4
                            y_regr_lable[ix][jy][6] = left_dow_x * 4 - ix * 4
                            y_regr_lable[ix][jy][7] = left_dow_y * 4 - jy * 4
                if visulise and img_data['imagePath'] == '/home/yuquanjie/Documents/icdar2017rctw_train_v1.2/train/part1/image/image_100.jpg' :
                    print y_regr_lable[59][75]
                    print y_regr_lable[59][76]
                    print y_regr_lable[59][77]
                y_regr_lable = np.expand_dims(y_regr_lable,axis = 0)
                yield np.copy(img_320), np.copy(y_class_lable), np.copy(y_regr_lable), img_data
            else:
                continue


# In[17]:
from keras import backend as K
import tensorflow as tf
HUBER_DELTA = 1.0
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
# In[18]:


from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras import backend as K

def regression_net(input_tensor=None, trainable=False):
    img_input = input_tensor
    #??
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
                     
    
    x = Convolution2D(128, (1, 1), strides=(1,1), padding='same', activation='relu')(upscore16)
    x = Convolution2D(8, (1, 1), strides=(1,1), padding='same',activation='sigmoid')(x)
    x_regr = Lambda(lambda t: 800 * t - 400)(x)
    return x_regr

# In[ ]:

from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import time
from PIL import Image, ImageFont, ImageDraw
from keras.layers import Input
from keras.models import Model
import sys
import h5py
sys.path.append('/home/yuquanjie/Documents/deep-direct-regression/tools')
from point_check import point_in_polygon
from get_data import get_raw_data
import os
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)

# Define Input
img_input = Input((320,320,3))
# Define network
regr = regression_net(img_input,trainable=True)
regr_model = Model(img_input,regr)
# Compile model
regr_model.compile(loss=smoothL1, optimizer='sgd')
# Read train data from file
file = h5py.File('../dataset/train_dataset-1500.h5','r')
X_train = file['X_train'][:]
Y_train = file['Y_train_merge'][:]
print 'train data shape'
print X_train.shape
print Y_train.shape
file.close()
# Read train data from file
file = h5py.File('../dataset/val_dataset-1000.h5','r')
print 'validation data shape'
X_val = file['X_train'][:]
Y_val = file['Y_train_merge'][:]
print X_val.shape
print Y_val.shape
file.close()
# Fit model
filepath = "model-regr/loss-decrease-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode='min')
callbacks_list = [checkpoint]
loss_class = regr_model.fit(X_train, Y_train, batch_size=20, epochs=5000, validation_data=(X_val,Y_val), callbacks = callbacks_list, verbose=1)
