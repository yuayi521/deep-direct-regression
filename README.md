# deep-direct-regression

### todo
- [ ]  loss / batch_size is strange (2017-07-19)
- [ ]  data augmentation 
    - [ ] [e.g. Kaggle Galaxy Zoo challenge](http://benanne.github.io/2014/04/05/galaxy-zoo.html)
    - [ ] [paper](https://arxiv.org/pdf/1503.07077.pdf)
        * 可视化了很多层，网络可视化做的不错，有时间读一下 
    - [x] crop 320 * 320
    - [ ] crop from sacled image
    - [x] rotate
- [ ] 网络参数初始化，有一次resume training, loss突然飘的特别高

### 2017年07月19日12:11:28
- [ ] icdar2017 data augmentation
- [ ] 学习keras fcn
    - [ ] 看看fcn的上采样的图, 有没有分块
    - [ ] fcn resume training 好用吗？
    - [ ] fcn的fit_generator与fit相比的训练速度


- [ ] Read paper: Fully Convolutional Networks for Semantic Segmentation
    - [ ] 什么叫做receptive fields 
        - [ ] Locations in higher layers correspond to the locations in the image they are path-connected to, which are called their receptive fields. 
    - [ ] we add skips between layers to fuse 
        * coarse  
        * semantic  
        * local   
        * appearance information
    - [ ] loss是什么？
        - [ ] per-pixel multinomial logistic loss
    - [ ] 最后是一个分类问题吗？把图片分为21类？？ 
- [ ] FCN-keras 实现
    - [ ] padding的目的和作用,我大概明白了
        - [ ]
            ```
            x = np.lib.pad(x, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=0.)
            y = np.lib.pad(y, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=self.label_cval)
            ```
        - [ ]
            ```
            a = np.ones((4, 3, 2))
            # npad is a tuple of (n_before, n_after) for each dimension
            npad = ((0, 0), (1, 2), (2, 1))
            b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)
            ```

### 2017-07-20 11:51:22


### Python 技巧
- [ ]
    - [ ] os.path.realpath(path)
        - [ ] Return the canonical(权威的) path of the specified filename, eliminating any symbolic links encountered in the path     
    - [ ] 在python下，获取当前执行主脚本的方法有两个：sys.argv[0]和 \_\_file\_\_
    ```   
    current_dir = os.path.dirname(os.path.realpath(__file__))   
    save_path = os.path.join(current_dir, 'Models/' + model_name_)   
    # I should learn this method   
    if os.path.exists(save_path) is False:   
        os.mkdir(save_path)   
    ```

### Keras
- [ ] from keras.preprocessing.image import image, 这个图像预处理工具包，挺有用的
    - [ ] img_to_arr
    - [ ] arr_to_img
    - [ ] list_pictures
    - [ ] DirectoryIterators class
        - [ ] 迭代器，能从硬盘上读取图片
