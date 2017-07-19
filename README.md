# deep-direct-regression

### todo
- [ ]  loss / batch_size is strange (2017-07-19)
- [ ]  data augmentation 
    - [ ] [e.g. Kaggle Galaxy Zoo challenge](http://benanne.github.io/2014/04/05/galaxy-zoo.html)
    - [ ] [paper](https://arxiv.org/pdf/1503.07077.pdf)
        * 可视化了很多层，网络可视化做的不错，有时间读一下 
    - [x] crop 320 * 320
    - [ ] crop from sacled image
    - [ ] rotate

### 2017年07月19日12:11:28
1. icdar2017 data augmentation
2. 学习keras fcn
    - [ ] 看看fcn的上采样的图, 有没有分块
    - [ ] fcn resume training 好用吗？
    - [ ] fcn的fit_generator与fit相比的训练速度
3. Read paper: Fully Convolutional Networks for Semantic Segmentation
    - [ ] 什么叫做receptive fields 
        - [ ] Locations in higher layers correspond to the locations in the image they are path-connected to, which are called their receptive fields. 
        - [ ] we add skips between layers to fuse 
            * coarse  
            * semantic  
            * local   
            * appearance information

