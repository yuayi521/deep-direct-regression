import re
import numpy as np
from matplotlib import pyplot as plt
import cv2


def heatmap_cls(ndarray, img_data):
    """

    :param ndarray:
    :param img_data:
    :return:
    """
    # new_list = []
    # list_nd = ndarray.tolist()
    # while list_nd:
    #     new_list.append(list_nd.pop())
    # new_nd = np.array(new_list)
    img = cv2.imread(img_data['imagePath'])
    plt.subplot(221)
    x = []
    y = []
    for i in xrange(1, 81):
        x.append(i)
        y.append(i)
    # intensity = np.transpose(new_nd).tolist()
    # intensity = ndarray.tolist()
    intensity = np.rot90(ndarray, 1).tolist()
    # setup the 2D grid with Numpy
    x, y = np.meshgrid(x, y)
    # convert intensity (list of lists) to a numpy array for plotting
    intensity = np.array(intensity)
    # now just plug the data into pcolormesh, it's that easy!
    plt.pcolormesh(x, y, intensity)
    plt.colorbar()  # need a colorbar to show the intensity scale
    plt.subplot(222)
    plt.imshow(img)
    plt.show()


def plot_model(cls_para, reg_para, val_cls_para, val_reg_para):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for cls loss
    axs[0].plot(range(1, len(cls_para) + 1), cls_para)
    axs[0].plot(range(1, len(val_cls_para) + 1), val_cls_para)
    axs[0].set_title('cls loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(cls_para) + 1), len(cls_para) / 10)
    axs[0].legend(['train'], loc='best')
    # summarize history for regr loss
    axs[1].plot(range(1, len(reg_para) + 1), reg_para)
    axs[1].plot(range(1, len(val_reg_para) + 1), val_reg_para)
    axs[1].set_title('regr Loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(reg_para) + 1), len(reg_para) / 10)
    axs[1].legend(['train'], loc='best')
    plt.show()


if __name__ == '__main__':
    #
    #
    file_content = []
    cls = []
    reg = []
    val_cls = []
    val_reg = []
    with open('../result/0711') as f:
        for line in f:
            file_content.append(line)

    print len(file_content)
    for idx in range(len(file_content)):
        pat = re.search(r'^100/100.*cls_loss: (0\.\d*) - lambda_1_loss: (\d*\.\d*) - .*- '
                        r'val_cls_loss: (\d\.\d*).*val_lambda_1_loss: (\d\.\d*) - val_cls_acc.*',
                        file_content[idx])
        if pat:
            cls.append(pat.group(1))
            reg.append(pat.group(2))
            val_cls.append(pat.group(3))
            val_reg.append(pat.group(4))

    plot_model(cls, reg, val_cls, val_reg)
