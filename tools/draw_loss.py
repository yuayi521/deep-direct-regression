import re
import matplotlib.pyplot as plt
import numpy as np


def plot_model(cls_para, reg_para):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for cls loss
    axs[0].plot(range(1, len(cls_para) + 1), cls_para)
    axs[0].set_title('cls loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(cls_para) + 1), len(cls_para) / 10)
    axs[0].legend(['train'], loc='best')
    # summarize history for regr loss
    axs[1].plot(range(1, len(reg_para) + 1), reg_para)
    axs[1].set_title('regr Loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(reg_para) + 1), len(reg_para) / 10)
    axs[1].legend(['train'], loc='best')
    plt.show()


if __name__ == '__main__':
    file_content = []
    cls = []
    reg = []
    # with open('ramdom_crop') as f:
    with open('center_crop') as f:
        for line in f:
            file_content.append(line)

    print len(file_content)
    for idx in range(len(file_content)):
        # pat = re.search(r'^1000/1000.*cls_loss: (0\.\d*) - lambda_1_loss: (\d*\.\d*) - .*', file_content[idx])
        pat = re.search(r'^400/400.*cls_loss: (0\.\d*) - lambda_1_loss: (\d*\.\d*) - .*', file_content[idx])
        if pat:
            cls.append(pat.group(1))
            reg.append(pat.group(2))

    plot_model(cls, reg)
