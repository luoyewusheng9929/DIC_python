import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from Fun import fun
matplotlib.use('TkAgg')

resolution_ratio = []  #存放输入的水平，垂直分辨率
maximum_deformation = [] #存放输入的最大变形
xx = []  #存放角点的坐标位置
yy = []  #存放角点的坐标位置
# Example usage
path = 'D:\\project\\pycharm\\DIC\\UCC\\UCC_mp\\cx\\MVI_1681_0-1.MOV'
f1 = 30  #开始帧
f2 = 200 #结束帧
nn = 1  #测点数
ROI = 1 # ROI的选择方式，1是点击选矩阵的两个角点，2是针对目标较小的情况，点击目标的中心，在中心周围自动生产一个9*9的网格
bili = 0.232

def dic_multipoint(path, f1, f2, nn, ROI):
    VidObj = cv2.VideoCapture(path)
    frameRate = VidObj.get(cv2.CAP_PROP_FPS)
    t = 1 / frameRate

    fend = f2
    CC = {}  # 创建一个字典来存储 cc
    UQ = {}
    VQ = {}
    k1 = f1
    # 创建图形和子图
    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    for k2 in range(f1 + 1, fend + 1):
        for ff in range(1, nn + 1):
            Uq, Vq, cc, I1 = fun(k1, k2, VidObj, ff, ROI, resolution_ratio, maximum_deformation, xx, yy)
            CC[ff, k2 - k1] = cc
            UQ[ff, k2 - k1] = Uq
            VQ[ff, k2 - k1] = Vq

            dx = []
            dy = []
            for i in range(1, k2 - k1 + 1):
                # 取出键为 (ff, i) 的值
                uq_value = UQ.get((ff, i))
                vq_value = VQ.get((ff, i))

                # 找到非零元素并计算平均值
                uq_nonzero = uq_value[uq_value != 0]
                vq_nonzero = vq_value[vq_value != 0]
                dx.append(np.mean(uq_nonzero))
                dy.append(np.mean(vq_nonzero))

            dx = [-1 * bili * x for x in dx]
            dy = [-1 * bili * y for y in dy]
            count = 0
            # 遍历字典的键
            for key in UQ.keys():
                if key[0] == ff:
                    count += 1
            TT = [t * i for i in range(1, count + 1)]

            # 更新子图数据并绘制
            ax1.clear()
            ax1.imshow(I1)
            ax2.clear()
            ax2.plot(TT, dx)
            ax2.set_title('Plot using dx')
            ax2.set_ylabel('dx')
            ax2.set_xlabel('TT')
            ax3.clear()
            ax3.plot(TT, dy)
            ax3.set_title('Plot using dy')
            ax3.set_ylabel('dy')
            ax3.set_xlabel('TT')

            # 调整子图布局
            plt.tight_layout()

            # 显示图形
            plt.show(block = False)

            print(11)


# Call the function
dic_multipoint(path, f1, f2, nn, ROI)

