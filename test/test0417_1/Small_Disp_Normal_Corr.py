import numpy as np
from normxcorr2 import normxcorr2
import cv2
def imcrop(img, rect_roi):
    return img[int(rect_roi[1]) : int(rect_roi[1] + rect_roi[3]),
        int(rect_roi[0]) : int(rect_roi[0] + rect_roi[2])]

def small_disp_normal_corr(x, y, u0, v0, original, deformed, n, sn):

    rect_original = (x - sn, y - sn, 2 * sn + 2, 2 * sn + 2)
    sub_original = imcrop(original, rect_original)
    rect_deformed = (x - 2 * sn, y - 2 * sn, 4 * sn + 2, 4 * sn + 2)
    sub_deformed = imcrop(deformed, rect_deformed)

    offset = []
    c = []
    # 比较两个数组的大小
    if np.array(sub_original.shape)[0] < np.array(sub_deformed.shape)[0] or \
            np.array(sub_original.shape)[1] < np.array(sub_deformed.shape)[1]:

        c = normxcorr2(sub_original, sub_deformed)
        # 使用模板匹配
        # c = cv2.matchTemplate(sub_deformed, sub_original, cv2.TM_CCOEFF_NORMED)
        # c = cv2.matchTemplate(sub_deformed, sub_original, cv2.TM_CCORR_NORMED)


        sc = np.shape(c)
        w = np.concatenate(
            (np.linspace(0.8, 1, int(np.ceil(sc[0] / 2))), np.linspace(1, 0.8, int(np.floor(sc[0] / 2)))))
        h = np.concatenate(
            (np.linspace(0.8, 1, int(np.ceil(sc[1] / 2))), np.linspace(1, 0.8, int(np.floor(sc[1] / 2)))))
        w = w[np.newaxis, :]
        h = h[np.newaxis, :]
        c = c * (w.T @ h)

        max_corr_index = np.unravel_index(np.argmax(c), c.shape)

        # 获取最大值位置的坐标
        yp, xp = max_corr_index
        if yp == 0 or yp == sc[0] - 1 or xp == 0 or xp == sc[0] - 1:
            print('Maximum position at matrix border. No subsample approximation possible.')
            return sc[1] / 2, sc[0] / 2, c


        K = c[yp - 1:yp + 2, xp - 1:xp + 2]
        # print('K:', K)

        # approximate polynomial parameter
        # 近似多项式参数

        # 这段代码通过特定的权重对矩阵 K 中的元素进行加权求和，以提取或强调图像中的某些特征或边缘。

        # 计算参数 a
        a = (K[1, 0] + K[0, 0] - 2 * K[0, 1] + K[0, 2] - 2 * K[2, 1] - 2 * K[1, 1] + K[1, 2] + K[2, 0] + K[2, 2])

        # 计算参数 b
        b = (K[2, 2] + K[0, 0] - K[0, 2] - K[2, 0])

        # 计算参数 c
        c_para = (-K[0, 0] + K[0, 2] - K[1, 0] + K[1, 2] - K[2, 0] + K[2, 2])

        # 计算参数 e
        e = (-2 * K[1, 0] + K[0, 0] + K[0, 1] + K[0, 2] + K[2, 1] - 2 * K[1, 1] - 2 * K[1, 2] + K[2, 0] + K[2, 2])

        # 计算参数 f
        f = (-K[0, 0] - K[0, 1] - K[0, 2] + K[2, 0] + K[2, 1] + K[2, 2])

        # (ys, xs) is subpixel shift of peak location relative to point (2,2)
        # ys 和 xs 表示的是峰值位置在垂直和水平方向上相对于矩阵 K 中心点的子像素偏移量。
        # 如果 ys 和 xs 都为零，那么峰值就位于矩阵的中心点（2，2）上。如果它们不为零，那么峰值就位于中心点的某个子像素偏移位置上。

        # 计算 ys
        ys = (6 * b * c_para - 8 * a * f) / (16 * e * a - 9 * b ** 2)

        # 计算 xs
        xs = (6 * b * f - 8 * e * c_para) / (16 * e * a - 9 * b ** 2)


        p = [ys + yp, xs + xp]
        ypeak = p[0]
        xpeak = p[1]
        corr_offset = [(xpeak - sub_original.shape[1]), (ypeak - sub_original.shape[0])]
        rect_offset = [(rect_deformed[0] - rect_original[0]), (rect_deformed[1] - rect_original[1])]
        offset = [corr_offset[0] + rect_offset[0], corr_offset[1] + rect_offset[1]]


    else:
        offset[0] = u0
        offset[1] = v0

    return offset[0], offset[1], c
