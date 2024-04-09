import cv2
import numpy as np

def imcrop(img, rect_roi):
    return img[int(rect_roi[1]) : int(rect_roi[1] + rect_roi[3]),
        int(rect_roi[0]) : int(rect_roi[0] + rect_roi[2])]

def small_disp_normal_corr(x, y, u0, v0, original, deformed, n, sn):
    # 在原始图像中选择pepper below the onion
    sub_original = original[y - sn:y + sn + 1, x - sn:x + sn + 1]
    rect_original = (x - sn, y - sn, 2 * sn + 1, 2 * sn + 1)
    # print('rect_original: ', rect_original)

    # sub_deformed = deformed[y + v0 - 2 * sn:y + v0 + 2 * sn + 1, x + u0 - 2 * sn:x + u0 + 2 * sn + 1]
    # rect_deformed = (x + u0 - 2 * sn, y + v0 - 2 * sn, 4 * sn + 1, 4 * sn + 1)
    sub_deformed = deformed[y - 2 * sn:y + 2 * sn + 1, x - 2 * sn:x + 2 * sn + 1]
    rect_deformed = (x - 2 * sn, y - 2 * sn, 4 * sn + 1, 4 * sn + 1)
    # print('rect_deformed: ', rect_deformed)

    # 显示裁剪后的图像
    # cv2.imshow('Original Subimage', sub_original)
    # cv2.imshow('Deformed Subimage', sub_deformed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    offset = []
    c = []
    # 比较两个数组的大小
    if np.array(sub_original.shape)[0] < np.array(sub_deformed.shape)[0] or \
            np.array(sub_original.shape)[1] < np.array(sub_deformed.shape)[1]:

        # 使用模板匹配
        c = cv2.matchTemplate(sub_deformed, sub_original, cv2.TM_CCOEFF_NORMED)
        # print('c', c)
        # print('c.max:', np.max(c))
        # print('c.shape:', c.shape)

        max_corr_index = np.unravel_index(np.argmax(c), c.shape)

        # 获取最大值位置的坐标
        yp, xp = max_corr_index
        # print('max_corr_index:', max_corr_index)
        # 计算偏移量 画图
        h, w = sub_original.shape
        # 计算矩形框的对角线顶点坐标
        box_top_left = (xp, yp)
        box_bottom_right = (xp + w, yp + h)
        # 在sub_deformed图像上绘制白色框
        sub_deformed_with_box = cv2.rectangle(sub_deformed.copy(), box_top_left, box_bottom_right, color=(255, 255, 255),
                                              thickness=1)
        sub_deformed_with_box = cv2.circle(sub_deformed_with_box, (xp, yp), radius=2, color=(255, 255, 255), thickness=-1)
        # 画一条水平线
        sub_deformed_with_box = cv2.line(sub_deformed_with_box, (0, yp), (sub_deformed_with_box.shape[1], yp),
                                         color=(255, 255, 255), thickness=1)
        # 画一条垂直线
        sub_deformed_with_box = cv2.line(sub_deformed_with_box, (xp, 0), (xp, sub_deformed_with_box.shape[0]),
                                         color=(255, 255, 255), thickness=1)
        # 显示结果
        # cv2.imshow('Sub Deformed Image with Box', sub_deformed_with_box)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # 获取最大值位置的坐标
        yp, xp = max_corr_index

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
        # print('p:', p)

        ypeak = p[0]
        xpeak = p[1]
        rect_offset = [(rect_deformed[1] - rect_original[1]), (rect_deformed[0] - rect_original[0])]

        offset = [ypeak + rect_offset[0], xpeak + rect_offset[1]]
        # print('offset:', offset)
    else:
        offset[1] = u0
        offset[0] = v0

    return offset[1], offset[0], c
