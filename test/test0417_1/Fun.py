import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import easygui
import Small_Disp_Normal_Corr as sdnc

def custom_round(x):
    return np.floor(x + 0.5)

def fun(k1, k2, path, ff, ROI, resolution_ratio, maximum_deformation, xx, yy):
    vidObj1 = cv2.VideoCapture(path)
    vidObj2 = cv2.VideoCapture(path)
    # 设置视频捕捉对象的帧号
    vidObj1.set(cv2.CAP_PROP_POS_FRAMES, k1 - 1)
    vidObj2.set(cv2.CAP_PROP_POS_FRAMES, k2 - 1)
    # 读取当前帧和计算帧
    success, I0 = vidObj1.read()
    success2, I1 = vidObj2.read()

    fr = 1
    '''
    displacement_smooth---begin
    '''

    '''
     grid_generator---begin
    '''
    x = xx[ff - 1]
    y = yy[ff - 1]

    # 计算数组 x 和 y 的最小值和最大值
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    res_ratio = resolution_ratio

    # 将 answer1 设置为 answer[ff]
    r_ratio = res_ratio[ff - 1]

    # 从字符串转换为浮点数，并计算xspacing和yspacing
    xspacing = float(r_ratio[0])
    yspacing = float(r_ratio[1])

    # 基于选择的间距“向上”四舍五入 xmin、xmax、ymin 和 ymax
    numXelem = np.ceil((xmax - xmin) / xspacing) - 1
    numYelem = np.ceil((ymax - ymin) / yspacing) - 1

    xmin_new = custom_round((xmax + xmin) / 2.0 - ((numXelem / 2.0) * xspacing))
    xmax_new = custom_round((xmax + xmin) / 2.0 + ((numXelem / 2.0) * xspacing))
    ymin_new = custom_round((ymax + ymin) / 2.0 - ((numYelem / 2.0) * yspacing))
    ymax_new = custom_round((ymax + ymin) / 2.0 + ((numYelem / 2.0) * yspacing))

    # 创建网格
    x, y = np.meshgrid(np.arange(xmin_new, xmax_new + xspacing, xspacing),
                       np.arange(ymin_new, ymax_new + yspacing, yspacing))

    # 获取行和列数
    rows, columns = x.shape

    # 初始化一个空数组，用于存放结果
    grid_x = []
    grid_y = []

    # 将矩阵展平为一个一维列表
    for j in range(columns):
        for i in range(rows):
            grid_x.append(x[i][j])
            grid_y.append(y[i][j])

    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)

    '''
     grid_generator---end
    '''

    # 读取原始图像和变形后的图像
    original = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    deformed = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    ans = maximum_deformation[ff - 1]
    # 将 ans 转换为字符串，然后转换为浮点数
    sn = float(ans)
    # 乘以 fr
    sn *= fr

    # 将 grid_x 和 grid_y 乘以 fr
    grid_x *= fr
    grid_y *= fr

    # 将 grid_x 和 grid_y 的值复制给 valid_x 和 valid_y
    valid_x = grid_x.copy()
    valid_y = grid_y.copy()

    n = 10
    i = 2
    j = 2
    p = len(grid_x)

    grid_side_y = grid_y[1] - grid_y[0]

    y_max = (grid_y[-1] - grid_y[0]) / grid_side_y + 1
    x_max = p / y_max
    x_max = round(x_max)
    y_max = round(y_max)

    grid_side_x = grid_x[y_max] - grid_x[0]

    u = np.zeros((y_max, x_max))
    v = np.zeros((y_max, x_max))

    plt.show(block=False)

    # 创建一个 xmax 行，ymax 列的二维数组，其中每个元素是一个空数组
    cc = np.empty((int(y_max), int(x_max)), dtype=object)

    # 对数组中的每个元素初始化为空数组
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            cc[i, j] = []
    k = y_max - 1
    for j in range(1, x_max):
        k += 1
        for i in range(1, y_max):
            k += 1
            # u0 = (u[i - 1, j] + u[i, j - 1]) / ((i != 2 or (i == 2 and j == 2)) + (j != 2))
            # v0 = (v[i - 1, j] + v[i, j - 1]) / ((i != 2 or (i == 2 and j == 2)) + (j != 2))
            u0 = 0
            v0 = 0
            u[i][j], v[i][j], c = sdnc.small_disp_normal_corr(int(grid_x[k]) - 1, int(grid_y[i]) - 1, int(u0), int(v0), original,
                                                              deformed, n, int(sn))
            valid_x[k] = grid_x[k] + u[i][j]
            valid_y[k] = grid_y[k] + v[i][j]
            cc[i][j] = c
    # valid_xy = np.column_stack((valid_x, valid_y))
    # grid_xy = np.column_stack((grid_x, grid_y))
    # input_correl = cp.cpcorr(valid_xy, grid_xy, deformed, original)
    # input_correl_x = input_correl[:, 0]
    # input_correl_y = input_correl[:, 1]
    # # 将 input_correl_x 和 input_correl_y 重新整形为二维数组
    # Uq = np.reshape(input_correl_x - grid_x, (y_max, x_max))
    # Vq = np.reshape(input_correl_y - grid_y, (y_max, x_max))
    Uq = u
    Vq = v

    cc = cc[1:, 1:]
    return Uq, Vq, cc, I1
