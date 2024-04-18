import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import easygui
import Small_Disp_Normal_Corr as sdnc

def fun(k1, k2, path, ff, ROI, resolution_ratio, maximum_deformation, xx, yy):
    vidObj1 = cv2.VideoCapture(path)
    vidObj2 = cv2.VideoCapture(path)
    # 设置视频捕捉对象的帧号
    vidObj1.set(cv2.CAP_PROP_POS_FRAMES, k1 - 1)
    vidObj2.set(cv2.CAP_PROP_POS_FRAMES, k2 - 1)
    # 读取当前帧和计算帧
    success, I0 = vidObj1.read()
    success2, I1 = vidObj2.read()


    # 保存图像的文件名
    U_filename = 'images/1.tif'
    D_filename = 'images/2.tif'
    # 拼接图像保存的完整路径
    path1 = os.path.join('', U_filename)
    path2 = os.path.join('', D_filename)
    # 保存图像
    cv2.imwrite(path1, I0)
    cv2.imwrite(path2, I1)
    fr = 1
    '''
    displacement_smooth---begin
    '''

    '''
     grid_generator---begin
    '''
    im_grid = cv2.imread(U_filename)
    if k2 - k1 == 1:
        if ff == 1:
            # 创建一个指定大小的图像窗口
            plt.figure(figsize=(17, 15))  # 设置图像窗口大小
            plt.imshow(im_grid)
            plt.title("Select area of interest")

        # print("选取感兴趣的区域")
        # 如果 ROI 为 1，则选择感兴趣的区域方式为点击选取矩形的两个角点
        x = []
        y = []
        if ROI == 1:
            print("请点击选取矩形的两个角点")
            for i in range(2):
                point = plt.ginput(1)
                x.append(point[0][0])
                y.append(point[0][1])
                plt.plot(x[-1], y[-1], '+b')
        # 如果 ROI 为 2，则选择感兴趣的区域方式为点击目标的中心，自动生成一个 8x8 的矩形
        elif ROI == 2:
            print("请点击选取目标的中心")
            point = plt.ginput(1)
            x.append(point[0][0])
            y.append(point[0][1])
            x[0] -= 3
            y[0] -= 3
            plt.plot(x[0], y[0], '+b')
            x.append(x[0] + 8)
            y.append(y[0] + 8)

        xx.append(x)
        yy.append(y)

    x = xx[ff - 1]
    y = yy[ff - 1]

    # 计算数组 x 和 y 的最小值和最大值
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # 创建线段的坐标数组
    lowerline = np.array([[xmin, ymin], [xmax, ymin]])
    upperline = np.array([[xmin, ymax], [xmax, ymax]])
    leftline = np.array([[xmin, ymin], [xmin, ymax]])
    rightline = np.array([[xmax, ymin], [xmax, ymax]])

    if k2 - k1 == 1:
        plt.plot(lowerline[:, 0], lowerline[:, 1], '-b')
        plt.plot(upperline[:, 0], upperline[:, 1], '-b')
        plt.plot(leftline[:, 0], leftline[:, 1], '-b')
        plt.plot(rightline[:, 0], rightline[:, 1], '-b')

    res_ratio = resolution_ratio
    if k2 - k1 == 1:
        if ROI == 1:
            # 定义对话框的标题和提示信息
            dlg_title = ''
            prompt = ['输入图像分析的水平(x)分辨率[pixels]:',
                      '输入图像分析的垂直(y)分辨率[pixels]:']
            num_lines = 1
            default = ['10', '10']

            # 显示对话框以获取用户输入
            user_input = easygui.multenterbox(msg="Enter resolution:", title=dlg_title, fields=prompt, values=default)

            # 将用户输入的分辨率存储到 answer 列表中
            res_ratio.append(user_input)
        elif ROI == 2:
            # 如果 ROI 为 2，设置默认分辨率为 3x3
            answer0 = [['3'], ['3']]
            res_ratio.append(answer0)
    # 将 answer1 设置为 answer[ff]
    r_ratio = res_ratio[ff - 1]

    # 从字符串转换为浮点数，并计算xspacing和yspacing
    xspacing = float(r_ratio[0])
    yspacing = float(r_ratio[1])

    # 基于选择的间距“向上”四舍五入 xmin、xmax、ymin 和 ymax
    numXelem = np.ceil((xmax - xmin) / xspacing) - 1
    numYelem = np.ceil((ymax - ymin) / yspacing) - 1

    xmin_new = np.ceil((xmax + xmin) / 2 - ((numXelem / 2) * xspacing))
    xmax_new = np.ceil((xmax + xmin) / 2 + ((numXelem / 2) * xspacing))
    ymin_new = np.ceil((ymax + ymin) / 2 - ((numYelem / 2) * yspacing))
    ymax_new = np.ceil((ymax + ymin) / 2 + ((numYelem / 2) * yspacing))

    # 创建网格
    x, y = np.meshgrid(np.arange(xmin_new, xmax_new + xspacing, xspacing),
                       np.arange(ymin_new, ymax_new + yspacing, yspacing))

    # 获取行和列数
    rows, columns = x.shape

    # 显示图像
    if k2 - k1 == 1:
        plt.title('Selected grid has ' + str(rows * columns) + ' rasterpoints')  # 给图像添加标题
        plt.plot(x, y, '+b')

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
    original = cv2.imread(U_filename)
    deformed = cv2.imread(D_filename)

    # 检查是否为彩色图像，如果是，则转换为灰度图像
    if len(original.shape) > 2:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(deformed.shape) > 2:
        deformed = cv2.cvtColor(deformed, cv2.COLOR_BGR2GRAY)

    if k2 - k1 == 1:
        # 定义对话框的标题和提示信息
        prompt = "输入像素坐标中的最大变形[pixels]："
        dlg_title = ""
        def_value = "15"
        # 使用 easygui 创建对话框，并获取用户输入的值
        ans = easygui.enterbox(prompt, dlg_title, def_value)
        maximum_deformation.append(ans)
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

    u = np.zeros((x_max, y_max))
    v = np.zeros((x_max, y_max))

    plt.show(block=False)

    # 创建一个 xmax 行，ymax 列的二维数组，其中每个元素是一个空数组
    cc = np.empty((int(x_max), int(y_max)), dtype=object)

    # 对数组中的每个元素初始化为空数组
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            cc[i, j] = []
    k = y_max - 1
    for i in range(1, x_max):
        k += 1
        for j in range(1, y_max):
            k += 1
            # u0 = (u[i - 1, j] + u[i, j - 1]) / ((i != 2 or (i == 2 and j == 2)) + (j != 2))
            # v0 = (v[i - 1, j] + v[i, j - 1]) / ((i != 2 or (i == 2 and j == 2)) + (j != 2))
            u0 = 0
            v0 = 0
            u[i, j], v[i, j], c = sdnc.small_disp_normal_corr(int(grid_x[k]), int(grid_y[j]), int(u0), int(v0), original,
                                                              deformed, n, int(sn))
            valid_x[k] = grid_x[k] + u[i, j]
            valid_y[k] = grid_y[k] + v[i, j]
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
