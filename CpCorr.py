#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on --/--/20--

@author: ---
Revised by Charlie Bourigault
@contact: bourigault.charlie@gmail.com

Please report issues and request on the GitHub project from ChrisEberl (Python_DIC)
More details regarding the project on the GitHub Wiki : https://github.com/ChrisEberl/Python_DIC/wiki

Current File: This file has been translated, adapted and further developed from 'Digital Image Correlation and Tracking' for Matlab exchanged by Melanie Senn on Mathworks
"""
import scipy

'''
这段代码实现了一个名为 `cpcorr` 的函数，用于在数字图像中进行特征点的亚像素级配准。具体来说，它是为了在两幅图像中匹配输入和基准点集。

这个函数的输入包括：

- `InputPoints` 和 `BasePoints`：输入和基准点的坐标，用于配准。
- `Input` 和 `Base`：输入和基准图像。

该函数首先解析输入数据，然后根据输入的参数，计算两幅图像中感兴趣区域的矩形。接着，它在这些区域内计算归一化互相关来寻找匹配。如果匹配结果符合要求，则进行亚像素级的修正。

在这段代码中，主要涉及的部分有：

- 使用 OpenCV 提供的 `cv2.matchTemplate` 函数计算归一化互相关，找到两幅图像中的最佳匹配。
- 对归一化互相关结果进行亚像素级的修正，以获得更精确的匹配位置。
- 对匹配结果进行了一些限制和检查，例如边缘区域处理、图像有效性检查、标准差检查等。

最后，函数返回了修正后的输入点集、匹配的标准差、相关系数、匹配的错误信息等。
'''

import numpy as np, cv2
from scipy import interpolate


def cpcorr(InputPoints, BasePoints, Input, Base):
    CORRSIZE = 5
    [xymoving_in, xyfixed_in, moving, fixed] = ParseInputs(InputPoints, BasePoints, Input, Base)

    # get all rectangle coordinates
    rects_moving = np.array(calc_rects(xymoving_in, CORRSIZE, moving)).astype(int)
    rects_fixed = np.array(calc_rects(xyfixed_in, 2 * CORRSIZE, fixed)).astype(int)
    ncp = len(np.atleast_1d(xymoving_in))

    xymoving = xymoving_in  # initialize adjusted control points matrix

    for icp in range(ncp):

        if (rects_moving[2][icp] == 0 and rects_moving[3][icp] == 0) or (
                rects_fixed[2][icp] == 0 and rects_moving[3][icp] == 0):
            # near edge, unable to adjust
            # print 'CpCorr : Edge area. No Adjustement.'
            continue

        sub_moving = moving[rects_moving[1][icp]:rects_moving[1][icp] + rects_moving[3][icp],
                     rects_moving[0][icp]:rects_moving[0][icp] + rects_moving[2][icp]]
        sub_fixed = fixed[rects_fixed[1][icp]:rects_fixed[1][icp] + rects_fixed[3][icp],
                    rects_fixed[0][icp]:rects_fixed[0][icp] + rects_fixed[2][icp]]


        # make sure the image data exist
        if sub_moving.shape[0] == 0 or sub_moving.shape[1] == 0 or sub_fixed.shape[0] == 0 or sub_fixed.shape[1] == 0:
            # print 'CpCorr : Marker out of image.'
            continue

        # make sure finite
        if np.logical_or(np.any(np.isfinite(sub_moving[:]) == False), np.any(np.isfinite(sub_fixed[:])) == False):
            # NaN or Inf, unable to adjust
            # print 'CpCorr : Wrong Number. No Adjustement.'
            continue

        # check that template rectangle moving has nonzero std
        if np.std(sub_moving[:]) == 0:
            # zero standard deviation of template image, unable to adjust
            # print 'CpCorr : No Std Dev. No Adjustement.'
            continue

        # norm_cross_corr = cv2.matchTemplate(sub_moving, sub_fixed, cv2.TM_CCOEFF_NORMED)
        norm_cross_corr = cv2.matchTemplate(sub_moving, sub_fixed, cv2.TM_CCORR_NORMED)
        # norm_cross_corr=scipy.signal.correlate2d(sub_fixed, sub_moving)
        # norm_cross_corr=sklearn.preprocessing.normalize(norm_cross_corr, norm='l2', axis=1, copy=True)
        # norm_cross_corr=match_template(sub_fixed,sub_moving)

        # get subpixel resolution from cross correlation
        subpixel = True
        [xpeak, ypeak, corrcoef] = findpeak(norm_cross_corr, subpixel)
        xpeak = float(xpeak)
        ypeak = float(ypeak)

        # eliminate any poor correlations
        THRESHOLD = 0.5
        if (corrcoef < THRESHOLD):
            # low correlation, unable to adjust
            # print 'CpCorr : Low Correlation. Marker avoided.'
            continue

        # offset found by cross correlation
        corroffset = [xpeak - CORRSIZE, ypeak - CORRSIZE]

        # eliminate any big changes in control points
        if corroffset[0] > (CORRSIZE - 1) or corroffset[1] > (CORRSIZE - 1):
            # peak of norxcorr2 not well constrained, unable to adjust
            # print 'CpCorr : Peak not well constrained. No adjustement'
            continue

        movingfractionaloffset = np.array([xymoving[icp, :] - np.around(xymoving[icp, :])])
        fixedfractionaloffset = np.array([xyfixed_in[icp, :] - np.around(xyfixed_in[icp, :])])

        # adjust control point
        xymoving[icp, :] = xymoving[icp, :] - movingfractionaloffset - corroffset + fixedfractionaloffset
        # xymoving[icp,:] = xymoving[icp,:] - corroffset

    return xymoving


def calc_rects(xy, halfwidth, img):
    # Calculate rectangles so imcrop will return image with xy coordinate inside center pixel
    default_width = 2 * halfwidth
    default_height = default_width
    [row, col] = img.shape

    # xy specifies center of rectangle, need upper left
    upperleft = np.around(xy) - halfwidth

    # need to modify for pixels near edge of images
    left = upperleft[:, 0]
    upper = upperleft[:, 1]
    right = left + default_width
    lower = upper + default_height

    width = default_width * np.ones(np.shape(upper))
    height = default_height * np.ones(np.shape(upper))

    # check edges for coordinates outside image
    [upper, height] = adjust_lo_edge(upper, 1, height)
    [lower, height] = adjust_hi_edge(lower, row, height)
    [left, width] = adjust_lo_edge(left, 1, width)
    [right, width] = adjust_hi_edge(right, col, width)

    # set width and height to zero when less than default size
    # iw = np.where(width < default_width)[0]
    # ih = np.where(height < default_height)[0]
    # idx = np.unique(np.concatenate((iw, ih)))
    #
    # width[idx] = 0
    # height[idx] = 0

    rect = [left.astype(int), upper.astype(int), width.astype(int), height.astype(int)]
    return rect


def adjust_lo_edge(coordinates, edge, breadth):
    for indx in range(0, len(coordinates)):
        if coordinates[indx] < edge:
            # breadth[indx] = breadth[indx] - np.absolute(coordinates[indx]-edge)
            breadth[indx] = 0
            coordinates[indx] = edge
    return coordinates, breadth


def adjust_hi_edge(coordinates, edge, breadth):
    for indx in range(0, len(coordinates)):
        if coordinates[indx] > edge:
            # breadth[indx] = breadth[indx] - np.absolute(coordinates[indx]-edge)
            breadth[indx] = 0
            coordinates[indx] = edge
    return coordinates, breadth


def ParseInputs(InputPoints, BasePoints, Input, Base):
    xymoving_in = InputPoints
    xyfixed_in = BasePoints
    if xymoving_in.shape[1] != 2 or xyfixed_in.shape[1] != 2:
        raise ValueError("cpMatrixMustBeMby2")
    moving = Input
    fixed = Base
    return xymoving_in, xyfixed_in, moving, fixed


# sub pixel accuracy by 2D polynomial fit (quadratic)
def findpeak(f, subpixel):
    # Get absolute peak pixel

    max_f = np.amax(f)
    [xpeak, ypeak] = np.unravel_index(f.argmax(), f.shape)  # coordinates of the maximum value in f

    if subpixel == False or xpeak == 0 or xpeak == np.shape(f)[0] - 1 or ypeak == 0 or ypeak == np.shape(f)[
        1] - 1:  # on edge
        # print 'CpCorr : No Subpixel Adjustement.'
        return ypeak, xpeak, max_f  # return absolute peak

    else:
        # fit a 2nd order polynomial to 9 points
        # using 9 pixels centered on irow,jcol
        u = f[xpeak - 1:xpeak + 2, ypeak - 1:ypeak + 2]
        u = np.reshape(np.transpose(u), (9, 1))
        x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        x = np.reshape(x, (9, 1))
        y = np.reshape(y, (9, 1))

        # u(x,y) = A(0) + A(1)*x + A(2)*y + A(3)*x*y + A(4)*x^2 + A(5)*y^2
        X = np.hstack((np.ones((9, 1)), x, y, x * y, x ** 2, y ** 2))
        # u = X*A

        # A = np.linalg.lstsq(X,u, rcond=1e-1)
        A = np.linalg.lstsq(X, u, rcond=1e-20)

        e = A[1]  # residuals returned by Linalg Lstsq
        A = np.reshape(A[0], (6, 1))  # A[0] array of least square solution to the u = AX equation

        # get absolute maximum, where du/dx = du/dy = 0
        x_num = (-A[2] * A[3] + 2 * A[5] * A[1])
        y_num = (-A[3] * A[1] + 2 * A[4] * A[2])

        den = (A[3] ** 2 - 4 * A[4] * A[5])
        x_offset = x_num / den
        y_offset = y_num / den

        # print x_offset, y_offset
        if np.absolute(x_offset) > 1 or np.absolute(y_offset) > 1:
            # print 'CpCorr : Subpixel outside limit. No adjustement'
            # adjusted peak falls outside set of 9 points fit,
            return ypeak, xpeak, max_f  # return absolute peak

        # x_offset = np.round(10*x_offset)/10
        # y_offset = np.round(10*y_offset)/10
        x_offset = np.around(x_offset, decimals=4)
        y_offset = np.around(y_offset, decimals=4)

        xpeak = xpeak + x_offset
        ypeak = ypeak + y_offset
        # print xpeak, ypeak

        # Calculate extremum of fitted function
        # 将 A 展平为一维数组
        A_flat = A[:, 0]
        max_f = np.dot(np.array([1, x_offset[0], y_offset[0], x_offset[0] * y_offset[0], x_offset[0] ** 2, y_offset[0] ** 2]), A_flat)
        # max_f = np.dot([1, x_offset, y_offset, x_offset*y_offset, x_offset**2, y_offset**2],A)
        max_f = np.absolute(max_f)

    return ypeak, xpeak, max_f


# sub pixel accuracy by upsampling and interpolation
def findpeak2(f, subpixel):
    stdx = 1e-4
    stdy = 1e-4

    kernelsize = 3

    # get absolute peak pixel
    max_f = np.amax(f)
    [xpeak, ypeak] = np.unravel_index(f.argmax(), f.shape)

    if subpixel == False or xpeak < kernelsize or xpeak > np.shape(f)[0] - kernelsize or ypeak < kernelsize or ypeak > \
            np.shape(f)[1] - kernelsize:  # on edge
        return xpeak, ypeak, stdx, stdy, max_f  # return absolute peak
    else:
        # determine sub pixel accuracy by upsampling and interpolation
        fextracted = f[xpeak - kernelsize:xpeak + kernelsize + 1, ypeak - kernelsize:ypeak + kernelsize + 1]
        totalsize = 2 * kernelsize + 1
        upsampling = totalsize * 10 + 1
        # step=2/upsampling
        x = np.linspace(-kernelsize, kernelsize, totalsize)
        # [X,Y]=np.meshgrid(x,x)
        xq = np.linspace(-kernelsize, kernelsize, upsampling)
        # [Xq,Yq]=np.meshgrid(xq,xq)

        bilinterp = interpolate.interp2d(x, x, fextracted, kind='cubic')
        fq = bilinterp(xq, xq)
        # splineint = RectBivariateSpline(x, x, fextracted, kx=3, ky=3, s=0)
        # fq=splineint(xq,xq)
        # fq=griddata((x, x), fextracted, (Xq, Yq), method='cubic')

        max_f = np.amax(fq)
        [xpeaknew, ypeaknew] = np.unravel_index(fq.argmax(), fq.shape)

        # xoffset=Xq[0,xpeaknew]
        # yoffset=Yq[ypeaknew,0]
        xoffset = xq[xpeaknew]
        yoffset = xq[ypeaknew]

        # return only one-thousandths of a pixel precision
        xoffset = np.round(1000 * xoffset) / 1000
        yoffset = np.round(1000 * yoffset) / 1000
        xpeak = xpeak + xoffset
        ypeak = ypeak + yoffset

        # peak width (full width at half maximum)
        scalehalfwidth = 1.1774
        fextractedx = np.mean(fextracted, 0)
        fextractedy = np.mean(fextracted, 1)
        stdx = scalehalfwidth * np.std(fextractedx)
        stdy = scalehalfwidth * np.std(fextractedy)

    return xpeak, ypeak, stdx, stdy, max_f


# sub pixel accuracy by centroid
def findpeak3(f, subpixel):
    stdx = 1e-4
    stdy = 1e-4

    kernelsize = 3

    # get absolute peak pixel
    max_f = np.amax(f)
    [xpeak, ypeak] = np.unravel_index(f.argmax(), f.shape)

    if subpixel == False or xpeak < kernelsize or xpeak > np.shape(f)[0] - kernelsize or ypeak < kernelsize or ypeak > \
            np.shape(f)[1] - kernelsize:  # on edge
        return xpeak, ypeak, stdx, stdy, max_f  # return absolute peak
    else:
        # determine sub pixel accuracy by centroid
        fextracted = f[xpeak - kernelsize:xpeak + kernelsize + 1, ypeak - kernelsize:ypeak + kernelsize + 1]
        fextractedx = np.mean(fextracted, 0)
        fextractedy = np.mean(fextracted, 1)
        x = np.arange(-kernelsize, kernelsize + 1, 1)
        y = np.transpose(x)

        xoffset = np.dot(x, fextractedx)
        yoffset = np.dot(y, fextractedy)

        # return only one-thousandths of a pixel precision
        xoffset = np.round(1000 * xoffset) / 1000
        yoffset = np.round(1000 * yoffset) / 1000
        xpeak = xpeak + xoffset
        ypeak = ypeak + yoffset

        # 2D linear interpolation
        bilinterp = interpolate.interp2d(x, x, fextracted, kind='linear')
        max_f = bilinterp(xoffset, yoffset)

        # peak width (full width at half maximum)
        scalehalfwidth = 1.1774
        stdx = scalehalfwidth * np.std(fextractedx)
        stdy = scalehalfwidth * np.std(fextractedy)

    return xpeak, ypeak, stdx, stdy, max_f
