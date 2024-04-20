import numpy as np

def rgb2gray(im):
    im=np.array(im,dtype='double')

    im_gray=im[:,:,2]*0.299+im[:,:,1]*0.587+im[:,:,0]*0.114
    im_gray=np.array(im_gray,dtype='uint8')
    return im_gray

