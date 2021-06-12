import cv2
import numpy as np
from skimage.measure import label

def fill(im_in):
    th, im_th = cv2.threshold(im_in, 200, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()  # Copy the thresholded image.
    h, w = im_th.shape[:2]  # Mask used to flood filling.
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Notice the size needs to be 2 pixels than the image.
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)  # Floodfill from point (0, 0)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # Invert floodfilled image
    im_out = im_th | im_floodfill_inv  # Combine the two images to get the foreground.
    return im_out

def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    max_label = 0
    max_num = 0
    if num <= 1 :
        return bw_img
    else:
        for j in range(num):
            i = j + 1
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
    return lcc.astype(np.float64)

def imfill_select_whole_seq(im):
    for i in range(im.shape[1]):
        im_i = im[0, i]
        ED_endo = (1 - (im_i == 1).astype(np.uint8)) * 255
        ED_endo_post_fill = fill(ED_endo) / 255
        ED_endo_post_select_lcc = largestConnectComponent(ED_endo_post_fill)
        ED_epi = ((im_i == 2).astype(np.float64) - ED_endo_post_select_lcc > 0).astype(
            np.float64) + ED_endo_post_select_lcc
        ED_epi = (1 - ED_epi.astype(np.uint8)) * 255
        ED_epi_post_fill = fill(ED_epi) / 255
        ED_myo_post_select_lcc = largestConnectComponent(ED_epi_post_fill) * 2 - ED_endo_post_select_lcc * 2
        ED_la = (1 - (im_i == 3).astype(np.uint8)) * 255
        ED_la_post_fill = fill(ED_la) / 255
        ED_la_post_select_lcc = largestConnectComponent(ED_la_post_fill) * 3
        im[0, i] = ED_endo_post_select_lcc + ED_myo_post_select_lcc + ED_la_post_select_lcc
    return im.astype(np.long)
