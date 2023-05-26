import numpy as np

def cutout(img, cut_width, holes=1):
    aug_img= img.copy()
    width= img.shape[1]
    for i in range(holes):
        indx= np.random.randint(width-cut_width)
        aug_img[:,indx:indx+cut_width]= 0
    return aug_img

def cutmix(img_1, img_2, cut_width):
    width= img_1.shape[1]
    cut_width= np.random.randint(cut_width+-20, cut_width+20)
    indx= np.random.randint(width-cut_width)
    img_1[:, indx:indx+cut_width]= img_2[:, indx:indx+cut_width]
    return img_1