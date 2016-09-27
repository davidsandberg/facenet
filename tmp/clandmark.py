from py_flandmark import PyFlandmark
from py_featurePool import PyFeaturePool

# flandmark = PyFlandmark("../../../data/flandmark_model.xml", False)
# flandmark = PyFlandmark("../../../data/FRONTAL_21L.xml", False)
flandmark = PyFlandmark("../../../data/300W/FDPM.xml", False)

bw = flandmark.getBaseWindowSize()
featurePool = PyFeaturePool(bw[0], bw[1], None)
featurePool.addFeatuaddSparseLBPfeatures()

flandmark.setFeaturePool(featurePool)



import time
import numpy as np
import os
from fnmatch import fnmatch
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline

def rgb2gray(rgb):
    """
    converts rgb array to grey scale variant
    accordingly to fomula taken from wiki
    (this function is missing in python)
    """  
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def read_bbox_from_txt(file_name):
    """
      returns 2x2 matrix coordinates of 
      left upper and right lower corners
      of rectangle that contains face stored
      in columns of matrix
    """
    f = open(file_name)
    str = f.read().replace(',', ' ')    
    f.close()
    ret = np.array(map(int,str.split()) ,dtype=np.int32)  
    ret = ret.reshape((2,2), order='F')  
    return ret



DIR = '../../../data/Images/'
JPGS = [f for f in os.listdir(DIR) if fnmatch(f, '*.jpg')]

jpg_name = JPGS[1]
file_name = jpg_name[:-4]
img = Image.open(DIR + jpg_name)       
arr = rgb2gray(np.asarray(img))


