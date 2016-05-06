"""Download the VGG face dataset from URLs given by http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz
"""
from scipy import misc
import numpy as np
from skimage import io

import os
from urllib2 import HTTPError, URLError
from httplib import HTTPException

def main():
  datasetDescriptor = '/home/david/datasets/download/vggface/vgg_face_dataset/files'
  textFileNames = os.listdir(datasetDescriptor)
  for textFileName in textFileNames:
    #textFileName = textFileNames[0]
    with open(os.path.join(datasetDescriptor, textFileName), 'rt') as f:
      lines = f.readlines()
    dirName = textFileName.split('.')[0]
    classPath = os.path.join(datasetDescriptor, dirName)
    if not os.path.exists(classPath):
      os.makedirs(classPath)
    for line in lines:
      x = line.split(' ')
      fileName = x[0]
      url = x[1]
      box = np.rint(np.array(map(float, x[2:6])))  # x1,y1,x2,y2
      imagePath = os.path.join(datasetDescriptor, dirName, fileName+'.png')
      if not os.path.exists(imagePath):
        try:
          img = io.imread(url, mode='RGB')
        except (HTTPException, HTTPError, URLError, IOError) as e:
          print('%s: "%s"' % (url, e))
        else:
          try:
            if img.ndim==2:  # If image is grayscale we need to convert it to RGB
              img = toRgb(img)
            # Crop image according to dataset descriptor
            imgCropped = img[box[1]:box[3],box[0]:box[2],:]
            # Scale to 256x256
            imgResized = misc.imresize(imgCropped, (256,256))
            # Save image as .png
            misc.imsave(imagePath, imgResized)
            print('.')
          except ValueError:
            print('%s: "%s"' % (url, e))
          
def toRgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret
  

if __name__ == '__main__':
    main()
