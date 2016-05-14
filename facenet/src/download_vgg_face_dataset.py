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
      errorPath = os.path.join(datasetDescriptor, dirName, fileName+'.err')
      if not os.path.exists(imagePath) and not os.path.exists(errorPath):
        try:
          img = io.imread(url, mode='RGB')
        except (HTTPException, HTTPError, URLError, IOError) as e:
          errorMessage = '{}: {}'.format(url, e)
          saveErrorMessageFile(errorPath, errorMessage)
        else:
          try:
            if img.ndim == 2:
              img = toRgb(img)
            if img.ndim != 3:
              raise ValueError('Wrong number of image dimensions')
            hist = np.histogram(img, 255, density=True)
            if hist[0][0]>0.9 and hist[0][254]>0.9:
              raise ValueError('Image is mainly black or white')
            else:
              # Crop image according to dataset descriptor
              imgCropped = img[box[1]:box[3],box[0]:box[2],:]
              # Scale to 256x256
              imgResized = misc.imresize(imgCropped, (256,256))
              # Save image as .png
              misc.imsave(imagePath, imgResized)
          except ValueError as e:
            errorMessage = '{}: {}'.format(url, e)
            saveErrorMessageFile(errorPath, errorMessage)
            
def saveErrorMessageFile(fileName, errorMessage):
  print(errorMessage)
  with open(fileName, "w") as textFile:
    textFile.write(errorMessage)
          
def toRgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret
  

if __name__ == '__main__':
    main()
