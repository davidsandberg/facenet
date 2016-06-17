#!/usr/bin/env python2

from scipy import misc
import numpy as np
from skimage import io

import os
import cv2
import pdb
import random
import socket
import binascii
from urllib2 import HTTPError, URLError
from httplib import HTTPException

import argparse



def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--datasetDir', type=str,
        help='The folder that store the files list of the vgg_face_dataset, Download the VGG face dataset from URLs given by http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz'),
  parser.add_argument('--tmpDir', type=str,
		default = '/tmp/maiguang',
        help='The folder to store the temp. images')
  args = parser.parse_args()
  
  socket.setdefaulttimeout(30)
  datasetDescriptor = args.datasetDir
  textFileNames = os.listdir(datasetDescriptor)
  random.shuffle(textFileNames)
  for textFileName in textFileNames:
    if textFileName.endswith('.txt'):
      with open(os.path.join(datasetDescriptor, textFileName), 'rt') as f:
        lines = f.readlines()
      dirName = textFileName[0:-4]
      datafolder = os.path.split(datasetDescriptor)[0]
      classPath = os.path.join(datafolder,'img', dirName)
      if not os.path.exists(classPath):
        os.makedirs(classPath)
      else:
        existFiles = os.listdir(classPath)
        if len(existFiles) >= 1000:
          continue
      for line in lines:
        x = line.split(' ')
        fileName = x[0]
        url = x[1]
        box = np.rint(np.array(map(float, x[2:6])))  # x1,y1,x2,y2
        imagePath = os.path.join(classPath, fileName+'.png')
        errorPath = os.path.join(classPath, fileName+'.err')
        if not os.path.exists(imagePath) and not os.path.exists(errorPath):
          try:
            tmpFile = args.tmpDir+'/{}-'.format(binascii.b2a_hex(os.urandom(8)))+os.path.split(url)[1]
            os.system('wget -O '+tmpFile+" --connect-timeout 10 -q --tries=1 " + url)
            img = cv2.imread(tmpFile)
            if img is None:
                raise IOError('image not found')
          except (HTTPException, HTTPError, URLError, IOError, ValueError, IndexError, OSError) as e:
            errorMessage = '{}: {}'.format(url, e)
            if os.path.exists(tmpFile):
              os.remove(tmpFile)
            saveErrorMessageFile(errorPath, errorMessage)
          else:
            try:
              if os.path.exists(tmpFile):
                os.remove(tmpFile)
              if img.ndim == 2:
                img = toRgb(img)
              if img.ndim != 3:
                raise ValueError('Wrong number of image dimensions')
              # Crop image according to dataset descriptor
              imgCropped = img[box[1]:box[3],box[0]:box[2],:]
              # Scale to 256x256
              imgResized = misc.imresize(imgCropped, (256,256))
              # Save image as .png
              hist = np.histogram(imgResized, 255, density=True)
              if hist[0][0]>0.9 and hist[0][254]>0.9:
                raise ValueError('Image is mainly black or white')
              else:
                cv2.imwrite(imagePath, imgResized)
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
