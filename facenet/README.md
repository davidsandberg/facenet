# FaceNet implementation in Tensorflow
This is a TensorFlow implementation of the face recognizer described in the paper
"FaceNet: A Unified Embedding for Face Recognition and Clustering"
http://arxiv.org/abs/1503.03832

## Inspiration:
The code is heavly inspired by the OpenFace implementation at https://github.com/cmusatyalab/openface

## Training data:
The FaceScrub dataset (http://vintage.winklerbros.net/facescrub.html) and the CASIA-WebFace dataset (http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) have been used for training.

## Pre-processing:
The data has been pre-processed as described on the OpenFace web page (https://cmusatyalab.github.io/openface/models-and-accuracies/), i.e. using
./util/align-dlib.py data/lfw/raw align outerEyesAndNose data/lfw/dlib-affine-sz:96 --size 96 --fallbackLfw data/lfw/deepfunneled

## Pre-trained model:
A model that has been train on a combination of FaceScrub and CASIA-Webface is available:
[model-20160506.ckpt-500000](https://drive.google.com/file/d/0B5MzpY9kBtDVVFRyU2JCVmZXUEk/view?usp=sharing)
This model has been trained for 500 epochs (with a batch size of 90 images).
To load the model Tensorflow needs a checkpoint file in the same directory as the model file.
The checkpoint file is created when the model is stored (during training), but can also be created
with a text editor (see below). But remember to adjust the paths to point to your model file.
checkpoint:
```
model_checkpoint_path: "/home/david/models/facenet/model-20160506.ckpt-500000"
all_model_checkpoint_paths: "/home/david/models/facenet/model-20160506.ckpt-500000"
```
## Performance:
The accuracy on LFW for the model "model-20160506.ckpt-500000" is 0.919+-0.008. The test can be run using "validate_on_lfw.py".

## Under development:
This project is currently very much under development (i will try to keep the issue tracker up-to-date with what is in the pipe).
