This is a TensorFlow implementation of the face recognizer described in the paper 
"FaceNet: A Unified Embedding for Face Recognition and Clustering"
http://arxiv.org/abs/1503.03832

Inspiration: 
The code is heavly inspired by the OpenFace implementation at https://github.com/cmusatyalab/openface

Training data: 
The FaceScrub dataset (http://vintage.winklerbros.net/facescrub.html) and the CASIA-WebFace dataset (http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) have been used for training.

Pre-processing: 
The data has been pre-processed as described on the OpenFace web page (https://cmusatyalab.github.io/openface/models-and-accuracies/), i.e. using
./util/align-dlib.py data/lfw/raw align outerEyesAndNose data/lfw/dlib-affine-sz:96 --size 96 --fallbackLfw data/lfw/deepfunneled

Pre-trained model:
A pretrained model in TensorFlow checkpoint format can be downloaded from Google Drive at https://drive.google.com/file/d/0B5MzpY9kBtDVUFNFR21kZnYybkE/view?usp=sharing.
This model has been trained on the FaceScrub dataset for 35 epochs. 

A model that has been train on a combination of FaceScrub and CASIA-Webface is also available:
https://drive.google.com/file/d/0B5MzpY9kBtDVYkhISWV5dFZnNHc/view?usp=sharing
This model has been trained for 500 epochs (with a batch size of 90 images). 

Performance:
The accuracy on LFW for the model "model-20160306.ckpt-500000" is 0.916±0.010. The test can be run using "validate_on_lfw.py".

Under development: 
This project is currently very much under development (i will try to keep the issue tracker up-to-date with what is in the pipe).
