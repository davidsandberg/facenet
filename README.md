# FaceNet implementation in Tensorflow
This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf) as well as the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

The accuracy measured on the LFW test set is ~0.98 when trained on a combination of FaceScrub and CASIA-WebFace.

## Inspiration
The code is heavly inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [FaceScrub](http://vintage.winklerbros.net/facescrub.html) dataset and the [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset have been used for training. This training set consists of total of 536 685 images over 11105 identities.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, siluettes, etc). This makes the training set to "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. Experimental code that has been used to align the training datasets can be found [here](https://github.com/davidsandberg/facenet/blob/master/tmp/align_dataset.m). However, work is ongoing to reimplement MTCNN face alignment in Python/Tensorflow. Currently some work still remains on this but the implementation can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Running training
Currently, the best results are achieved by training the model as a classifier with the addition of [Center loss](http://ydwen.github.io/papers/WenECCV16.pdf). Details on how to train a model as a classifier can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1).

## Pre-trained model
### Inception-ResNet-v1 model
Currently, the best performing model is an Inception-Resnet-v1 model trained on a combination of FaceScrub and CASIA-Webface aligned with [MTCNN](https://github.com/davidsandberg/facenet/blob/master/tmp/align_dataset.m). This alignment step requires Matlab and Caffe installed which requires some extra work. This will be easier when the [Python/Tensorflow implementation](https://github.com/davidsandberg/facenet/tree/master/src/align) is fully functional.

## Performance
The accuracy on LFW for the model [20161116-234200](https://drive.google.com/file/d/0B5MzpY9kBtDVSTgxX25ZQzNTMGc/view?usp=sharing) is 0.980+-0.006. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw).
