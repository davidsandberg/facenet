# Facial Recognition and Alignment
## What's this?

This repository contains a refactored implementation of David Sandberg's [FaceNet](https://github.com/davidsandberg/facenet) and [InsightFace](https://github.com/deepinsight/insightface) for facial recognition. It also contains an implementation of [MTCNN](https://github.com/ipazc/mtcnn) and [Faceboxes](https://github.com/TropComplique/FaceBoxes-tensorflow) for face cropping and alignment. What is in the refactor:

- Made algorithms easily and efficiently usable with [convenience classes](https://github.com/armanrahman22/facenet/tree/master/facenet_sandberg/inference). 
- Added much more efficient methods of batch processing face recognition and alignment
- Added true face alignment (with affine transformation) to align face to bottom-center of image: [code](https://github.com/armanrahman22/facenet/blob/f6cb32a193925002da41fb491c52bb85384bee55/facenet_sandberg/utils.py#L187)
- Added proportional margin to alignment as per this [issue](https://github.com/davidsandberg/facenet/issues/283)
- Ability to easily switch between [insightface](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/inference/insightface_encoder.py) and [facenet](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/inference/facenet_encoder.py) at [inference time](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/inference/identifier.py)

More information on customizing and implementing new face detection algorithms can be found [here](./algorithms/README.md).

## Installation
To use in other projects, this implementation can be pip installed as follows:
```
pip install facenet_sandberg
```

To use locally:
1. Clone repo
2. cd to base directory with setup.py 
3. run:
```
pip install -e .
```
^(installs package in [development mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode))

## Important Requirements
1. Python 3.5 
2. Tensorflow==1.7
3. Tensorlayer==1.7
The rest is specified in [requirements.txt](https://github.com/armanrahman22/facenet/blob/master/requirements.txt)

## Models
Links to pretrained models: 

- [Facenet](https://redcrossstorage.blob.core.windows.net/images/facenet_model.pb)
  - Uses RGB images of size 160x160
- [Insightface.zip](https://redcrossstorage.blob.core.windows.net/images/insightface_ckpt.zip)
  - Uses BGR images of size 112x112

## Datasets
Links to download training datasets (!big files!):

- [Emore](https://redcrossstorage.blob.core.windows.net/datasets/faces_emore.zip) 
- [MSM_refined_112x112](https://redcrossstorage.blob.core.windows.net/datasets/faces_ms1m-refine-v2_112x112.zip)
- [VGG2_112x112](https://redcrossstorage.blob.core.windows.net/datasets/faces_vgg2_112x112.zip)

## Image directory structure
This repo assumes images are in [LFW format](http://vis-www.cs.umass.edu/lfw/README.txt):
```
-/base_images_folder
  -/person_1
    -person_1_0001.jpg
    -person_1_0002.jpg
    -person_1_0003.jpg
  -/person_2
    -person_2_0001.jpg
    -person_2_0002.jpg
  ...
```

If your dataset is not like this you can use [lfw.py](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/lfw.py) to put your images into the right format like so (from facenet_sandberg/facenet_sandberg):
```
python lfw.py --image_directory PATH_TO_YOUR_BASE_IMAGE_DIRECTORY
```

## Alignment
Alignment is done with a combination of Faceboxes and MTCNN. While Faceboxes is more accurate and works with more images than MTCNN, it does not return [facial landmarks](https://raw.githubusercontent.com/ipazc/mtcnn/master/result.jpg). Whichever algorithm returns more results is used.

Use the [align_dataset.py](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/align_dataset.py) script to align an entire image directory:
```
python align_dataset.py --input_dir PATH_TO_YOUR_BASE_IMAGE_DIRECTORY \
                        --output_dir PATH_TO_OUTPUT_ALIGNED_IMAGES \
                        --facenet_model_checkpoint PATH_TO_PRETRAINED_FACENET_MODEL \
                        --image_height DESIRED_IMAGE_HEIGHT \
                        --image_width DESIRED_IMAGE_WIDTH \
                        --margin DESIRED_PROPORTIONAL_MARGIN \
                        --scale_factor DESIRED_SCALE_FACTOR \
                        --steps_threshold DESIRED_STEPS \
                        --detect_multiple_faces \
                        --use_faceboxes \
                        --use_affine \
                        --num_processes NUM_PROCESSES_TO_USE
```
* Default values for most arguments are provided [here](https://github.com/armanrahman22/facenet/blob/f6cb32a193925002da41fb491c52bb85384bee55/facenet_sandberg/align_dataset.py#L262) 

## Generate Pairs.txt
A pairs.txt file is used in training and testing. It follows this [format](http://vis-www.cs.umass.edu/lfw/README.txt). In order to generate your own pairs.txt run:
```
python align_dataset.py --image_dir PATH_TO_YOUR_BASE_IMAGE_DIRECTORY \
                        --pairs_file_name OUTPUT_NAME_OF_PAIRS_FILE \
                        --num_folds NUMBER_OF_FOLDS_FOR_CROSS_VALIDATION \
                        --num_matches_mismatches NUMBER_OF_MATCHES_AND_MISMATCHES
```

## Copyright
MIT License from original repo https://github.com/davidsandberg/facenet/blob/master/LICENSE.md


