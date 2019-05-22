# Usage

## Image directory structure

This repo assumes images are in [LFW format](http://vis-www.cs.umass.edu/lfw/README.txt):

```bash
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

```bash
python lfw.py --image_directory PATH_TO_YOUR_BASE_IMAGE_DIRECTORY
```

## Alignment

Alignment is done with a combination of Faceboxes and MTCNN. While Faceboxes is more accurate and works with more images than MTCNN, it does not return [facial landmarks](https://raw.githubusercontent.com/ipazc/mtcnn/master/result.jpg). Whichever algorithm returns more results is used.

Use the [align_dataset.py](https://github.com/armanrahman22/facenet/blob/master/facenet_sandberg/align_dataset.py) script to align an entire image directory:

```bash
python align_dataset.py --config_file PATH_TO_YOUR_CONFIG_FILE
```

* You can either alter the default config files provided in facenet_config.json and insightface_config.json or create your own following the same format

## Generate Pairs.txt

A pairs.txt file is used in training and testing. It follows this [format](http://vis-www.cs.umass.edu/lfw/README.txt). In order to generate your own pairs.txt run:

```bash
python align_dataset.py --image_dir PATH_TO_YOUR_BASE_IMAGE_DIRECTORY \
                        --pairs_file_name OUTPUT_NAME_OF_PAIRS_FILE \
                        --num_folds NUMBER_OF_FOLDS_FOR_CROSS_VALIDATION \
                        --num_matches_mismatches NUMBER_OF_MATCHES_AND_MISMATCHES
```