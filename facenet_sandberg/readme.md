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


