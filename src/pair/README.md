This module is help to generate pairs.txt used on validation

## 1. Organize your dataset
First, you should orgnize your original image name as class_number.png like this:
```
/home/david/datasets/my_dataset/test
├── Ariel_Sharon
│   ├── Ariel_Sharon_0006.png
│   ├── Ariel_Sharon_0007.png
│   ├── Ariel_Sharon_0008.png
├── Arnold_Schwarzenegger
│   ├── Arnold_Schwarzenegger_0006.png
...
...
```
To rename your image, you can use this script:
```
python rename.py your_data_folder your_path_to_save_renamed_data
```
The path should end with '/'.

## 2. Align the dataset
Alignment of the dataset can be done using `align_dataset_mtcnn` in the `align` module.

Alignment of the LFW dataset is done something like this:<br>
```
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/lfw/raw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```

The parameter `margin` controls how much wider aligned image should be cropped compared to the bounding box given by the face detector. 32 pixels with an image size of 160 pixels corresponds to a margin of 44 pixels with an image size of 182, which is the image size that has been used for training of the model below.

## 3. Generate the pars.txt
```
python generate_path.py your_renamed_and_aligned_data_path your_path_to_save_pairlist
```
