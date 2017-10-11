# Align / Crop to face
_Give it folder(s) of photos, get new folders with images cropped to just the face_

This Python 2.x/3.x utility ([ 	align_dataset_mtcnn.py](align_dataset_mtcnn.py)) takes either a folder of images or a folder-of-folders of images (e.g. you may be using one folder per class/identity) and used the excellent [MTCNN model](https://kpzhang93.github.io/MTCNN_face_detection_alignment/) to crop down to just the face (if it is found).  Tests on a private benchmark show MTCNN detecting a lot of faces that the OpenCV face detector misses (74% of faces present versus 58%).

### Parameters:
- `input_dir`  A folder with photos that may contain faces, or a folder of folders with photos (perhaps your images are organised by class or identity?).
- `output_dir` The folder where you want us to put the cropped photos.  If you passed us a folder of folders, the same structure will be recreated.
- `--image_size` Output image size.  Defaults to 182.
- `--margin` Margin for the crop around the bounding box (height, width) in pixels. Defaults to 44.
- `--random_order` Shuffles the order of the images to enable alignment using multiple processes. Defaults to on.
- `--gpu_memory_fraction`  Upper bound on the amount of GPU memory that will be used by the process, defaults to 1.0.
- `--has_classes` Specifies your input folder is divided into sub-folders (class folders). Note this structure will be preserved in the output folder.  If not, use `--no_classes`.


    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--has_classes', dest='has_classes', action='store_true',
        help='Input folder is split into class subfolders, and these should be replicated', default=True)
    parser.add_argument('--no_classes', dest='has_classes', action='store_false',
help='Input folder is split into class subfolders, and these should be replicated', default=True)
