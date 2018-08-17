import os
from PIL import Image


def check_and_delete(image):
    try:
        v_image = Image.open(image)
        v_image.verify()
    except:
        os.remove(image)


base_dir = 'test_mtcnn_1'
for image_dir in os.listdir(base_dir):
    image_dir = os.path.join(base_dir, image_dir)
    for image_file in os.listdir(image_dir):
        image = os.path.join(image_dir, image_file)
        check_and_delete(image)
