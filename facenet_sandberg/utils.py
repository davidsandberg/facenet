import math
import os
import pathlib
from glob import iglob
from multiprocessing import Pool
from os.path import exists, join
from typing import List, Optional, Tuple, cast
from urllib.request import urlopen

import cv2
import numpy as np
import PIL
from sklearn.metrics.pairwise import paired_distances

from facenet_sandberg.common_types import (AlignResult, DistanceMetric,
                                           Embedding, Image, ImageExtensions,
                                           ImageGenerator, Label, Landmarks,
                                           Match, Mismatch, Pair, PersonClass)


def normalize_image(image: Image) -> Image:
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    y = np.multiply(np.subtract(image, mean), 1 / std_adj)
    return y


def fixed_standardize(image: Image) -> Image:
    image = image - 127.5
    image = image / 128.0
    return image


def fix_image(image: Image) -> Image:
    if image.ndim < 2:
        image = image[:, :, np.newaxis]
    if image.ndim == 2:
        image = add_color(image)
    image = image[:, :, 0:3]
    return image


def resize(image: Image, height: int, width: int) -> Image:
    img = PIL.Image.fromarray(image)
    img.thumbnail((height, width), PIL.Image.ANTIALIAS)
    resized = np.array(img)
    return resized


def add_color(image: Image) -> Image:
    w, h = image.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = image
    return ret


def fix_mtcnn_bb(max_y: int, max_x: int, bounding_box: List[int]) -> List[int]:
    """ mtcnn results can be out of image so fix results
    """
    x1, y1, dx, dy = bounding_box[:4]
    x2 = x1 + dx
    y2 = y1 + dy
    x1 = max(min(x1, max_x), 0)
    x2 = max(min(x2, max_x), 0)
    y1 = max(min(y1, max_y), 0)
    y2 = max(min(y2, max_y), 0)
    return [x1, y1, x2, y2]


def fix_faceboxes_bb(
        max_y: int,
        max_x: int,
        bounding_box: List[int]) -> List[int]:
    """ faceboxes order is different
    """
    y1, x1, y2, x2 = bounding_box[:4]
    x1 = max(min(x1, max_x), 0)
    x2 = max(min(x2, max_x), 0)
    y1 = max(min(y1, max_y), 0)
    y2 = max(min(y2, max_y), 0)
    return [x1, y1, x2, y2]


def embedding_distance(embedding_1: Embedding,
                       embedding_2: Embedding,
                       distance_metric: DistanceMetric) -> float:
    """Compares the distance between two embeddings
    """
    distance = embedding_distance_bulk(embedding_1.reshape(
        1, -1), embedding_2.reshape(1, -1), distance_metric=distance_metric)[0]
    return distance


def embedding_distance_bulk(
        embeddings1: Embedding,
        embeddings2: Embedding,
        distance_metric: DistanceMetric) -> np.ndarray:
    """Compares the distance between two arrays of embeddings
    """
    if distance_metric == DistanceMetric.EUCLIDEAN_SQUARED:
        return np.square(
            paired_distances(
                embeddings1,
                embeddings2,
                metric='euclidean'))
    elif distance_metric == DistanceMetric.ANGULAR_DISTANCE:
        # Angular Distance: https://en.wikipedia.org/wiki/Cosine_similarity
        similarity = 1 - paired_distances(
            embeddings1,
            embeddings2,
            metric='cosine')
        return np.arccos(similarity) / math.pi


def download_image(url: str, is_rgb: bool = True) -> Optional[Image]:
    try:
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # BGR color space
        image = cv2.imdecode(arr, -1)
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except BaseException:
        print('Couldn\'t read: {}'.format(url))
        return None


def get_image_from_path_rgb(image_path: str) -> Optional[Image]:
    # BGR color space
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return fix_image(image)
    except BaseException:
        print('Couldn\'t read: {}'.format(image_path))
        return None


def get_image_from_path_bgr(image_path: str) -> Optional[Image]:
    # BGR color space
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    except BaseException:
        print('Couldn\'t read: {}'.format(image_path))
        return None
    return fix_image(image)


def get_images_from_dir(
        directory: str,
        recursive: bool,
        is_rgb: bool = True) -> ImageGenerator:
    if recursive:
        image_paths = iglob(os.path.join(
            directory, '**', '*.*'), recursive=recursive)
    else:
        image_paths = iglob(os.path.join(directory, '*.*'))
    for image_path in image_paths:
        # BGR color space
        image = cv2.imread(image_path)
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image


def get_dataset(path: str, is_flat: bool = False) -> List[PersonClass]:
    """Gets a dataset from a directory. If is_flat then it assumes that
       there is only one image per person in a flat directory.
    """

    dataset = cast(List[PersonClass], [])
    path_exp = os.path.expanduser(path)
    if is_flat:
        people = [os.path.basename(path) for path in iglob(path_exp)]
        image_paths = get_image_paths(path_exp)
        dataset = [PersonClass(name, [image_path])
                   for name, image_path in zip(people, image_paths)]
        return dataset

    people = sorted([path for path in os.listdir(path_exp)
                     if os.path.isdir(os.path.join(path_exp, path))])
    num_people = len(people)
    for i in range(num_people):
        person_name = people[i]
        facedir = os.path.join(path_exp, person_name)
        image_paths = get_image_paths(facedir)
        dataset.append(PersonClass(person_name, image_paths))

    return dataset


def get_image_paths(facedir: str) -> List[str]:
    image_paths = cast(List[str], [])
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img)
                       for img in images if is_image(img)]
    return image_paths


def is_image(image_path: str) -> bool:
    suffix = pathlib.Path(image_path).suffix
    return suffix == '.jpg' or suffix == '.png' or suffix == '.jpeg'


def find_image_with_type(image_base_path: str, image_dir: str):
    for image_ext in ImageExtensions:
        image_path = '{}.{}'.format(image_base_path, image_ext.value)
        possible_path = join(image_dir, image_path)
        if exists(possible_path):
            return possible_path
    err = 'No Image found with name {} in directory {}'.format(
        image_base_path,
        image_dir)
    raise FileNotFoundError(err)


def get_pair_image_path(person_name: str, image_number: int, image_dir: str):
    """This is a utility function for parsing a pairs.txt file in LFW format
    """
    # e.g. person_name: Noam_Chomsky, image_number: 2 -> Noam_Chomsky_0002
    image_number_name = '{}_{}'.format(person_name, '%04d' % int(image_number))
    # e.g. Noam_Chomsky_0002 -> Noam_Chomsky/Noam_Chomsky_0002
    # This is the relative path to the image assuming LFW directory format
    relative_path = join(person_name, image_number_name)
    # e.g. Noam_Chomsky/Noam_Chomsky_0002 ->
    # {path_to_image_dir}/Noam_Chomsky/Noam_Chomsky_0002.jpg
    path_with_type = find_image_with_type(relative_path, image_dir)
    return path_with_type


def read_pairs_file(pairs_filename: str) -> Tuple[List[Pair], int, int]:
    pairs = []
    with open(pairs_filename, 'r') as pair_file:
        num_sets, num_matches_mismatches = [int(i)
                                            for i in next(pair_file).split()]
        for line in pair_file:
            pair = cast(Pair, tuple([int(i) if i.isdigit() else i
                                     for i in line.strip().split()]))
            pairs.append(pair)
    return pairs, num_sets, num_matches_mismatches


def get_paths_and_labels(
        image_dir: str, pairs: List[Pair]) -> Tuple[List[Tuple[str, str]], List[Label]]:
    paths = []
    labels = []
    for pair in pairs:
        if len(pair) == 3:
            person, num_0, num_1 = cast(Match, pair)
            rel_image_path_0 = get_pair_image_path(person, num_0, image_dir)
            rel_image_path_1 = get_pair_image_path(person, num_1, image_dir)
            is_same_person = True
        elif len(pair) == 4:
            person_0, num_0, person_1, num_1 = cast(Mismatch, pair)
            rel_image_path_0 = get_pair_image_path(person_0, num_0, image_dir)
            rel_image_path_1 = get_pair_image_path(person_1, num_1, image_dir)
            is_same_person = False
        else:
            raise SyntaxError(
                "Bad LFW format in pairs.txt: pair {} doesn't have length 3 or 4".format(pair))
        paths.append((rel_image_path_0, rel_image_path_1))
        labels.append(is_same_person)
    return paths, labels


def transform_to_lfw_format(image_directory: str,
                            num_processes: Optional[int]=os.cpu_count()):
    """Transforms an image dataset to lfw format image names.
       Base directory should have a folder per person with the person's name:
       -/base_folder
        -/person_1
          -image_1.jpg
          -image_2.jpg
          -image_3.jpg
        -/person_2
          -image_1.jpg
          -image_2.jpg
        ...
    """
    all_folders = os.path.join(image_directory, "*", "")
    people_folders = iglob(all_folders)
    process_pool = Pool(num_processes)
    process_pool.imap(_rename, people_folders)
    process_pool.close()
    process_pool.join()


def _rename(person_folder: str):
    """Renames all the images in a folder in lfw format
    """
    all_image_paths = iglob(os.path.join(person_folder, "*.*"))
    all_image_paths = sorted([image for image in all_image_paths if image.endswith(
        ".jpg") or image.endswith(".png") or image.endswith(".jpeg")])
    person_name = os.path.basename(os.path.normpath(person_folder))
    concat_name = '_'.join(person_name.split())
    for index, image_path in enumerate(all_image_paths):
        image_name = concat_name + '_' + '%04d' % (index + 1)
        file_ext = pathlib.Path(image_path).suffix
        new_image_path = os.path.join(person_folder, image_name + file_ext)
        os.rename(image_path, new_image_path)
    os.rename(person_folder, person_folder.replace(person_name, concat_name))


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode == 'SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes * (1 - split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode == 'SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))
            if split == nrof_images_in_class:
                split = nrof_images_in_class - 1
            if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
                train_set.append(PersonClass(cls.name, paths[:split]))
                test_set.append(PersonClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def crop(image: Image, bounding_box: List[int], margin: float) -> Image:
    """
    img = image from misc.imread, which should be in (H, W, C) format
    bounding_box = pixel coordinates of bounding box: (x0, y0, x1, y1)
    margin = float from 0 to 1 for the amount of margin to add, relative to the
        bounding box dimensions (half margin added to each side)
    """

    if margin < 0:
        raise ValueError("the margin must be a value between 0 and 1")
    if margin > 1:
        raise ValueError(
            "the margin must be a value between 0 and 1 - this is a change from the existing API")

    img_height = image.shape[0]
    img_width = image.shape[1]
    x_0, y_0, x_1, y_1 = bounding_box[:4]
    margin_height = (y_1 - y_0) * margin / 2
    margin_width = (x_1 - x_0) * margin / 2
    x_0 = int(np.maximum(x_0 - margin_width, 0))
    y_0 = int(np.maximum(y_0 - margin_height, 0))
    x_1 = int(np.minimum(x_1 + margin_width, img_width))
    y_1 = int(np.minimum(y_1 + margin_height, img_height))
    return image[y_0:y_1, x_0:x_1, :], (x_0, y_0, x_1, y_1)


def get_transform_matrix(left_eye: Tuple[int,
                                         int],
                         right_eye: Tuple[int,
                                          int],
                         desired_left_eye: Tuple[float,
                                                 float]=(0.35,
                                                         0.35),
                         desired_face_height: int=112,
                         desired_face_width: int=112,
                         margin: float=0.0):
    # compute the angle between the eye centers
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # compute the desired right eye x-coordinate
    desiredRightEyeX = 1.0 - desired_left_eye[0]

    # determine the scale of the new resulting image by taking
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desired_left_eye[0])
    desiredDist *= desired_face_width
    scale = (desiredDist / dist)

    # median point between the two eyes in the input image
    x_center = (left_eye[0] + right_eye[0]) // 2
    y_center = (left_eye[1] + right_eye[1]) // 2
    eye_center = (x_center, y_center)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # update the translation component of the matrix
    tX = (desired_face_width * (margin + 1)) * 0.5
    tY = (desired_face_height * (margin + 1)) * desired_left_eye[1]
    x_shift = (tX - eye_center[0])
    y_shift = (tY - eye_center[1])
    M[0, 2] += x_shift
    M[1, 2] += y_shift
    return M


def preprocess(
        image: Image,
        desired_height: int,
        desired_width: int,
        margin: float,
        bbox: List[int]=None,
        landmark: Landmarks=None,
        use_affine: bool=False):
    image_height, image_width = image.shape[:2]
    margin_height = int(desired_height + desired_height * margin)
    margin_width = int(desired_width + desired_width * margin)
    M = None
    if landmark is not None and use_affine:
        M = get_transform_matrix(landmark['left_eye'],
                                 landmark['right_eye'],
                                 (0.35, 0.35),
                                 desired_height,
                                 desired_width,
                                 margin)

    if bbox is None:
        # use center crop
        bbox = [0, 0, 0, 0]
        bbox[0] = int(image_height * 0.0625)
        bbox[1] = int(image_width * 0.0625)
        bbox[2] = image.shape[1] - bbox[0]
        bbox[3] = image.shape[0] - bbox[1]
    if M is None:
        cropped = crop(image, bbox, margin)[0]
        return cropped
    else:
        # do align using landmark
        warped = cv2.warpAffine(
            image, M, (margin_height, margin_width), flags=cv2.INTER_CUBIC)
        return warped


def get_center_box(img_size: np.ndarray, results: List[AlignResult]):
    # x1, y1, x2, y2
    all_bbs = np.asarray([result.bounding_box for result in results])
    all_landmarks = [result.landmarks for result in results]
    bounding_box_size = (all_bbs[:, 2] - all_bbs[:, 0]) * \
        (all_bbs[:, 3] - all_bbs[:, 1])
    img_center = img_size / 2
    offsets = np.vstack([(all_bbs[:, 0] + all_bbs[:, 2]) / 2 - img_center[1],
                         (all_bbs[:, 1] + all_bbs[:, 3]) / 2 - img_center[0]])
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    index = np.argmax(
        bounding_box_size -
        offset_dist_squared *
        2.0)  # some extra weight on the centering
    out_bb = all_bbs[index, :]
    out_landmark = all_landmarks[index] if index < len(all_landmarks) else None
    align_result = AlignResult(bounding_box=out_bb, landmarks=out_landmark)
    return [align_result]
