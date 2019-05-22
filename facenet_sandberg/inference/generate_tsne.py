from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from sklearn import preprocessing
from sklearn.manifold import TSNE

from facenet_sandberg.common_types import Face, Image
from facenet_sandberg.utils import (get_dataset, get_image_from_path_bgr,
                                    get_image_from_path_rgb)
from tensorboardX import SummaryWriter

from .identifier import Identifier


def _normalize(image: Image):
    return (image.astype(float) - 128) / 128


def _resize(img_array: List[Image]):
    resized = []
    for image in img_array:
        img = PIL.Image.fromarray(image)
        img.thumbnail((64, 64), PIL.Image.ANTIALIAS)
        resized.append(_normalize(np.array(img)))
    resized = np.array(resized)
    return resized


def _get_data(img_dir: str, is_insightface: bool,
              is_flat: bool) -> Tuple[np.ndarray, List[str]]:
    people = get_dataset(img_dir, is_flat=is_flat)

    # Get Images
    people_paths = [person.image_paths for person in people]
    all_image_paths = [path for paths in people_paths for path in paths]
    if is_insightface:
        images = map(get_image_from_path_bgr, all_image_paths)
    else:
        images = map(get_image_from_path_rgb, all_image_paths)

    # Get Labels
    names = [person.name for person in people]
    labels = []
    for index, person in enumerate(names):
        labels += [person] * len(people_paths[index])
    return np.array(list(images)), labels


def _vectorize_images(
        model_path: str,
        is_insightface: bool,
        prealigned: bool,
        images: List[Image],
        labels: List[str]) -> List[Face]:

    identifier = Identifier(
        model_path=model_path,
        is_insightface=is_insightface)
    all_faces = identifier.detect_encode_all(
        images, detect_multiple_faces=prealigned, urls=labels)
    faces_flat = [faces[0] for faces in all_faces if faces]
    return faces_flat


def write_to_tf_logdir(
        features: np.ndarray,
        labels: np.ndarray,
        torch_array: torch.Tensor,
        log_dir: str) -> None:

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_embedding(features, metadata=labels, label_img=torch_array)
    writer.close()


def generate_tsne_data(
        img_dir: str,
        model_path: str,
        is_insightface: bool,
        prealigned: bool,
        is_flat: bool):
    images, labels = _get_data(img_dir, is_insightface, is_flat)
    faces = _vectorize_images(
        model_path,
        is_insightface,
        prealigned,
        images,
        labels)
    face_images = np.array([face.image for face in faces])
    face_vectors = np.array([face.embedding for face in faces])
    face_labels = np.array([face.url for face in faces])
    resized = _resize(face_images)
    torch_array = torch.from_numpy(resized).permute(0, 3, 1, 2).float()
    return face_vectors, face_labels, torch_array


def tsne_tensorboard(
        img_dir: str,
        model_path: str,
        is_insightface: bool,
        prealigned: bool,
        is_flat: bool,
        log_dir: str):

    face_vectors, face_labels, torch_array = generate_tsne_data(
        img_dir, model_path, is_insightface, prealigned, is_flat)
    write_to_tf_logdir(face_vectors, face_labels, torch_array, log_dir)


def tsne_sklearn(
        img_dir: str,
        model_path: str,
        is_insightface: bool,
        prealigned: bool,
        is_flat: bool,
        save_plt: bool=True):

    face_vectors, face_labels, _ = generate_tsne_data(
        img_dir, model_path, is_insightface, prealigned, is_flat)
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(face_vectors)
    plt = tsne_plt(reduced, face_labels, save_plt)
    return plt


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def tsne_plt(
        reduced: np.ndarray,
        labels_str: List[str],
        save_plt: bool):
    le = preprocessing.LabelEncoder()
    labels_num = le.fit_transform(labels_str)
    plt.figure(figsize=(30, 30))
    # cmap = get_cmap(len(le.classes_))
    cmap = plt.cm.get_cmap("hsv", len(le.classes_) + 1)
    for i, label in zip(range(len(le.classes_)), le.classes_):
        plt.scatter(reduced[labels_num == i, 0],
                    reduced[labels_num == i, 1], c=cmap(i), cmap=cmap, label=label)
    plt.legend()
    if save_plt:
        plt.savefig("tsne.jpg")
    return plt


def _cli() -> None:
    args = _parse_arguments()
    if args.save_plt:
        tsne_sklearn(
            args.img_dir,
            args.model_path,
            args.is_insightface,
            args.prealigned,
            args.is_flat,
            True)
    else:
        tsne_tensorboard(
            args.img_dir,
            args.model_path,
            args.is_insightface,
            args.prealigned,
            args.is_flat,
            args.log_dir)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str,
                        required=True,
                        help='Path to the image directory.')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='path to facial recognition model')
    parser.add_argument('--is_insightface',
                        action='store_true',
                        help='Set this flag if the model is insightface')
    parser.add_argument(
        '--prealigned_flag',
        action='store_true',
        help='Specify if the images have already been aligned.')
    parser.add_argument(
        '--is_flat',
        action='store_true',
        help='Set this flag if the image directory is flat with one photo per person')
    parser.add_argument(
        '--save_plt',
        action='store_true',
        help='Set this flag if you want to save a matplot lib plot instead of tensorboard')
    parser.add_argument('--log_dir',
                        type=str,
                        required=True,
                        help='path to output the tensorflow logs')
    return parser.parse_args()


if __name__ == '__main__':
    _cli()
