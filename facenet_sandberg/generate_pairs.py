# Implementation of pairs.txt from lfw dataset
# Section f: http://vis-www.cs.umass.edu/lfw/lfw.pdf
# More succint, less explicit: http://vis-www.cs.umass.edu/lfw/README.txt

import glob
import io
import os
import random
from argparse import ArgumentParser, Namespace
from multiprocessing import Lock, Manager, Pool, Queue, Value
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import List, Optional, Set, Tuple, cast

import numpy as np

Mismatch = Tuple[str, int, str, int]
Match = Tuple[str, int, int]
CommandLineArgs = Namespace


def write_pairs(fname: str,
                match_folds: List[List[Match]],
                mismatch_folds: List[List[Mismatch]],
                num_folds: int,
                num_matches_mismatches: int) -> None:
    metadata = '{}\t{}\n'.format(num_folds, num_matches_mismatches)
    with io.open(fname,
                 'w',
                 io.DEFAULT_BUFFER_SIZE,
                 encoding='utf-8') as fpairs:
        fpairs.write(metadata)
        for match_fold, mismatch_fold in zip(match_folds, mismatch_folds):
            for match in match_fold:
                line = '{}\t{}\t{}\n'.format(match[0], match[1], match[2])
                fpairs.write(line)
            for mismatch in mismatch_fold:
                line = '{}\t{}\t{}\t{}\n'.format(
                    mismatch[0], mismatch[1], mismatch[2], mismatch[3])
                fpairs.write(line)
        fpairs.flush()


def _split_people_into_folds(image_dir: str,
                             num_folds: int) -> List[List[str]]:
    names = [d for d in os.listdir(image_dir)
             if os.path.isdir(os.path.join(image_dir, d))]
    random.shuffle(names)
    return [list(arr) for arr in np.array_split(names, num_folds)]


def _make_matches(image_dir: str,
                  people: List[str],
                  total_matches: int) -> List[Match]:
    matches = cast(Set[Match], set())
    curr_matches = 0
    while curr_matches < total_matches:
        person = random.choice(people)
        images = _clean_images(image_dir, person)
        if len(images) > 1:
            img1, img2 = sorted(
                [images.index(random.choice(images)) + 1,
                 images.index(random.choice(images)) + 1])
            match = (person, img1, img2)
            if (img1 != img2) and (match not in matches):
                matches.add(match)
                curr_matches += 1
    return sorted(list(matches), key=lambda x: x[0].lower())


def _make_mismatches(image_dir: str,
                     people: List[str],
                     total_matches: int) -> List[Mismatch]:
    mismatches = cast(Set[Mismatch], set())
    curr_matches = 0
    while curr_matches < total_matches:
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1 != person2:
            person1_images = _clean_images(image_dir, person1)
            person2_images = _clean_images(image_dir, person2)
            if person1_images and person2_images:
                img1 = person1_images.index(random.choice(person1_images)) + 1
                img2 = person2_images.index(random.choice(person2_images)) + 1
                if person1.lower() > person2.lower():
                    person1, img1, person2, img2 = person2, img2, person1, img1
                mismatch = (person1, img1, person2, img2)
                if mismatch not in mismatches:
                    mismatches.add(mismatch)
                    curr_matches += 1
    return sorted(list(mismatches), key=lambda x: x[0].lower())


def _clean_images(base: str, folder: str):
    images = os.listdir(os.path.join(base, folder))
    images = [image for image in images if image.endswith(
        ".jpg") or image.endswith(".png")]
    return images


def _transform_to_lfw_format(image_directory: str,
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
    people_folders = glob.iglob(all_folders)
    process_pool = Pool(num_processes)
    process_pool.imap(_rename, people_folders)
    process_pool.close()
    process_pool.join()


def _rename(person_folder: str):
    """Renames all the images in a folder in lfw format
    """
    all_image_paths = glob.glob(os.path.join(person_folder, "*.*"))
    all_image_paths = [image for image in all_image_paths if image.endswith(
        ".jpg") or image.endswith(".png")]
    person_name = os.path.basename(os.path.normpath(person_folder))
    concat_name = '_'.join(person_name.split())
    for index, image_path in enumerate(all_image_paths):
        image_name = concat_name + '_' + '%04d' % (index + 1)
        file_ext = Path(image_path).suffix
        new_image_path = os.path.join(person_folder, image_name + file_ext)
        os.rename(image_path, new_image_path)
    os.rename(person_folder, person_folder.replace(person_name, concat_name))


def _main(args: CommandLineArgs) -> None:
    _transform_to_lfw_format(args.image_dir)
    people_folds = _split_people_into_folds(args.image_dir, args.num_folds)
    matches = []
    mismatches = []
    for fold in people_folds:
        matches.append(_make_matches(args.image_dir,
                                     fold,
                                     args.num_matches_mismatches))
        mismatches.append(_make_mismatches(args.image_dir,
                                           fold,
                                           args.num_matches_mismatches))
    write_pairs(args.pairs_file_name,
                matches,
                mismatches,
                args.num_folds,
                args.num_matches_mismatches)


def _cli() -> None:
    args = _parse_arguments()
    _main(args)


def _parse_arguments() -> CommandLineArgs:
    parser = ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        help='Path to the image directory.')
    parser.add_argument('--pairs_file_name',
                        type=str,
                        required=True,
                        help='Filename of pairs.txt')
    parser.add_argument('--num_folds',
                        type=int,
                        required=True,
                        help='Number of folds for k-fold cross validation.')
    parser.add_argument('--num_matches_mismatches',
                        type=int,
                        required=True,
                        help='Number of matches/mismatches per fold.')
    return parser.parse_args()


if __name__ == '__main__':
    _cli()
