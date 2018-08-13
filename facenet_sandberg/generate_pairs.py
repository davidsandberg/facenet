# Implementation of pairs.txt from lfw dataset
# Section f: http://vis-www.cs.umass.edu/lfw/lfw.pdf
# More succint, less explicit: http://vis-www.cs.umass.edu/lfw/README.txt

import os
import random
import sys
import argparse
import numpy as np
from typing import List, Tuple, cast


Mismatch = Tuple[str, int, str, int]
Match = Tuple[str, int, int]


def write_pairs(fname: str,
                match_folds: List[List[Match]],
                mismatch_folds: List[List[Mismatch]],
                k_num_sets: int,
                total_matches_mismatches: int) -> None:
    file_contents = f'{k_num_sets}\t{total_matches_mismatches}\n'
    for match_fold, mismatch_fold in zip(match_folds, mismatch_folds):
        for match in match_fold:
            file_contents += f'{match[0]}\t{match[1]}\t{match[2]}\n'
        for mismatch in mismatch_fold:
            file_contents += f'{mismatch[0]}\t{mismatch[1]}\t\
{mismatch[2]}\t{mismatch[3]}\n'
    with open(fname, 'w') as fpairs:
        fpairs.write(file_contents)


def _split_people_into_folds(image_dir: str,
                             k_num_sets: int) -> List[List[str]]:
    names = [d for d in os.listdir(image_dir)
             if os.path.isdir(os.path.join(image_dir, d))]
    random.shuffle(names)
    return [list(arr) for arr in np.array_split(names, k_num_sets)]


def _make_matches(image_dir: str,
                  people: List[str],
                  total_matches: int) -> List[Match]:
    matches = cast(List[Match], [])
    curr_matches = 0
    while curr_matches < total_matches:
        person = random.choice(people)
        images = os.listdir(os.path.join(image_dir, person))
        if len(images) > 1:
            img1, img2 = sorted(
                [
                    int(''.join([i for i in random.choice(images)
                                 if i.isnumeric()]).lstrip('0')),
                    int(''.join([i for i in random.choice(images)
                                 if i.isnumeric()]).lstrip('0'))
                ]
            )
            match = (person, img1, img2)
            if (img1 != img2) and (match not in matches):
                matches.append(match)
                curr_matches += 1
    return sorted(matches, key=lambda x: x[0].lower())


def _make_mismatches(image_dir: str,
                     people: List[str],
                     total_matches: int) -> List[Mismatch]:
    mismatches = cast(List[Mismatch], [])
    curr_matches = 0
    while curr_matches < total_matches:
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1 != person2:
            person1_images = os.listdir(os.path.join(image_dir, person1))
            person2_images = os.listdir(os.path.join(image_dir, person2))
            if person1_images and person2_images:
                img1 = int(''.join([i for i in random.choice(person1_images)
                                    if i.isnumeric()]).lstrip('0'))
                img2 = int(''.join([i for i in random.choice(person2_images)
                                    if i.isnumeric()]).lstrip('0'))
                if person1.lower() > person2.lower():
                    person1, img1, person2, img2 = person2, img2, person1, img1
                mismatch = (person1, img1, person2, img2)
                if mismatch not in mismatches:
                    mismatches.append(mismatch)
                    curr_matches += 1
    return sorted(mismatches, key=lambda x: x[0].lower())


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()
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
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _parse_arguments(sys.argv[1:])
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
