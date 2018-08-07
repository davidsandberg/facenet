# Implementation of pairs.txt from lfw dataset
# Section f: http://vis-www.cs.umass.edu/lfw/lfw.pdf
# More succint, less explicit: http://vis-www.cs.umass.edu/lfw/README.txt

import os
import random
import numpy as np
from typing import List, Tuple

def split_people_into_sets(image_dir: str, k_num_sets: int) -> List[List[str]]:
    names = os.listdir(image_dir)
    random.shuffle(names)
    return [list(arr) for arr in np.array_split(names, k_num_sets)]

def make_matches(image_dir:str , people: List[str], total_matches: int) -> List[Tuple[str, int, int]]:
    matches: List[Tuple[str, int, int]] = []
    curr_matches = 0
    while curr_matches < total_matches:
        person = random.choice(people)
        images = os.listdir(os.path.join(image_dir, person))
        if len(images) > 1:
            img1, img2 = sorted(
                [
                    int(''.join([i for i in random.choice(images) if i.isnumeric() and i != '0'])),
                    int(''.join([i for i in random.choice(images) if i.isnumeric() and i != '0']))
                ]
            )
            match = (person, img1, img2)
            if (img1 != img2) and (match not in matches):
                matches.append(match)
                curr_matches += 1
    return sorted(matches, key=lambda x: x[0].lower())

def make_mismatches(image_dir: str, people: List[str], total_matches: int) -> List[Tuple[str, int, str, int]]:
    mismatches: List[Tuple[str, int, str, int]] = []
    curr_matches = 0
    while curr_matches < total_matches:
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1 != person2:
            person1_images = os.listdir(os.path.join(image_dir, person1))
            person2_images = os.listdir(os.path.join(image_dir, person2))

            if person1_images and person2_images:
                img1 = int(''.join([i for i in random.choice(person1_images) if i.isnumeric() and i != '0']))
                img2 = int(''.join([i for i in random.choice(person2_images) if i.isnumeric() and i != '0']))

            if person1.lower() > person2.lower():
                person1, img1, person2, img2 = person2, img2, person1, img1
            
            mismatch = (person1, img1, person2, img2)
            if mismatch not in mismatches:
                mismatches.append(mismatch)
                curr_matches += 1
    return sorted(mismatches, key=lambda x: x[0].lower())

def write_pairs(fname: str, match_sets: List[List[Tuple[str, int, int]]], mismatch_sets: List[List[Tuple[str, int, str, int]]], k_num_sets: int, total_matches_mismatches: int) -> None:
    file_contents = f'{k_num_sets}\t{total_matches_mismatches}\n'
    for match_set, mismatch_set in zip(match_sets, mismatch_sets):
        for match in match_set:
            file_contents += f'{match[0]}\t{match[1]}\t{match[2]}\n'
        for mismatch in mismatch_set:
            file_contents += f'{mismatch[0]}\t{mismatch[1]}\t{mismatch[2]}\t{mismatch[3]}\n' 

    with open(fname, 'w') as fpairs:
        fpairs.write(file_contents)

if __name__ == '__main__':
    k_num_sets = 10
    total_matches_mismatches = 100
    image_dir = os.path.join(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    ), 
                'images')

    people_lists = split_people_into_sets(image_dir, k_num_sets)
    matches = []
    mismatches = []
    for people in people_lists:
        matches.append(make_matches(image_dir, people, total_matches_mismatches))
        mismatches.append(make_mismatches(image_dir, people, total_matches_mismatches))
    
    fname = 'new_pairs.txt'
    write_pairs(fname, matches, mismatches, k_num_sets, total_matches_mismatches)
