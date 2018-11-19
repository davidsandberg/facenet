"""Rename the image based on the folder name"""
import os
import shutil
import sys
import argparse

def main(args):
    original_path = args.data_dir
    saved_path = args.save_dir
    make_path(saved_path)
    all_folders = traversalDir_FirstDir(original_path)
    for folder in all_folders:
        files = os.listdir(original_path + folder)
        i = 1
        for file in files:
            suffix = '.png'
            name = folder + '_' + str(i).zfill(4) + suffix
            i = i + 1
            sub_saved_path = saved_path + folder
            make_path(sub_saved_path)
            shutil.copyfile(original_path + folder + '/' + file, sub_saved_path + '/'  + name)

# To get all sub folders in one folder
def traversalDir_FirstDir(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
        return list

# To judge whether a folder is existed.
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with aligned images.')
    parser.add_argument('save_dir', type=str, help='Directory to save renamed images.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))