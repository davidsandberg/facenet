import shutil
import argparse
import os
import sys

def main(args):
  
    identity_map = {}
    with open(os.path.expanduser(args.map_file_name), "r") as f:
        for line in f:
            fields = line.split(' ')
            dir_name = fields[0]
            class_name = fields[1].replace('\n', '').replace('\r', '')
            if class_name not in identity_map.values():
                identity_map[dir_name] = class_name
            else:
                print('Duplicate class names: %s' % class_name)
            
    dataset_path_exp = os.path.expanduser(args.dataset_path)
    dirs = os.listdir(dataset_path_exp)
    for f in dirs:
        old_path = os.path.join(dataset_path_exp, f)
        if f in identity_map:
            new_path = os.path.join(dataset_path_exp, identity_map[f])
            if os.path.isdir(old_path):
                print('Renaming %s to %s' % (old_path, new_path))
                shutil.move(old_path, new_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('map_file_name', type=str, help='Name of the text file that contains the directory to class name mappings.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
