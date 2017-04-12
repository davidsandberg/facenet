import shutil
import argparse
import os
import sys

def main(args):
    src_path_exp = os.path.expanduser(args.src_path)
    dst_path_exp = os.path.expanduser(args.dst_path)
    if not os.path.exists(dst_path_exp):
        os.makedirs(dst_path_exp)
    files = os.listdir(src_path_exp)
    for f in files:
        file_name = '.'.join(f.split('.')[0:-1])
        x = file_name.split('_')
        dir_name = '_'.join(x[1:-1])
        class_dst_path = os.path.join(dst_path_exp, dir_name)
        if not os.path.exists(class_dst_path):
            os.makedirs(class_dst_path)
        src_file_path = os.path.join(src_path_exp, f)
        dst_file = os.path.join(class_dst_path, f)
        print('%s -> %s' % (src_file_path, dst_file))
        shutil.copyfile(src_file_path, dst_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('src_path', type=str, help='Path to the source directory.')
    parser.add_argument('dst_path', type=str, help='Path to the destination directory.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
