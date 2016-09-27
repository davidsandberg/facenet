import facenet
import argparse
import sys
import time
import numpy as np

def main(args):

    dataset = facenet.get_dataset(args.dir)
    paths, _ = facenet.get_image_paths_and_labels(dataset)
    t = np.zeros((len(paths)))
    x = time.time()
    for i, path in enumerate(paths):
        start_time = time.time()
        with open(path, mode='rb') as f:
            _ = f.read()
        duration = time.time() - start_time
        t[i] = duration
        if i % 1000 == 0 or i==len(paths)-1:
            print('File %d/%d  Total time: %.2f  Avg: %.3f  Std: %.3f' % (i, len(paths), time.time()-x, np.mean(t[0:i])*1000, np.std(t[0:i])*1000))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, 
        help='Directory with dataset to test')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
