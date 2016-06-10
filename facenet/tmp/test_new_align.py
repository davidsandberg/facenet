import facenet
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    image_size = 96
    old_dataset = '/home/david/datasets/facescrub/fs_aligned_new_oean/'
    new_dataset = '/home/david/datasets/facescrub/facescrub_110_96/'
    eq = 0
    num = 0
    l = []
    dataset = facenet.get_dataset(old_dataset)
    for cls in dataset:
        new_class_dir = os.path.join(new_dataset, cls.name)
        for image_path in cls.image_paths:
          try:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            new_filename = os.path.join(new_class_dir, filename+'.png')
            #print(image_path)
            if os.path.exists(new_filename):
                a = facenet.load_data([image_path, new_filename], False, False, image_size, do_prewhiten=False)
                if np.array_equal(a[0], a[1]):
                  eq+=1
                num+=1
                err = np.sum(np.square(np.subtract(a[0], a[1])))
                #print(err)
                l.append(err)
                if err>2000:
                  fig = plt.figure(1)
                  p1 = fig.add_subplot(121)
                  p1.imshow(a[0])
                  p2 = fig.add_subplot(122)
                  p2.imshow(a[1])
                  print('%6.1f: %s\n' % (err, new_filename))
                  pass
            else:
                pass
                #print('File not found: %s' % new_filename)
          except:
            pass
if __name__ == '__main__':
    main()
