import os
import shutil


def main():
    dataset = '/home/david/datasets/facescrub/test/facescrub/'
    target_dir = '/home/david/datasets/facescrub/facescrub_prealigned/'
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    
    classes = os.listdir(dataset)
    
    for cls in classes:
      source_dir = os.path.join(dataset, cls, 'face')
      target_class_dir = os.path.join(target_dir, cls)
      if not os.path.exists(target_class_dir):
        os.mkdir(target_class_dir)
        if os.path.exists(source_dir):
          files = os.listdir(source_dir)
          print('Moving %s to %s' % (source_dir, target_class_dir))
          for fname in files:
            src = os.path.join(source_dir, fname)
            shutil.move(src, target_class_dir)
          os.rmdir(source_dir)
        else:
          print('Could not find %s\n' % source_dir)
          
if __name__ == '__main__':
    main()
