import os
import glob
import numpy as np
from os import path

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
        
    def __str__( self ):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
    
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path):
    classes = os.listdir(path)
    classes.sort()
    nrofClasses = len(classes)
    dataset = [None] * nrofClasses
    for i in range(nrofClasses):
        class_name = classes[i]
        facedir = os.path.join(path, class_name)
        images = os.listdir(facedir)
        image_paths = map(lambda x: os.path.join(facedir,x), images)
        dataset[i] = ImageClass(class_name, image_paths)
        #print dataset[i]

    return dataset

def split_dataset(dataset, split_ratio):
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    split = int(round(nrof_classes*split_ratio))
    train_set = [dataset[i] for i in class_indices[0:split]]  # Range does not include the last index
    test_set = [dataset[i] for i in class_indices[split:nrof_classes]]
    return train_set, test_set


def sample_people(dataset, peoplePerBatch, imagesPerPerson):
    
    # Sample peoplePerBatch classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    classes = class_indices[0:peoplePerBatch]
    sampled_dataset = [dataset[i] for i in classes]
   
    # Find the number of images for each class
    num_per_class = [min(len(dataset[i]), imagesPerPerson) for i in classes]
    print num_per_class
    print len(num_per_class)

    # Sample maximum imagesPerPerson images from each class 
    #  and create a concatenated list of image paths
    image_paths = []
    for i in range(0,len(sampled_dataset)):
        nrof_images_in_class = len(sampled_dataset[i])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        idx = image_indices[0:num_per_class[i]]
        image_paths_for_class = [sampled_dataset[i].image_paths[j] for j in idx]
        image_paths.append(image_paths_for_class)

    return image_paths, num_per_class
    

path = '/home/david/datasets/fs_aligned/'
dataset = get_dataset(path)
train_set, test_set = split_dataset(dataset, 0.9)
peoplePerBatch = 45
imagesPerPerson = 40
sample = sample_people(train_set, peoplePerBatch, imagesPerPerson)

#print 'Training set size:' + str(len(train_set))
#print 'Test set size:' + str(len(test_set))
