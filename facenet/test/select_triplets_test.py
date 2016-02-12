import facenet
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('people_per_batch', 45,
                            """Number of people per batch.""")
tf.app.flags.DEFINE_integer('alpha', 0.2,
                            """Positive to negative triplet distance margin.""")


embeddings = np.zeros((1800,128))

np.random.seed(123)
for ix in range(embeddings.shape[0]):
    for jx in range(embeddings.shape[1]):
        rnd = 1.0*np.random.randint(1,2**32)/2**32
        embeddings[ix][jx] = rnd


emb_array = embeddings
image_data = np.zeros((1800,96,96,3))


num_per_class = [40 for i in range(45)]


np.random.seed(123)
apn, nrof_random_negs, nrof_triplets = facenet.select_triplets(emb_array, num_per_class, image_data)
