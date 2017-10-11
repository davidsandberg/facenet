# Faces to Vectors
_An easy way to get embeddings for your faces_

This Python 3.x utility ([face_vectors.py](face_vectors.py)) takes a folder of images (which we assume have already been cropped to faces, perhaps using the [align_dataset_mtcnn.py](https://github.com/EdwardDixon/facenet/blob/master/src/align/align_dataset_mtcnn.py) utility) and converts them to face embeddings using the Facenet Tensorflow model (or a similar model whose embedding layer has a matching name).

### Parameters:
- `--inpath`  A folder with pictures of faces.  For best results, crop tightly with MTCNN or equivalent.
- `--outpath` The full path for the results file, which will be in JSON format (filenames will be used as keys, with face embeddings as the values).
- `--mdlpath` Where to find the Tensorflow model to use for the embedding
- `--imgsize` Purely optional, defaults to 160 pixels.

