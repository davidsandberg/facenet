"""Performs face alignment and stores face thumbnails in the output directory."""

from scipy import misc
import tensorflow as tf
import facenet
import os
import align_dlib

tf.app.flags.DEFINE_string('input_dir', '', """Directory with unaligned images""")
tf.app.flags.DEFINE_string('output_dir', '', """Directory with aligned face thumbnails.""")
tf.app.flags.DEFINE_string('dlib_face_predictor', '~/repo/openface/models/dlib/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_integer('image_size', 256, """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_integer('face_size', 224, """Size of the face thumbnail (height, width) in pixels.""")

FLAGS = tf.app.flags.FLAGS

def main():
    align = align_dlib.AlignDlib(os.path.expanduser(FLAGS.dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
    dataset = facenet.get_dataset(FLAGS.input_dir)
    # Scale the image such that the face fills the frame when cropped to crop_size
    scale = float(FLAGS.face_size) / FLAGS.image_size
    for cls in dataset:
        output_class_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            if not os.path.exists(output_filename):
                print(image_path)
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim == 2:
                        img = facenet.toRgb(img)
                    aligned = align.align(FLAGS.image_size, img, landmarkIndices=landmarkIndices, 
                                          skipMulti=True, scale=scale)
                    if aligned is not None:
                        misc.imsave(output_filename, aligned)

            
if __name__ == '__main__':
    main()
