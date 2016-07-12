"""Performs face alignment and stores face thumbnails in the output directory."""

from scipy import misc
import tensorflow as tf
import facenet
import os
import align_dlib
import random


tf.app.flags.DEFINE_string('input_dir', '', """Directory with unaligned images""")
tf.app.flags.DEFINE_string('output_dir', '', """Directory with aligned face thumbnails.""")
tf.app.flags.DEFINE_string('dlib_face_predictor', '~/repo/openface/models/dlib/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_integer('image_size', 110, """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_integer('face_size', 96, """Size of the face thumbnail (height, width) in pixels.""")
tf.app.flags.DEFINE_boolean('use_new_alignment', False,
                            """Indicates if the improved alignment transformation should be used.""")
tf.app.flags.DEFINE_string('prealigned_dir', '', """Replace image with a pre-aligned version when face detection fails.""")
tf.app.flags.DEFINE_float('prealigned_scale', 0.87, """The amount of scaling to apply to prealigned images before taking the center crop.""")

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    align = align_dlib.AlignDlib(os.path.expanduser(FLAGS.dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(argv))
    dataset = facenet.get_dataset(FLAGS.input_dir)
    random.shuffle(dataset)
    # Scale the image such that the face fills the frame when cropped to crop_size
    scale = float(FLAGS.face_size) / FLAGS.image_size
    nrof_images_total = 0
    nrof_prealigned_images = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    if FLAGS.use_new_alignment:
                        aligned = align.align_new(FLAGS.image_size, img, landmarkIndices=landmarkIndices, 
                                              skipMulti=False, scale=scale)
                    else:
                        aligned = align.align(FLAGS.image_size, img, landmarkIndices=landmarkIndices, 
                                              skipMulti=False, scale=scale)
                    if aligned is not None:
                        print(image_path)
                        nrof_successfully_aligned += 1
                        misc.imsave(output_filename, aligned)
                    elif FLAGS.prealigned_path:
                        # Face detection failed. Use center crop from pre-aligned dataset
                        class_name = os.path.split(output_class_dir)[1]
                        image_path_without_ext = os.path.join(os.path.expanduser(FLAGS.prealigned_path), 
                                                              class_name, filename)
                        # Find the extension of the image
                        exts = ('jpg', 'png')
                        for ext in exts:
                            temp_path = image_path_without_ext + '.' + ext
                            image_path = ''
                            if os.path.exists(temp_path):
                                image_path = temp_path
                                break
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            scaled = misc.imresize(img, FLAGS.prealigned_scale, interp='bilinear')
                            sz1 = scaled.shape[1]/2
                            sz2 = FLAGS.image_size/2
                            cropped = scaled[(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
                            print(image_path)
                            nrof_prealigned_images += 1
                            misc.imsave(output_filename, cropped)
                    else:
                        print('Unable to align "%s"' % image_path)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    print('Number of pre-aligned images: %d' % nrof_prealigned_images)
            

if __name__ == '__main__':
    tf.app.run()
