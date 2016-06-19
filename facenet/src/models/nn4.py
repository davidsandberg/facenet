import tensorflow as tf
import facenet

def inference(images, pool_type, use_lrn, keep_probability, phase_train=True):
  """ Define an inference network for face recognition based 
         on inception modules using batch normalization
  
  Args:
    images: The images to run inference on, dimensions batch_size x height x width x channels
    phase_train: True if batch normalization should operate in training mode
  """
  conv1 = facenet.conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train, use_batch_norm=True)
  pool1 = facenet.mpool(conv1,  3, 3, 2, 2, 'SAME')
  if use_lrn:
    lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
  else:
    lrn1 = pool1
  conv2 = facenet.conv(lrn1,  64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True)
  conv3 = facenet.conv(conv2,  64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True)
  if use_lrn:
    lrn2 = tf.nn.local_response_normalization(conv3, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
  else:
    lrn2 = conv3
  pool3 = facenet.mpool(lrn2,  3, 3, 2, 2, 'SAME')

  incept3a = facenet.inception(pool3,    192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train, use_batch_norm=True)
  incept3b = facenet.inception(incept3a, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, pool_type, 'incept3b', phase_train=phase_train, use_batch_norm=True)
  incept3c = facenet.inception(incept3b, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train, use_batch_norm=True)
  
  incept4a = facenet.inception(incept3c, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, pool_type, 'incept4a', phase_train=phase_train, use_batch_norm=True)
  incept4b = facenet.inception(incept4a, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, pool_type, 'incept4b', phase_train=phase_train, use_batch_norm=True)
  incept4c = facenet.inception(incept4b, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, pool_type, 'incept4c', phase_train=phase_train, use_batch_norm=True)
  incept4d = facenet.inception(incept4c, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, pool_type, 'incept4d', phase_train=phase_train, use_batch_norm=True)
  incept4e = facenet.inception(incept4d, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train, use_batch_norm=True)
  
  incept5a = facenet.inception(incept4e,    1024, 1, 384, 192, 384, 0, 0, 3, 128, 1, pool_type, 'incept5a', phase_train=phase_train, use_batch_norm=True)
  incept5b = facenet.inception(incept5a, 896, 1, 384, 192, 384, 0, 0, 3, 128, 1, 'MAX', 'incept5b', phase_train=phase_train, use_batch_norm=True)
  pool6 = facenet.apool(incept5b,  3, 3, 1, 1, 'VALID')

  resh1 = tf.reshape(pool6, [-1, 896])
  affn1 = facenet.affine(resh1, 896, 128)
  dropout = tf.nn.dropout(affn1, keep_probability)
  norm = tf.nn.l2_normalize(dropout, 1, 1e-10, name='embeddings')

  return norm
