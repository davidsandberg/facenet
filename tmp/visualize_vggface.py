import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tmp.vggface16

def main():
  
    sess = tf.Session()
  
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    image_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-image_mean, 0)
     
    # Build the inference graph
    nodes = tmp.vggface16.load('data/vgg_face.mat', t_preprocessed)
        
    img_noise = np.random.uniform(size=(224,224,3)) + 117.0

    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    layer = 'conv5_3'
    channel = 140 # picking some feature channel to visualize
    img = render_naive(sess, t_input, nodes[layer][:,:,:,channel], img_noise)
    showarray(img)

def showarray(a):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.show()
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def render_naive(sess, t_input, t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for _ in range(iter_n):
        g, _ = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
    return visstd(img)

  
if __name__ == '__main__':
    main()
