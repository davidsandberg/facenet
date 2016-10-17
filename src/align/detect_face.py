""" Hej
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import cv2
from math import floor

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item() #pylint: disable=no-member
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(inp, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, inp)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            print('Conv: %s' % name)
            return output

    @layer
    def relu(self, inp, name):
        print('ReLU: %s' % name)
        return tf.nn.relu(inp, name=name)

    @layer
    def prelu(self, inp, name):
        print('PReLU: %s' % name)
        with tf.variable_scope(name):
            i = inp.get_shape().as_list()
            alpha = self.make_var('alpha', shape=(i[-1]))
            output = tf.nn.relu(inp) + tf.mul(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        print('MaxPool: %s' % name)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, inp, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        print('AvgPool: %s' % name)
        self.validate_padding(padding)
        return tf.nn.avg_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, inp, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(inp,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        print('Fc: %s' % name)
        with tf.variable_scope(name) as scope:
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc


    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    @layer
    def softmax(self, target, axis, name=None):
        print('Softmax: %s' % name)
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax
    
    @layer
    def batch_normalization(self, inp, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name):
            shape = [inp.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                inp,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, inp, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(inp, keep, name=name)


class PNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))

        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
        
class RNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1,name='prob1'))

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))

def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    # im: input image
    # minsize: minimum of faces' size
    # pnet, rnet, onet: caffemodel
    # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
    # fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
    factor_count=0
    total_boxes=np.empty((0,9))
    points=[]
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    #img=single(img);
#     if fastresize
#         im_data=(single(img)-127.5)*0.0078125;
#     end
    m=12.0/minsize
    minl=minl*m
    # creat scale pyramid
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1

    # first stage
    #for j = 1:size(scales,2)
    for j in range(len(scales)):
        scale=scales[j]
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))
        #im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
        #im_data = (misc.imresize(img, (hs, ws), interp='bilinear')-127.5)*0.0078125
        #im_data = (cv2.resize(img, (hs, ws), interpolation=cv2.INTER_LINEAR)-127.5)*0.0078125
        #im_data = (cv2.resize(img, (hs, ws), interpolation=cv2.INTER_NEAREST)-127.5)*0.0078125
        #im_data = (img[0:hs,0:ws,:]-127.5)*0.0078125
        im_data = imResample2(img, (hs, ws), 'bilinear')
        im_data = (im_data-127.5)*0.0078125
        
        # PNet.blobs('data').reshape([hs ws 3 1]);
        # out=PNet.forward({im_data});

        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0,2,1,3))
        #out0 = out[0]
        out1 = np.transpose(out[1], (0,2,1,3))
        #out1 = out[1]
        
        #boxes=generateBoundingBox(out{2}(:,:,2), out{1}, scale, threshold[0]);
        boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])
        
        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        boxes = boxes[pick,:]  # boxes=boxes(pick,:);
        if boxes.size>0: # if ~isempty(boxes)
            total_boxes = np.append(total_boxes, boxes, axis=0)  # total_boxes=[total_boxes;boxes];

    numbox = total_boxes.shape[0]  # numbox=size(total_boxes,1);
    if numbox>0:  #   if ~isempty(total_boxes)
        pick = nms(total_boxes.copy(), 0.7, 'Union')  #     pick=nms(total_boxes,0.7,'Union');
        total_boxes = total_boxes[pick,:] #     total_boxes=total_boxes(pick,:);
        regw = total_boxes[:,2]-total_boxes[:,0] #     regw=total_boxes(:,3)-total_boxes(:,1);
        regh = total_boxes[:,3]-total_boxes[:,1] #     regh=total_boxes(:,4)-total_boxes(:,2);
        # total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw   total_boxes(:,2)+total_boxes(:,7).*regh   total_boxes(:,3)+total_boxes(:,8).*regw   total_boxes(:,4)+total_boxes(:,9).*regh   total_boxes(:,5)];
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy()) # total_boxes=rerec(total_boxes);
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])  # total_boxes(:,1:4)=fix(total_boxes(:,1:4));
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0] # numbox=size(total_boxes,1);
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,3,numbox)) # tempimg=zeros(24,24,3,numbox);
        for k in range(0,numbox):  # for k=1:numbox
            tmp = np.zeros((tmph[k],tmpw[k],3))  #       tmp=zeros(tmph(k),tmpw(k),3);
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]  # tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:  # if size(tmp,1)>0 && size(tmp,2)>0 || size(tmp,1)==0 && size(tmp,2)==0
                tempimg[:,:,:,k] = imResample2(tmp, (24, 24), 'bilinear')  #                 tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
            else:
                return np.empty()  #  total_boxes = []; return;
        tempimg = (tempimg-127.5)*0.0078125 #         tempimg=(tempimg-127.5)*0.0078125;
        # RNet.blobs('data').reshape([24 24 3 numbox]);
        # out=RNet.forward({tempimg});
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]  # score=squeeze(out{2}(2,:));
        ipass = np.where(score>threshold[1]) # pass=find(score>threshold(2));
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])   # total_boxes=[total_boxes(pass,1:4) score(pass)'];
        mv = out0[:,ipass[0]]  # mv=out{1}(:,pass);
        if total_boxes.shape[0]>0:  #     if size(total_boxes,1)>0    
            pick = nms(total_boxes, 0.7, 'Union')  # pick=nms(total_boxes,0.7,'Union');
            total_boxes = total_boxes[pick,:]  # total_boxes=total_boxes(pick,:);     
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick])) # total_boxes=bbreg(total_boxes,mv(:,pick)');  
            total_boxes = rerec(total_boxes.copy()) # total_boxes=rerec(total_boxes);

    numbox = total_boxes.shape[0]  #     numbox=size(total_boxes,1);
    if numbox>0:
        # third stage
        total_boxes=np.fix(total_boxes)  # total_boxes=fix(total_boxes);
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h) # [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
        tempimg = np.zeros((48,48,3,numbox))  # tempimg=zeros(48,48,3,numbox);
        for k in range(0,numbox):  # for k=1:numbox
            tmp = np.zeros((tmph[k],tmpw[k],3))#         tmp=zeros(tmph(k),tmpw(k),3);
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]          # tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:  # if size(tmp,1)>0 && size(tmp,2)>0 || size(tmp,1)==0 && size(tmp,2)==0
                tempimg[:,:,:,k] = imResample2(tmp, (48, 48), 'bilinear')  # tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
            else:
                return np.empty()  #  total_boxes = []; return;
        tempimg = (tempimg-127.5)*0.0078125 #         tempimg=(tempimg-127.5)*0.0078125;
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = onet(tempimg1)  # out=ONet.forward({tempimg});
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1,:]  # score=squeeze(out{3}(2,:));
        points = out1 # points=out{2};
        ipass = np.where(score>threshold[2]) #       pass=find(score>threshold(3));
        points = points[:,ipass[0]] # points=points(:,pass);
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])   # total_boxes=[total_boxes(pass,1:4) score(pass)'];
        mv = out0[:,ipass[0]]  # mv=out{1}(:,pass);

        w = total_boxes[:,2]-total_boxes[:,0]+1 # w=total_boxes(:,3)-total_boxes(:,1)+1;
        h = total_boxes[:,3]-total_boxes[:,1]+1   # h=total_boxes(:,4)-total_boxes(:,2)+1;
        points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1   # points(1:5,:)=repmat(w',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;
        points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1 # points(6:10,:)=repmat(h',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;
        if total_boxes.shape[0]>0:  #     if size(total_boxes,1)>0    
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))  # total_boxes=bbreg(total_boxes,mv(:,:)');
            pick = nms(total_boxes.copy(), 0.7, 'Min')  # pick=nms(total_boxes,0.7,'Min');
            total_boxes = total_boxes[pick,:]  # total_boxes=total_boxes(pick,:);
            points = points[:,pick]  # points=points(:,pick);
                
    return total_boxes, points
            
 
# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    # calibrate bouding boxes
    if reg.shape[1]==1:  # if size(reg,2)==1
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))  #     reg=reshape(reg,[size(reg,3) size(reg,4)])';

    w = boundingbox[:,2]-boundingbox[:,0]+1  #   w=[boundingbox(:,3)-boundingbox(:,1)]+1;
    h = boundingbox[:,3]-boundingbox[:,1]+1  #   h=[boundingbox(:,4)-boundingbox(:,2)]+1;
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ])) # boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w boundingbox(:,2)+reg(:,2).*h boundingbox(:,3)+reg(:,3).*w boundingbox(:,4)+reg(:,4).*h];
    return boundingbox
 
def generateBoundingBox(imap, reg, scale, t):
    # use heatmap to generate bounding boxes
    stride=2
    cellsize=12

    imap = np.transpose(imap)  #     map=map';
    dx1 = np.transpose(reg[:,:,0]) # dx1=reg(:,:,1)';
    dy1 = np.transpose(reg[:,:,1]) # dy1=reg(:,:,2)';
    dx2 = np.transpose(reg[:,:,2]) # dx2=reg(:,:,3)';
    dy2 = np.transpose(reg[:,:,3]) # dy2=reg(:,:,4)';
    y, x = np.where(imap > t)  # [y x]=find(map>=t);
    if y.shape[0]==1:  # if size(y,1)==1
      # *** not checked ***
#         y=y';x=x';score=map(a)';dx1=dx1';dy1=dy1';dx2=dx2';dy2=dy2';
        y = np.transpose(y)
        x = np.transpose(x)
        dx1 = np.transpose(dx1)
        dy1 = np.transpose(dy1)
        dx2 = np.transpose(dx2)
        dy2 = np.transpose(dy2)
        score = imap[(y,x)]
    else:
        score = imap[(y,x)]

    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))  #     reg=[dx1(a) dy1(a) dx2(a) dy2(a)];
    if reg.size==0:   #     if isempty(reg)
        # *** not checked ***
        reg = np.empty((0,3))    # reg=reshape([],[0 3]);
    bb = np.transpose(np.vstack([y,x]))  #     boundingbox=[y x];
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])  # boundingbox=[fix((stride*(boundingbox-1)+1)/scale) fix((stride*(boundingbox-1)+cellsize-1+1)/scale) score reg];
    return boundingbox, reg
 
# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:  # if isempty(boxes)
        return np.empty((0,3))   ###### check dimension  #######
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)  # [vals, I] = sort(s);
    #vals = s[I]
    pick = np.zeros_like(s, dtype=np.int16)   # s*0;
    counter = 0
    while I.size>0:   # ~isempty(I)
        #last = I.size
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx]) # xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = np.maximum(y1[i], y1[idx]) # yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = np.minimum(x2[i], x2[idx]) # xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = np.minimum(y2[i], y2[idx]) # yy2 = min(y2(i), y2(I(1:last-1)));  
        w = np.maximum(0.0, xx2-xx1+1)   # w = max(0.0, xx2-xx1+1);
        h = np.maximum(0.0, yy2-yy1+1)   # h = max(0.0, yy2-yy1+1); 
        inter = w * h #     inter = w.*h;
        if method is 'Min':  #     if strcmp(type,'Min')
            o = inter / np.minimum(area[i], area[idx])  # o = inter ./ min(area(i),area(I(1:last-1)));
        else:
            o = inter / (area[i] + area[idx] - inter)   # o = inter ./ (area(i) + area(I(1:last-1)) - inter);
        I = I[np.where(o<=threshold)]  #     I = I(find(o<=threshold));
    pick = pick[0:counter]  #   pick = pick(1:(counter-1));
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)
    tmpw = total_boxes[:,2]-total_boxes[:,0]+1  # tmpw=total_boxes(:,3)-total_boxes(:,1)+1;
    tmph = total_boxes[:,3]-total_boxes[:,1]+1 # tmph=total_boxes(:,4)-total_boxes(:,2)+1;
    numbox=total_boxes.shape[0]

    dx = np.ones((numbox,1))   # dx=ones(numbox,1);
    dy = np.ones((numbox,1))   # dy=ones(numbox,1);
    edx=tmpw.copy()
    edy=tmph.copy()

    x = total_boxes[:,0].copy()  # x=total_boxes(:,1);
    y = total_boxes[:,1].copy()  # y=total_boxes(:,2);
    ex = total_boxes[:,2].copy() # ex=total_boxes(:,3);
    ey = total_boxes[:,3].copy()  # ey=total_boxes(:,4);    

    tmp = np.where(ex>w)  #   tmp=find(ex>w);
    edx[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)  # edx(tmp)=-ex(tmp)+w+tmpw(tmp);ex(tmp)=w;
    ex[tmp] = w
    
    tmp = np.where(ey>h) # tmp=find(ey>h);
    edy[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1) #   edy(tmp)=-ey(tmp)+h+tmph(tmp);ey(tmp)=h;  
    ey[tmp] = h

    tmp = np.where(x<1) #   tmp=find(x<1);
    dx[tmp] = np.expand_dims(2-x[tmp],1)    #   dx(tmp)=2-x(tmp);  
    x[tmp] = 1

    tmp = np.where(y<1) #   tmp=find(y<1);
    dy[tmp] = np.expand_dims(2-y[tmp],1) #   dy(tmp)=2-y(tmp);y(tmp)=1;
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    # convert bboxA to square
    #bboxB = bboxA[:,0:4]  #   bboxB=bboxA(:,1:4);
    h = bboxA[:,3]-bboxA[:,1] #   h=bboxA(:,4)-bboxA(:,2);
    w = bboxA[:,2]-bboxA[:,0] #   w=bboxA(:,3)-bboxA(:,1);
    l = np.maximum(w, h) # l=max([w h]')';
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5 # bboxA(:,1)=bboxA(:,1)+w.*0.5-l.*0.5;
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5 # bboxA(:,2)=bboxA(:,2)+h.*0.5-l.*0.5;
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1))) # bboxA(:,3:4)=bboxA(:,1:2)+repmat(l,[1 2]);
    return bboxA

#pylint: disable=unused-argument
def imResample2(img, sz, method):
    h=img.shape[0]
    w=img.shape[1]
    hs, ws = sz
    dx = float(w) / ws
    dy = float(h) / hs
    im_data = np.zeros((hs,ws,3))
    for a1 in range(0,hs):
        for a2 in range(0,ws):
            for a3 in range(0,3):
                im_data[a1,a2,a3] = img[floor(a1*dy),floor(a2*dx),a3]
    return im_data


