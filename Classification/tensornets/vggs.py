"""Collection of VGG variants

The reference paper:

 - Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015
 - Karen Simonyan, Andrew Zisserman
 - https://arxiv.org/abs/1409.1556

The reference implementation:

1. Keras
 - https://github.com/keras-team/keras/blob/master/keras/applications/vgg{16,19}.py
2. Caffe VGG
 - http://www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import conv2d
from .layers import dropout
from .layers import flatten
from .layers import fc
from .layers import max_pool2d
from .layers import convrelu as conv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([dropout], {'is_training': is_training}),
            ([flatten], {'scope': 'flatten'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'})]


@var_scope('stack')
def _stack(x, filters, blocks, scope=None):
    for i in range(1, blocks+1):
        x = conv(x, filters, 3, scope=str(i))
    x = max_pool2d(x, 2, stride=2)
    return x


def vgg(x, blocks, is_training, classes, stem, scope=None, reuse=None):
    x = _stack(x, 64, blocks[0], scope='conv1')
    x = _stack(x, 128, blocks[1], scope='conv2')
    x = _stack(x, 256, blocks[2], scope='conv3')
    x = _stack(x, 512, blocks[3], scope='conv4')
    x = _stack(x, 512, blocks[4], scope='conv5')
    if stem: return x
    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')
    x = dropout(x, keep_prob=0.5, scope='drop6')
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    x = dropout(x, keep_prob=0.5, scope='drop7')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x

def vgg_arm(x, blocks, is_training, classes, stem, scope=None, reuse=None, \
    batch_size=32, first_pass=True, one_parameter=False):
    x = _stack(x, 64, blocks[0], scope='conv1')
    x = _stack(x, 128, blocks[1], scope='conv2')
    x = _stack(x, 256, blocks[2], scope='conv3')
    x = _stack(x, 512, blocks[3], scope='conv4')
    x = _stack(x, 512, blocks[4], scope='conv5')
    if stem: return x

    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')

    # Dropout 1
    drop_gen = tf.distributions.Uniform()

    if one_parameter == False:
        p1 = tf.get_variable("p1", x.shape[1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        prob_exp = tf.expand_dims(tf.sigmoid(p1), axis=0)
        u1 = tf.get_variable("u1", [batch_size, 4096], trainable=False)
        u1 = tf.cond(tf.equal(first_pass,True), lambda: \
            drop_gen.sample([tf.shape(x)[0], tf.shape(x)[1]]), lambda: u1)

        mask11 = tf.cast(tf.greater(-u1,-prob_exp),tf.float32)/(prob_exp)
        mask12 = tf.cast(tf.greater(u1,1.-prob_exp),tf.float32)/(prob_exp)
        mask1 = tf.cond(tf.equal(first_pass,True), lambda: mask11, lambda: mask12)
        x = x * mask1
    else:
        p1 = tf.get_variable("p1", [1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        prob_exp = tf.expand_dims(tf.sigmoid(p1), axis=[0])
        u1 = tf.get_variable("u1", [batch_size, 4096], trainable=False)
        u1 = tf.cond(tf.equal(first_pass,True), lambda: \
            drop_gen.sample([tf.shape(x)[0], tf.shape(x)[1]]), lambda: u1)

        mask11 = tf.cast(tf.greater(-u1,-prob_exp),tf.float32)/(prob_exp)
        mask12 = tf.cast(tf.greater(u1,1.-prob_exp),tf.float32)/(prob_exp)
        mask1 = tf.cond(tf.equal(first_pass,True), lambda: mask11, lambda: mask12)
        x = x * mask1
    # x = dropout(x, keep_prob=1, scope='drop6')
    
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    
    # Dropout 2
    if one_parameter == False:
        p2 = tf.get_variable("p2", x.shape[1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        prob_exp = tf.expand_dims(tf.sigmoid(p2), axis=0)
        u2 = tf.get_variable("u2", [batch_size, 4096], trainable=False)
        u2 = tf.cond(tf.equal(first_pass,True), lambda: \
            drop_gen.sample([tf.shape(x)[0], tf.shape(x)[1]]), lambda: u2)

        mask21 = tf.cast(tf.greater(-u2, -prob_exp),tf.float32)/(prob_exp)
        mask22 = tf.cast(tf.greater(u2, 1. - prob_exp),tf.float32)/(prob_exp)
        mask2 = tf.cond(tf.equal(first_pass,True), lambda: mask21, lambda: mask22)
        x = x * mask2
    else:
        p2 = tf.get_variable("p2", [1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        prob_exp = tf.expand_dims(tf.sigmoid(p2), axis=[0])
        u2 = tf.get_variable("u2", [batch_size, 4096], trainable=False)
        u2 = tf.cond(tf.equal(first_pass,True), lambda: \
            drop_gen.sample([tf.shape(x)[0], tf.shape(x)[1]]), lambda: u2)

        mask21 = tf.cast(tf.greater(-u2, -prob_exp),tf.float32)/(prob_exp)
        mask22 = tf.cast(tf.greater(u2, 1. - prob_exp),tf.float32)/(prob_exp)
        mask2 = tf.cond(tf.equal(first_pass,True), lambda: mask21, lambda: mask22)
        x = x * mask2
    # x = dropout(x, keep_prob=1, scope='drop7')
    
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    
    return x

def vgg_concrete(x, blocks, is_training, classes, stem, scope=None, reuse=None, one_parameter=False):
    x = _stack(x, 64, blocks[0], scope='conv1')
    x = _stack(x, 128, blocks[1], scope='conv2')
    x = _stack(x, 256, blocks[2], scope='conv3')
    x = _stack(x, 512, blocks[3], scope='conv4')
    x = _stack(x, 512, blocks[4], scope='conv5')
    if stem: return x
    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')

    #   Add Concrete Dropout
    drop_gen = tf.distributions.Uniform()
    eps2=1e-15
    temper=0.67

    if one_parameter == False:
        p = tf.get_variable("p1", x.shape[1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = drop_gen.sample(tf.shape(x))
        prob_exp = tf.expand_dims(tf.sigmoid(p), axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
        x = x * mask
        x = dropout(x, keep_prob=1, scope='drop6')
    else:
        p = tf.get_variable("p1", [1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = drop_gen.sample(tf.shape(x))
        prob_exp = tf.expand_dims(tf.sigmoid(p), axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
        x = x * mask
        x = dropout(x, keep_prob=1, scope='drop6')
    
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')

    if one_parameter == False:
        p = tf.get_variable("p2", x.shape[1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = drop_gen.sample(tf.shape(x))
        prob_exp = tf.expand_dims(tf.sigmoid(p), axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
        x = x * mask
        x = dropout(x, keep_prob=1, scope='drop7')
    else:
        p = tf.get_variable("p2", [1], \
            initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = drop_gen.sample(tf.shape(x))
        prob_exp = tf.expand_dims(tf.sigmoid(p), axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
        x = x * mask
        x = dropout(x, keep_prob=1, scope='drop7')        
    
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('vgg16')
@set_args(__args__)
def vgg16(x, is_training=False, classes=1000,
          stem=False, scope=None, reuse=None):
    return vgg(x, [2, 2, 3, 3, 3], is_training, classes, stem, scope, reuse)


@var_scope('vgg19')
@set_args(__args__)
def vgg19(x, is_training=False, classes=1000,
          stem=False, scope=None, reuse=None):
    return vgg(x, [2, 2, 4, 4, 4], is_training, classes, stem, scope, reuse)

@var_scope('vgg19_arm')
@set_args(__args__)
def vgg19_arm(x, is_training=False, classes=1000,
          stem=False, scope=None, reuse=None, batch_size=32, first_pass=True, one_parameter=False):
    return vgg_arm(x, [2, 2, 4, 4, 4], is_training, classes, \
        stem, scope, reuse, batch_size, first_pass, one_parameter)

@var_scope('vgg19_concrete')
@set_args(__args__)
def vgg19_concrete(x, is_training=False, classes=1000,
          stem=False, scope=None, reuse=None, one_parameter=False):
    return vgg_concrete(x, [2, 2, 4, 4, 4], is_training, classes, stem, scope, reuse, one_parameter=one_parameter)

# Simple alias.
VGG16 = vgg16
VGG19 = vgg19
VGG19_ARM = vgg19_arm
VGG19_Concrete = vgg19_concrete
