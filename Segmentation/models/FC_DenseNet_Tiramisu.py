from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # Stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features


def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
  """
  with tf.name_scope(scope) as sc:
    # Upsample
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l

def build_fc_densenet(inputs, num_classes, preset_model='FC-DenseNet56', n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None):
    """
    Builds the FC-DenseNet model

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)

    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
      stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net
  
def preact_conv_concrete(inputs, n_filters, kernel_size=[3, 3], one_parameter=False):
    """
    Basic pre-activation layer for DenseNets with Concrete dropout
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    
    """
    preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    
    # Add Dropout
    eps2=1e-15
    drop_gen = tf.distributions.Uniform()
    
    if one_parameter == False:
        p = tf.get_variable("p", [n_filters],initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05))
        temper=0.67
        u_samp = drop_gen.sample([tf.shape(conv)[0],1,1,tf.shape(conv)[3]])
        prob_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.sigmoid(p),axis=0),axis=0),axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
    else:
        p = tf.get_variable("p", [1],initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05))
        temper=0.67
        u_samp = drop_gen.sample([tf.shape(conv)[0],1,1,tf.shape(conv)[3]])
        prob_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.sigmoid(p),axis=0),axis=0),axis=0)
        mask = tf.sigmoid((1.0/temper) * (p - tf.log(u_samp+eps2) + tf.log(1.0-u_samp+eps2)))/(prob_exp)
    
    conv = conv * mask
    return conv

def DenseBlockConcrete(stack, n_layers, growth_rate, scope=None, one_parameter=False):
  """
  DenseBlock for DenseNet and FC-DenseNet with Concrete dropout
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.variable_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
        with tf.variable_scope('layer%d' % (j)) as sc:
          # Compute new feature maps
          layer = preact_conv_concrete(stack, growth_rate, one_parameter=one_parameter)
          new_features.append(layer)
          # Stack new layer
          stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features

def build_fc_densenet_concrete(inputs, num_classes, preset_model='FC-DenseNet103', n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, scope=None, one_parameter=False):
    """
    Builds the FC-DenseNet model with Concrete dropout

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1

    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    
    dropout_p = 0.2
    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
      stack, block_to_upsample = DenseBlockConcrete(stack, n_layers_per_block[n_pool], growth_rate, scope='denseblock%d' % (n_pool + 1), one_parameter=one_parameter)


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net

def build_fc_densenet_full_concrete(inputs, num_classes, preset_model='FC-DenseNet103', n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, scope=None, one_parameter=False):
    """
    Builds the FC-DenseNet model with full Concrete dropout

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1

    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    
    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlockConcrete(stack, n_layers_per_block[i], growth_rate, scope='denseblock%d' % (i+1), one_parameter=one_parameter)
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
      stack, block_to_upsample = DenseBlockConcrete(stack, n_layers_per_block[n_pool], growth_rate, scope='denseblock%d' % (n_pool + 1), one_parameter=one_parameter)


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlockConcrete(stack, n_layers_per_block[n_pool + i + 1], growth_rate, scope='denseblock%d' % (n_pool + i + 2), one_parameter=one_parameter)


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net

def preact_conv_bayesian(inputs, first_pass, batch_size, n_filters, kernel_size=[3, 3], one_parameter=False):
    """
    Basic pre-activation layer for DenseNets with LBD
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    """
    preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    
    # Add dropout here
    if one_parameter == False:             
        drop_gen = tf.distributions.Uniform()
        u_samp = tf.get_variable("u", [batch_size,1,1,n_filters], trainable=False)
        p = tf.get_variable("p", [n_filters], initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = tf.cond(tf.equal(first_pass,True), lambda: drop_gen.sample([tf.shape(conv)[0],1,1,tf.shape(conv)[3]]), lambda: u_samp)
        prob = tf.sigmoid(p)
        prob_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(prob,axis=0),axis=0),axis=0)
        mask = tf.cond(tf.equal(first_pass,True),lambda: tf.cast(tf.greater(-u_samp,-prob_exp),tf.float32)/(prob_exp),lambda: tf.cast(tf.greater(u_samp,1.-prob_exp),tf.float32)/(prob_exp))    
    else:
        drop_gen = tf.distributions.Uniform()
        u_samp = tf.get_variable("u", [batch_size,1,1,n_filters], trainable=False)
        p = tf.get_variable("p", [1], initializer=tf.initializers.random_normal(mean=0.0,stddev=0.05), trainable=False)
        u_samp = tf.cond(tf.equal(first_pass,True), lambda: drop_gen.sample([tf.shape(conv)[0],1,1,tf.shape(conv)[3]]), lambda: u_samp)
        prob = tf.sigmoid(p)
        prob_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(prob,axis=0),axis=0),axis=0)
        mask = tf.cond(tf.equal(first_pass,True),lambda: tf.cast(tf.greater(-u_samp,-prob_exp),tf.float32)/(prob_exp),lambda: tf.cast(tf.greater(u_samp,1.-prob_exp),tf.float32)/(prob_exp))    
        

    conv = conv * mask
    return conv

def DenseBlockBayesian(stack, first_pass, batch_size, n_layers, growth_rate, scope=None, one_parameter=False):
  """
  DenseBlock for DenseNet and FC-DenseNet with LBD
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
#  with tf.name_scope(scope) as sc:
  with tf.variable_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
        with tf.variable_scope('layer%d' % (j)) as sc:
          # Compute new feature maps
          layer = preact_conv_bayesian(stack, first_pass, batch_size, growth_rate, one_parameter=one_parameter)
          new_features.append(layer)
          # Stack new layer
          stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features

def build_fc_densenet_bayesian(inputs, first_pass, batch_size, num_classes, preset_model='FC-DenseNet103', \
                               n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, scope=None, one_parameter=False):
    """
    Builds the FC-DenseNet model with LBD

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1

    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    
    dropout_p = 0.2
    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
#      stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))
      stack, block_to_upsample = DenseBlockBayesian(stack, first_pass, batch_size, n_layers_per_block[n_pool], growth_rate, scope='denseblock%d' % (n_pool + 1), one_parameter=one_parameter)


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net
  
def build_fc_densenet_full_bayesian(inputs, first_pass, batch_size, num_classes, preset_model='FC-DenseNet103', \
                               n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, scope=None, one_parameter=False):
    """
    Builds the FC-DenseNet model with LBD

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1

    Returns:
      Fc-DenseNet model
    """

    if preset_model == 'FC-DenseNet56':
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate=16
      n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    
    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None)

      n_filters = n_filters_first_conv
      
      #####################
      # Downsampling path #
      #####################

      skip_connection_list = []

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlockBayesian(stack, first_pass, batch_size, n_layers_per_block[i], growth_rate, scope='denseblock%d' % (i+1), one_parameter=one_parameter)
        n_filters += growth_rate * n_layers_per_block[i]
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))

      skip_connection_list = skip_connection_list[::-1]

      #####################
      #     Bottleneck    #
      #####################

      # Dense Block
      # We will only upsample the new feature maps
#      stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + 1))
      stack, block_to_upsample = DenseBlockBayesian(stack, first_pass, batch_size, n_layers_per_block[n_pool], growth_rate, scope='denseblock%d' % (n_pool + 1), one_parameter=one_parameter)


      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlockBayesian(stack, first_pass, batch_size, n_layers_per_block[n_pool + i + 1], growth_rate, scope='denseblock%d' % (n_pool + i + 2), one_parameter=one_parameter)


      #####################
      #      Softmax      #
      #####################
      net = slim.conv2d(stack, num_classes, [1, 1], activation_fn=None, scope='logits')
      return net