import tensorflow as tf
from tensorflow.contrib import slim
from frontends import resnet_v2
from frontends import mobilenet_v2
from frontends import inception_v4
from frontends import xception
import os 


def build_frontend_bayesian(inputs, first_pass, batch_size, frontend, is_training=True, pretrained_dir="models", pretrained_file="xception_65.ckpt",one_parameter=False):
    if frontend == 'XceptionBayesian':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception_65_bayesian(inputs, first_pass, batch_size, is_training=is_training, scope='xception_65',one_parameter=one_parameter)
            frontend_scope='xception_65'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, pretrained_file), var_list=slim.get_model_variables('xception_65'), ignore_missing_vars=True)
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports Xception" % (frontend))

    return logits, end_points, frontend_scope, init_fn 

def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models", pretrained_file=None,one_parameter=False):

    if frontend == 'XceptionConcrete':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception_65_concrete(inputs, is_training=is_training, scope='xception_65',one_parameter=one_parameter)
            frontend_scope='xception_65'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, pretrained_file), var_list=slim.get_model_variables('xception_65'), ignore_missing_vars=True)
    
    elif frontend == 'Xception':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception_65_concrete(inputs, is_training=is_training, scope='xception_65')
            frontend_scope='xception_65'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, pretrained_file), var_list=slim.get_model_variables('xception_65'), ignore_missing_vars=True)
    elif frontend == 'XceptionMC':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception_65_mc(inputs, is_training=is_training, scope='xception_65')
            frontend_scope='xception_65'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, pretrained_file), var_list=slim.get_model_variables('xception_65'), ignore_missing_vars=True)
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports Xception" % (frontend))

    return logits, end_points, frontend_scope, init_fn 