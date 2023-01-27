import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.FC_DenseNet_Tiramisu import build_fc_densenet
from models.FC_DenseNet_Tiramisu import build_fc_densenet_concrete
from models.FC_DenseNet_Tiramisu import build_fc_densenet_full_concrete
from models.FC_DenseNet_Tiramisu import build_fc_densenet_bayesian
from models.FC_DenseNet_Tiramisu import build_fc_densenet_full_bayesian
from models.DeepLabV3_plus import build_deeplabv3_plus
from models.DeepLabV3_plus import build_deeplabv3_plus_bayesian
from models.DeepLabV3_plus import build_deeplabv3_plus_concrete


SUPPORTED_MODELS = ["FC-DenseNet103", "DeepLabV3_plus","DeepLabV3_plus_concrete", "DeepLabV3_plus_bayesian", 
    "DenseNet_concrete", "DenseNet_bayesian", "DenseNet_full_bayesian", "DenseNet_full_concrete"]

SUPPORTED_FRONTENDS = ["Xception","XceptionBayesian", "XceptionConcrete", "XceptionMC"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_model_bayesian(model_name, net_input, first_pass, batch_size, num_classes, crop_width, crop_height, frontend="XceptionBayesian", is_training=True, pretrained_file="xception_65.ckpt", one_parameter=False):
    if model_name == "DeepLabV3_plus_bayesian":
	    # DeepLabV3+ requires pre-trained Xception weights
	    network, init_fn, frontend_scope = build_deeplabv3_plus_bayesian(net_input, first_pass, batch_size, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training,pretrained_file=pretrained_file, one_parameter=one_parameter)
    elif model_name == "DenseNet_bayesian":
        network = build_fc_densenet_bayesian(net_input, first_pass, batch_size, preset_model = "FC-DenseNet103", num_classes=num_classes, one_parameter=one_parameter)
        init_fn = None
        frontend_scope = None
    elif model_name == "DenseNet_full_bayesian":
        network = build_fc_densenet_full_bayesian(net_input, first_pass, batch_size, preset_model = "FC-DenseNet103", num_classes=num_classes, one_parameter=True)
        init_fn = None
        frontend_scope = None
    else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")
    return network, init_fn, frontend_scope

def build_model_concrete(model_name, net_input, num_classes, crop_width, crop_height, frontend="XceptionConcrete", is_training=True,  pretrained_file="xception_65.ckpt", one_parameter=False):
    if model_name == "DeepLabV3_plus_concrete":
	    # DeepLabV3+ requires pre-trained Xception weights
	    network, init_fn, frontend_scope = build_deeplabv3_plus_concrete(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training,pretrained_file=pretrained_file, one_parameter=one_parameter)
    elif model_name == "DenseNet_concrete":
        network = build_fc_densenet_concrete(net_input, preset_model = "FC-DenseNet103", num_classes=num_classes, one_parameter=one_parameter)
        init_fn = None
        frontend_scope = None
    elif model_name == "DenseNet_full_concrete":
        network = build_fc_densenet_full_concrete(net_input, preset_model = "FC-DenseNet103", num_classes=num_classes, one_parameter=True)
        init_fn = None
        frontend_scope = None
    else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")
    return network, init_fn, frontend_scope


def build_model_NoMC_dense(model_name, net_input, num_classes, crop_width, crop_height, frontend="XceptionMC", is_training=True,  pretrained_file="xception_65.ckpt"):
    network = None
    init_fn = None
    if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
	    network = build_fc_densenet(net_input, preset_model = model_name, num_classes=num_classes,dropout_p=0.0)
    else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

    return network, init_fn
def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True,  pretrained_file="xception_65.ckpt"):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	if frontend not in SUPPORTED_FRONTENDS:
		raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_FRONTENDS))


	network = None
	init_fn = None
	if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
	    network = build_fc_densenet(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "DeepLabV3_plus":
	    # DeepLabV3+ requires pre-trained ResNet weights
	    network, init_fn = build_deeplabv3_plus(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training,pretrained_file=pretrained_file)

	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn