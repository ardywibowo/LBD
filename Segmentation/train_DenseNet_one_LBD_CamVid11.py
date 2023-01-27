from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess

import matplotlib
matplotlib.use('Agg')

from utils import utils_eval as utils
from utils import helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=700, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=500, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=500, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid11", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=352, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=480, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="DenseNet_bayesian", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="Xception", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()


def data_augmentation(input_image, output_image):
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

###### HYPERPARAMETERS
l = 0.01

weight_regularizer = 1e-8
drop_regularizer = 1.6e-8 #(1/(nwh))

ignore_label = 11

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values_3 = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
label_values=np.array(label_values_3)[:,0]
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess=tf.Session(config=config)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# Compute loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
scalar_output = tf.placeholder(tf.float32,shape=[None,None,None])
first_pass = tf.placeholder(tf.bool,shape=())

network, _, _ = \
    model_builder.build_model_bayesian(model_name=args.model, frontend=args.frontend, net_input=net_input, first_pass=first_pass, \
                                       batch_size=args.batch_size, num_classes=num_classes, crop_width=args.crop_width, \
                                       crop_height=args.crop_height, one_parameter=True)


kl_weight = tf.placeholder_with_default(0.001, shape=())
kl_term = tf.Variable(0.0)

f_d = tf.placeholder(shape=([args.batch_size]),dtype=tf.float32)
p_all = []
u_all = []

for i in range(15):
    densenet_scope = 'FC-DenseNet103' + "/denseblock6" + "/layer%d" % (i)
    
    with tf.variable_scope(densenet_scope, reuse=tf.AUTO_REUSE):
        p_current = tf.get_variable('p')
        u_current = tf.get_variable('u')
        
    with tf.variable_scope(densenet_scope, reuse=tf.AUTO_REUSE):
        w_current = tf.get_variable('Conv/weights')
    
    p_all.append(p_current)
    u_all.append(u_current)
    
    sig = tf.sigmoid(p_current)
    log_sig = tf.log_sigmoid(p_current)
    log_sig_ = tf.log_sigmoid(-p_current)
    
    kl_term += tf.reduce_sum( ( weight_regularizer / (sig) * (tf.reduce_sum(tf.square(w_current),axis=[0,1,2]))) + \
                                                        drop_regularizer*(sig*log_sig + (1.0-sig)*log_sig_))

not_ignore = tf.to_float(tf.not_equal(scalar_output, ignore_label))

loss_all_ = tf.losses.softmax_cross_entropy(tf.reshape(net_output, shape=[-1, num_classes]), tf.reshape(network, shape=[-1, num_classes]), \
                                       weights = tf.reshape(not_ignore, shape=[-1]), reduction=tf.losses.Reduction.NONE)
loss_all_ = tf.reduce_mean(tf.reshape(loss_all_, shape=[-1, tf.shape(net_output)[1], tf.shape(net_output)[2]]),axis=[1,2])
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.reshape(net_output, shape=[-1, num_classes]), tf.reshape(network, shape=[-1, num_classes]), \
                                       weights = tf.reshape(not_ignore, shape=[-1])) )
    
loss_all = loss_all_ +  kl_weight * kl_term
loss = loss + kl_weight * kl_term

opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
trainer2=tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995)

f_d_expanded=tf.expand_dims(f_d,axis=1)

grads2=[tf.reshape(tf.reduce_sum((tf.squeeze(u)-0.5)*f_d_expanded), shape=[1]) for u in u_all]#u is (batch,1,1,channel)
grads_vars_21=trainer2.compute_gradients(kl_term,var_list=[p_all])
grads_21=[a for a,b in grads_vars_21]
grads_vars_2 = [(a+c,b) for a,c,b in zip(grads2,grads_21,p_all)]
opt2 = trainer2.apply_gradients(grads_vars_2)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + "_one.ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = \
    utils.prepare_data(dataset_dir=args.dataset)


print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))


random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

#Train
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt=0

    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):

        input_image_batch = []
        output_image_batch = []
        one_hot_out_batch = []
        
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_gt(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)


                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(output_image)
                
                
                one_hot_out = np.float32(helpers.one_hot_it_gt(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))
                one_hot_out_batch.append(np.expand_dims(one_hot_out, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
            one_hot_out_batch = one_hot_out_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
            one_hot_out_batch = np.squeeze(np.stack(one_hot_out_batch, axis=0))

        _,current,current_all_1=sess.run((opt,loss,loss_all), feed_dict= {net_input:input_image_batch, net_output:one_hot_out_batch, scalar_output:output_image_batch, first_pass:True})
        current_all_2=sess.run((loss_all),feed_dict={net_input:input_image_batch,net_output:one_hot_out_batch,scalar_output:output_image_batch,first_pass:False})
        f_delta = current_all_2 - current_all_1
        _ = sess.run([opt2], feed_dict={f_d:f_delta})
        
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)



    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        if not os.path.isdir("%s/%s/%04d"%("checkpoints", args.model  + args.dataset + "_one", epoch)):
            os.makedirs("%s/%s/%04d"%("checkpoints", args.model + args.dataset + "_one", epoch))
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%s/%04d"%("checkpoints", args.model  + args.dataset + "_one", epoch))


    if epoch % args.validation_step == 0:
        
        if not os.path.isdir("%s/%s/%04d"%("checkpoints", args.model  + args.dataset + "_one", epoch)):
            os.makedirs("%s/%s/%04d"%("checkpoints", args.model  + args.dataset + "_one", epoch))
        print("Performing validation")
        target=open("%s/%s/%04d/val_scores.csv"%("checkpoints", args.model  + args.dataset + "_one", epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []


        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0

            gt = utils.load_gt(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = helpers.reverse_one_hot(helpers.one_hot_it_gt(gt, label_values))


            output_image = sess.run(network,feed_dict={net_input:input_image, first_pass:True})

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou , num_corr_pixs, num_tot_pixs = utils.evaluate_segmentation_vec(pred=output_image, label=gt, num_classes=num_classes, ignore_label = ignore_label)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            
            

        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []

    asghar=[iiii*args.validation_step for iiii in range(len(avg_iou_per_epoch))]


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(asghar, avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch-args.epoch_start_i+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(asghar, avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs.png')



