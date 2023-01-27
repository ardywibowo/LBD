from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm 
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
import skimage
import skimage.io
import skimage.transform

import tensorflow as tf
import tensornets as nets

# GPU Configuration
config = tf.ConfigProto(allow_soft_placement=True)

# "Best-fit with coalescing" algorithm for memory allocation
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80

# CIFAR-10 Dataset Preparation
cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Download CIFAR-10 Dataset
class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

""" 
    check if the data (zip) file is already downloaded
    if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz
"""
if not isfile('cifar-10-python.tar.gz'):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

# Load raw data and reshape input images
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

# One hot encoding for label daa
def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded

# Save the modified input and labels
def _preprocess_and_save(one_hot_encode, features, labels, filename):
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(cifar10_dataset_folder_path, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation], 
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_testing.p')

preprocess_and_save_data(cifar10_dataset_folder_path, one_hot_encode)

# Train the model for CIFAR-10

# Load data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# Hyperparameters
model_name = "VGG_ARM_One"
learning_rate = 0.00001
epochs = 300
batch_size = 100
weight_regularizer = 1e-8
drop_regularizer = 0.00002
checkpoint_steps = 5
save_model_path = './image_classification/OneARM'

# Define input (features) and output (labels)
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
first_pass = tf.placeholder(tf.bool,shape=(), name='first_pass')
kl_weight = tf.placeholder_with_default(1e-8, shape=())
f_d = tf.placeholder(shape=(batch_size), dtype=tf.float32)

# Load VGG19 model, define loss, train, and accuracy operation
logits = nets.VGG19_ARM(x, is_training=True, classes=10, first_pass=first_pass, \
  batch_size=batch_size, one_parameter=True)
model = tf.identity(logits, name='logits')

# Compute ARM Gradients
model_scope = 'vgg19_arm'
with tf.variable_scope(model_scope, reuse=tf.AUTO_REUSE):
  p1 = tf.get_variable('p1')
  p2 = tf.get_variable('p2')
  u1 = tf.get_variable('u1')
  u2 = tf.get_variable('u2')
  w1 = tf.get_variable('fc6/weights')
  w2 = tf.get_variable('fc7/weights')

sig1 = tf.sigmoid(p1)
sig2 = tf.sigmoid(p2)

log_sig1 = tf.log_sigmoid(p1)
log_sig2 = tf.log_sigmoid(p2)
log_sig_1 = tf.log_sigmoid(-p1)
log_sig_2 = tf.log_sigmoid(-p2)

kl_term1 = tf.reduce_sum( (weight_regularizer * tf.reduce_sum(tf.square(w1),axis=0) / sig1) + \
   drop_regularizer * (sig1 * log_sig1 + (1.0 - sig1) * log_sig_1) )
kl_term2 = tf.reduce_sum( (weight_regularizer * tf.reduce_sum(tf.square(w2),axis=0) / sig2) + \
   drop_regularizer * (sig2 * log_sig1 + (1.0 - sig2) * log_sig_2) )

kl_term = kl_term1 + kl_term2

p_all = [p1, p2]
u_all = [u1, u2]

loss = tf.losses.softmax_cross_entropy(y, logits, reduction=tf.losses.Reduction.NONE)
loss = loss + kl_weight * kl_term

loss_sum = tf.reduce_mean(loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_sum)

trainer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)

f_d_expanded=tf.expand_dims(f_d,axis=1)
grads2 = [tf.reshape(tf.reduce_sum((tf.squeeze(u) - 0.5) * f_d_expanded), shape=[1]) for u in u_all]

grads_vars_21 = trainer2.compute_gradients(kl_term, var_list=[p_all])
grads_21=[a for a,b in grads_vars_21]
grads_vars_2 = [(a+c,b) for a,c,b in zip(grads2,grads_21,p_all)]
opt2 = trainer2.apply_gradients(grads_vars_2)

correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# Print outputs of each layer
logits.print_outputs()
logits.print_summary()

# Get batches
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        feature_out = [np.expand_dims(f, axis=0) for f in features[start:end]]
        labels_out = labels[start:end]
        while len(feature_out) < batch_size:
            feature_next = [np.expand_dims(f, axis=0) for f in features[0 : batch_size - len(feature_out)]]
            for f in feature_next:
                feature_out.append(f)
            labels_out = np.append(labels_out, labels[0 : batch_size - len(feature_out)], axis=0)

        yield np.vstack(feature_out), labels_out


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    
    tmpFeatures = []
    
    for feature in features:
        tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
        tmpFeatures.append(tmpFeature)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(tmpFeatures, labels, batch_size)

# Get inputs for validation

tmpValidFeatures = []

for feature in valid_features:
    tmpValidFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    tmpValidFeatures.append(tmpValidFeature)
    
tmpValidFeatures = np.array(tmpValidFeatures)

print(tmpValidFeatures.shape)

print('Training...')
with tf.Session() as sess:    
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    print('global_variables_initializer ... done ...')
    sess.run(logits.pretrained())
    print('model.pretrained ... done ... ')    
    
    # Training cycle
    print('starting training ... ')
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                _, current_all_1 = sess.run((train, loss), {x: batch_features, y: batch_labels, first_pass:True})
                current_all_2=sess.run((loss),feed_dict={x: batch_features, y: batch_labels, first_pass:False})
                f_delta = current_all_2 - current_all_1
                sess.run([opt2], feed_dict={f_d:f_delta})
                
                
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            
            # calculate the mean accuracy over all validation dataset
            valid_acc = 0
            for batch_valid_features, batch_valid_labels in batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
                valid_acc += sess.run(accuracy, {x:batch_valid_features, y:batch_valid_labels, first_pass:True})
            
            tmp_num = tmpValidFeatures.shape[0]/batch_size
            print('Validation Accuracy: {:.6f}'.format(valid_acc/tmp_num))
            
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)

        if epoch % checkpoint_steps == 0:
            # Create directories if needed
            if not os.path.isdir("%s/%s/%04d"%("checkpoints", model_name, epoch)):
                os.makedirs("%s/%s/%04d"%("checkpoints", model_name, epoch))
            print("Saving checkpoint for this epoch")
            saver.save(sess,"%s/%s/%04d"%("checkpoints", model_name, epoch))

# Testing the model over random sample images

# Mapping label index to label name
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display prediction results
from sklearn.preprocessing import LabelBinarizer

def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axs = plt.subplots(10, 2, figsize=(12,24))

    margin = 0.05
    ind = np.arange(n_classes)
    width = (1. - 2. * margin) / n_classes    
    
    for image_i, (feature, label_id, prediction) in enumerate(zip(features, label_ids, predictions)):
        correct_name = label_names[label_id]
        pred_name = label_names[np.argmax(prediction)]
        
        is_match = 'False'        
        
        if np.argmax(prediction) == label_id:
            is_match = 'True'
            
        predictions_array = []
        pred_names = []
        
        for index, pred_value in enumerate(prediction):
            tmp_pred_name = label_names[index]
            predictions_array.append({tmp_pred_name : pred_value})
            pred_names.append(tmp_pred_name)
        
        print('[{}] ground truth: {}, predicted result: {} | {}'.format(image_i, correct_name, pred_name, is_match))
        print('\t- {}\n'.format(predictions_array))
        
#         print('image_i: ', image_i)
#         print('axs: ', axs, ', axs len: ', len(axs))
        axs[image_i][0].imshow(feature)
        axs[image_i][0].set_title(pred_name)
        axs[image_i][0].set_axis_off()
        
        axs[image_i][1].barh(ind + margin, prediction, width)
        axs[image_i][1].set_yticks(ind + margin)
        axs[image_i][1].set_yticklabels(pred_names)
        
    plt.tight_layout()

# Load dataset for testing
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))
tmpFeatures = []

for feature in test_features:
    tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    tmpFeatures.append(tmpFeature)

tmpFeatures = np.asarray(tmpFeatures)

# Testing
import random

save_model_path = './image_classification'
batch_size = 100
n_samples = 10
top_n_predictions = 5

def test_model(tmpFeatures):
    loaded_graph = tf.Graph()
    
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
        loaded_first_pass = loaded_graph.get_tensor_by_name('first_pass:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        for train_feature_batch, train_label_batch in batch_features_labels(tmpFeatures, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_first_pass:True})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        
        tmpTestFeatures = []
    
        for feature in random_test_features:
            tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
            tmpTestFeatures.append(tmpFeature)
           
        random_test_predictions = sess.run(
            tf.nn.softmax(loaded_logits),
            feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels, loaded_first_pass:True})
        
        display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

test_model(tmpFeatures)
