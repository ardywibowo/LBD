# Testing the model over random sample images
from sklearn.preprocessing import LabelBinarizer
import random
import pickle
import skimage
import skimage.transform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Mapping label index to label name
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display prediction results
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

# Get batches
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        feature_out = [np.expand_dims(f, axis=0) for f in features[start:end]]
        labels_out = labels[start:end]
        # true_batch_size = len(feature_out)
        while len(feature_out) < batch_size:
            feature_next = [np.expand_dims(f, axis=0) for f in features[0 : batch_size - len(feature_out)]]
            for f in feature_next:
                feature_out.append(f)
            labels_out = np.append(labels_out, labels[0 : batch_size - len(feature_out)], axis=0)

        yield np.vstack(feature_out), labels_out

# Load dataset for testing
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))
tmpFeatures = []

for feature in test_features:
    tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    tmpFeatures.append(tmpFeature)

tmpFeatures = np.asarray(tmpFeatures)

# Testing
save_model_path = './image_classification/ARM/ARM'
batch_size = 100
n_samples = 10
top_n_predictions = 5
num_mc = 10

def test_model(tmpFeatures):
    loaded_graph = tf.Graph()
    
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        test_batch_softmax = np.zeros([10000, 10])
        test_batch_entropy = np.zeros(10000)

        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        # loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        first_pass = loaded_graph.get_tensor_by_name('first_pass:0')
        
        for train_feature_batch, train_label_batch in batch_features_labels(tmpFeatures, test_labels, batch_size):
            softmax_total = np.zeros([batch_size, 10])
            for i in range(num_mc):
                softmax_total += sess.run(tf.nn.softmax(loaded_logits),
                    feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, first_pass: True})

            # Softmax
            softmax_total = softmax_total/float(num_mc)
            prediction = np.argmax(softmax_total, axis=1)
            trues = np.argmax(train_label_batch, axis=1)
            test_batch_softmax[test_batch_count * batch_size : (test_batch_count+1) * batch_size] = softmax_total

            # Entropy
            log_softmax_total = np.log(softmax_total+1e-30)
            test_batch_entropy[test_batch_count * batch_size : (test_batch_count+1) * batch_size] = -np.sum(softmax_total * log_softmax_total)

            # Accuracy
            current_accuracy = np.sum(prediction == trues)/float(batch_size)
            test_batch_acc_total += current_accuracy

            # Count
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
        np.save('accuracy_ARM', test_batch_acc_total/test_batch_count)
        np.save('softmax_ARM', test_batch_softmax)
        np.save('entropy_ARM', test_batch_entropy)

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        
        tmpTestFeatures = []
    
        for feature in random_test_features:
            tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
            tmpTestFeatures.append(tmpFeature)
           
        random_test_predictions = sess.run(
            tf.nn.softmax(loaded_logits),
            feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels, first_pass: True})
        
        display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

test_model(tmpFeatures)

