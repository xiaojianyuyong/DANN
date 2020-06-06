import tensorflow as tf
import model
import scipy.io as scio
import numpy as np
import time
import pickle
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Parameters
n_classes = 3
n_input = 310
learning_rate = 0.001
training_epochs = 10000
batch_size = 2000
display_step = 100
MODE = sys.argv[1]
sub_dataset = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4']
sub_accuracy = {'sub_0': 0.0, 'sub_1': 0.0, 'sub_2': 0.0, 'sub_3': 0.0, 'sub_4': 0.0}
mean_accuracy = 0.0

if MODE == 'DANN':
    r = 1
else:
    r = 0

# Load data
def load_data(sub_id):
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    for item in list(data.keys()):
        if(item != sub_id):
            train_feature.extend(data[item]['data'])
            train_label.extend(data[item]['label'])
        else:
            test_feature.extend(data[item]['data'])
            test_label.extend(data[item]['label'])

    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    test_feature = np.array(test_feature)
    test_label = np.array(test_label)
    
    return train_feature, train_label, test_feature, test_label

def load_domain_data(sub_id):
    domain_data = []
    domain_label = []
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    for item in list(data.keys()):
        domain_data.extend(data[item]['data'])
        if (item == sub_id):
            label = [[1,0]]*3397
        else:
            label = [[0,1]]*3397
        
        domain_label.extend(label)
  
    domain_data = np.array(domain_data)
    domain_label = np.array(domain_label)

    return domain_data, domain_label

# Convert label to one-hot vector
def one_hot_label(input_label):
    one_hot = np.empty([input_label.shape[0],n_classes])
    for k,v in enumerate(input_label):
        if(v == -1):
            one_hot[k] = [1,0,0]
        elif (v == 0):
            one_hot[k] = [0,1,0]
        elif (v == 1):
            one_hot[k] = [0,0,1]
    return one_hot

# Define a dataset class, for get-batch in training
class Dataset(object):
    def __init__(self,x,y):
        one_hot = np.empty([y.shape[0],n_classes])
        for k,v in enumerate(y):
            if(v == -1):
                one_hot[k] = [1,0,0]
            elif (v == 0):
                one_hot[k] = [0,1,0]
            elif (v == 1):
                one_hot[k] = [0,0,1]
        self.x = x
        self.y = one_hot
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    def get_next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if(self._index_in_epoch > self._num_examples): 
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.x = self.x[perm]
            self.y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.x[start:end], self.y[start:end]

class domain_Dataset(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    def get_next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if(self._index_in_epoch > self._num_examples): 
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.x = self.x[perm]
            self.y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.x[start:end], self.y[start:end]

# Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

X_domain = tf.placeholder("float", [None, n_input])
Y_domain = tf.placeholder("float", [None, 2])

# Construct model
logits = model.MLP(X)
domain_logits = model.domain_classifier(X_domain)

# Define loss and optimizer
predict_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
domain_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=domain_logits, labels=Y_domain))
total_loss = predict_loss+r*domain_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(total_loss)

# Initializing the variables
for sub in sub_dataset:
    train_features, train_labels, test_features, test_labels = load_data(sub)
    domain_features, domain_labels = load_domain_data(sub)
    dataset = Dataset(train_features, train_labels)
    domain_dataset = domain_Dataset(domain_features, domain_labels)
    test_one_hot = one_hot_label(test_labels)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        time0 = time.time()
        # Training
        for epoch in range(training_epochs):
            avg_loss_d = 0.0
            avg_loss_p = 0.0
            avg_loss_t = 0.0
            total_batch = int(dataset._num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.get_next_batch(batch_size)
                domain_batch_x, domain_batch_y = domain_dataset.get_next_batch(batch_size)
                _, c_p, c_d, c_t = sess.run([train_op, predict_loss, domain_loss, total_loss],feed_dict={X: batch_x,
                                                                                                         Y: batch_y,
                                                                                           X_domain: domain_batch_x,
                                                                                           Y_domain: domain_batch_y})
                # Compute average loss
                avg_loss_p += c_p / total_batch
                avg_loss_d += c_d / total_batch
                avg_loss_t += c_t / total_batch
            # Display logs
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "p_loss={:.5f} ".format(avg_loss_p), "d_loss={:.5f} ".format(avg_loss_d), "t_loss={:.5f} ".format(avg_loss_t))
        time1 = time.time()-time0
        print("Optimization Finished! Total time: %s"%(time1))

        # Test model
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sub_accuracy[sub] = accuracy.eval({X: test_features, Y: test_one_hot})
        mean_accuracy = mean_accuracy+sub_accuracy[sub]

for sub in sub_dataset:
    print(sub, " accuracy: ", sub_accuracy[sub])
print("mean accuracy: ", mean_accuracy/5)