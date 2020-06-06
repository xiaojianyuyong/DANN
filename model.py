import tensorflow as tf
from flip_gradient import flip_gradient
# Store layers weight & bias
# Network Parameters
n_input = 310 # data input
n_hidden_1 = 128 # feature extractor layer1
n_hidden_2 = 128 # feature extractor layer2
n_hidden_3 = 64 # label predictor layer1
n_hidden_4 = 64 # label predictor layer2
n_hidden_3_domain = 64 # domain discriminator layer1
n_hidden_4_domain = 64 # domain discriminator layer2
n_classes = 3 # total classes

feature_extractor_layer1_weight = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
feature_extractor_layer1_bias = tf.Variable(tf.random_normal([n_hidden_1]))
feature_extractor_layer2_weight = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
feature_extractor_layer2_bias = tf.Variable(tf.random_normal([n_hidden_2]))

classifer_layer1_weight = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
classifer_layer1_bias = tf.Variable(tf.random_normal([n_hidden_3]))
classifer_layer2_weight = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]))
classifer_layer2_bias = tf.Variable(tf.random_normal([n_hidden_4]))
classifer_output_weight = tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
classifer_output_bias = tf.Variable(tf.random_normal([n_classes]))

domain_classifer_layer1_weight = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3_domain]))
domain_classifer_layer1_bias = tf.Variable(tf.random_normal([n_hidden_3_domain]))
domain_classifer_layer2_weight = tf.Variable(tf.random_normal([n_hidden_3_domain, n_hidden_4_domain]))
domain_classifer_layer2_bias = tf.Variable(tf.random_normal([n_hidden_4_domain]))
domain_classifer_output_weight = tf.Variable(tf.random_normal([n_hidden_4_domain, 2]))
domain_classifer_output_bias = tf.Variable(tf.random_normal([2]))


# Create model
def MLP(x):
    # Hidden layer
    h1 = tf.add(tf.matmul(x, feature_extractor_layer1_weight), feature_extractor_layer1_bias)
    layer_1 = tf.sigmoid(h1)

    h2 = tf.add(tf.matmul(layer_1, feature_extractor_layer2_weight), feature_extractor_layer2_bias)
    layer_2 = tf.sigmoid(h2)    
    
    h3 = tf.add(tf.matmul(layer_2, classifer_layer1_weight), classifer_layer1_bias)
    layer_3 = tf.sigmoid(h3)

    h4 = tf.add(tf.matmul(layer_3, classifer_layer2_weight), classifer_layer2_bias)
    layer_4 = tf.sigmoid(h4)

    # Output layer
    output_layer = tf.add(tf.matmul(layer_4, classifer_output_weight), classifer_output_bias)
    return output_layer

def domain_classifier(x):
    # Hidden layer
    h1 = tf.add(tf.matmul(x, feature_extractor_layer1_weight), feature_extractor_layer1_bias)
    layer_1 = tf.sigmoid(h1)

    h2 = tf.add(tf.matmul(layer_1, feature_extractor_layer2_weight), feature_extractor_layer2_bias)
    layer_2 = tf.sigmoid(h2)

    layer_2_inverse = flip_gradient(layer_2)   
    
    h3 = tf.add(tf.matmul(layer_2_inverse, domain_classifer_layer1_weight), domain_classifer_layer1_bias)
    layer_3 = tf.sigmoid(h3)

    h4 = tf.add(tf.matmul(layer_3, domain_classifer_layer2_weight), domain_classifer_layer2_bias)
    layer_4 = tf.sigmoid(h4)

    # Output layer
    output_layer = tf.add(tf.matmul(layer_4, domain_classifer_output_weight), domain_classifer_output_bias)
    return output_layer
