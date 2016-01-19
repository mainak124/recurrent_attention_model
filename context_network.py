import numpy as np
import gzip
import os
import re
import sys
import tarfile
import tensorflow.python.platform
import tensorflow as tf
from create_dataset import get_dataset 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")

parameters = []

conv_counter = 1
fc_counter = 1
softmax_counter = 1

def _conv(X, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    #global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(X, kernel, [1, dH, dW, 1], padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32), trainable=True, name='biases')
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, conv1

def _fc(X, nIn, nOut):
    global fc_counter
    #global parameters
    name = 'fc' + str(fc_counter)
    fc_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32), trainable=True, name='biases')
        fc1 = tf.nn.relu_layer(X, kernel, biases, name=name)
        #fc1 = tf.nn.relu_layer(tf.matmul(X, kernel) + biases, name=name)
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, fc1

def _softmax(X, Y, nIn, nOut):
    global softmax_counter
    #global parameters
    name = 'softmax' + str(softmax_counter)
    softmax_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32), trainable=True, name='biases')
        softmax1 = tf.nn.softmax(tf.matmul(X, kernel) + biases, name=name)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(X, kernel) + biases, Y, name=name))
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, cost, softmax1

def context_net(x, y_, keep_prob, n_classes):

    global parameters
    x_ = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print "max pool out shape: ", x_.get_shape()

    params_conv1, conv1 = _conv(x_, 3, 10, 8, 8, 4, 4, 'SAME')
    #print "conv1 out shape: ", conv1.get_shape()
    params_conv2, conv2 = _conv(conv1,  10, 10, 6, 6, 2, 2, 'SAME')
    #print "conv2 out shape: ", conv2.get_shape()
    params_conv3, conv3 = _conv(conv2,  10, 20, 4, 4, 2, 2, 'SAME')
    #print "conv3 out shape: ", conv3.get_shape()
    flat1 = tf.reshape(conv3, [-1, 20 * 8 * 8])
    params_fc1, fc1 = _fc(flat1, 20 * 8 * 8, 128)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    params_softmax1, cost, out = _softmax(fc1_drop, y_, 128, n_classes)
    parameters = params_conv1 + params_conv2 + params_conv3 + params_fc1 + params_softmax1
    
    return cost, out

def train_model(n_epoch = 1000, batch_size = 50, n_classes = 1011, dataset_path = '/home/mainak/datasets/nabirds/birds_train_data_1k.pkl'):

    train_set_x, train_set_y_ = get_dataset(dataset_path)
    train_set_y_ = train_set_y_.astype(int)
    n_train_batches = train_set_x.shape[0]
    train_set_y = np.zeros((n_train_batches, n_classes))
    train_set_y[np.arange(n_train_batches), train_set_y_-1] = 1
    #train_set_y = dense_to_one_hot(train_set_y_, num_classes=1000)
    print "train_set_y shape: ", train_set_y.shape
    n_train_batches = np.ceil(n_train_batches/batch_size)
    n_train_batches = n_train_batches.astype(int)
    x = tf.placeholder(tf.float32, shape=(None,256,256,3))
    y_ = tf.placeholder(tf.float32, shape=(None,n_classes))
    keep_prob = tf.placeholder("float")
    
    cost, y_pred = context_net(x=x, y_=y_, keep_prob=keep_prob, n_classes=n_classes)
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_pred))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Launch the graph in a session.
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    with sess.as_default():
        for epoch in range(n_epoch):
            for minibatch_index in xrange(n_train_batches):
                batch = [train_set_x[minibatch_index*batch_size: (minibatch_index+1)*batch_size], 
                train_set_y[minibatch_index*batch_size: (minibatch_index+1)*batch_size]] 
                if minibatch_index%100 == 0:
                    #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    #train_cost = cost.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    train_accuracy, train_cost = sess.run(fetches=[accuracy, cost], feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, minibatch_index %d, train cost %g, training accuracy %g"%(epoch, minibatch_index, train_cost, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def main(_):
    train_model()
  
if __name__ == '__main__':
    tf.app.run()    
