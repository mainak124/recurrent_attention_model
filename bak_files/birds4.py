import numpy as np
import gzip
import cv2
import os
import re
import sys
import tarfile
import tensorflow.python.platform
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from create_dataset import get_dataset, load_image_batch, get_data_length, find_image_mean 
from tensorflow.python.ops import attention_ops
from tensorflow.python.ops import control_flow_ops

global var_list
global var_list_2
global var_list_3
global baseline
global lstm_state_size
global lambda_reward
global n_sample
global stddev
global global_step
global l2_decay
global num_step
global n_classes
global batch_size
global dis_reward
global image_size
global context_net_type
global n_epoch
global lr
global glimpse_size

var_list = [] #parameters of rnn2, emission
var_list_2 = [] #paramters of rnn1, glimpse, classification
var_list_3 = [] 
rnn_dim = 512
context_net_type = "alex"
#context_net_type = "3-layer"
#image_size  = 227
image_size  = 227 if context_net_type == "alex" else 256
num_step = 5
lambda_reward = 0.7
n_sample = 10
n_classes = 555
stddev = 15
global_step = 0
#l2_decay = 0.0001
l2_decay = 0.0
batch_size = 64
dis_reward = 0.9
n_epoch = 100
lr = 0.1
glimpse_size = 32

#baseline = num_step*[0]

conv_counter = 1
pool_counter = 1
lrn_counter = 1
fc_counter = 1
dropout_counter = 1
softmax_counter = 1

# baseline = {0: tf.cast(0, dtype=tf.float32)}
# for idx in xrange(num_step-1):
#     baseline[idx+1] = tf.cast(0, dtype=tf.float32)

# baseline = {0: 0.}
# for idx in xrange(num_step-1):
#     baseline[idx+1] = 0.

# baseline = np.zeros([num_step], dtype=np.float32)

def _conv(X, nIn, nOut, ker, st, padType, bias_val=0.0):
    global conv_counter
    #global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[ker, ker, nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        conv = tf.nn.conv2d(X, kernel, [1, st, st, 1], padding=padType)
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, conv1

def _pool(X, ker, st, padType):
    global pool_counter
    #global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        return tf.nn.max_pool(X, ksize=[1, ker, ker, 1], strides=[1, st, st, 1], padding=padType, name=scope)

def _norm(l_input, lsize=5, alpha=0.0001, beta=0.75):
    global lrn_counter
    #global parameters
    name = 'lrn' + str(lrn_counter)
    lrn_counter += 1
    with tf.name_scope(name) as scope:
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=alpha, beta=beta, name=scope)        

def _fc(X, nIn, nOut, std, activation=True, bias_val = 0.0):
    global fc_counter
    #global parameters
    name = 'fc' + str(fc_counter)
    fc_counter += 1
    with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=std), shape=[nIn, nOut], dtype=tf.float32, trainable=True, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(bias_val), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        if (activation==True):
            fc1 = tf.nn.relu_layer(X, kernel, biases, name=name)
        else:    
            fc1 = tf.nn.bias_add(tf.matmul(X, kernel), biases, name=name)
        #fc1 = tf.nn.relu_layer(tf.matmul(X, kernel) + biases, name=name)
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, fc1

def _dropout(X, keep_prob, is_train):
    global dropout_counter
    #global parameters
    name = 'dropout' + str(dropout_counter)
    dropout_counter += 1
    #with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    with tf.name_scope(name) as scope:
        if is_train:
            return tf.nn.dropout(X, keep_prob)
        else:
            return X*keep_prob    

def _softmax(X, Y, nIn, nOut):
    global softmax_counter
    #global parameters
    name = 'softmax' + str(softmax_counter)
    softmax_counter += 1
    with tf.name_scope(name) as scope, tf.device('/gpu:1'):
    #with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=1e-2), shape=[nIn, nOut], trainable=True, dtype=tf.float32, name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[nOut], dtype=tf.float32, trainable=True, name='biases')
        softmax1 = tf.nn.softmax(tf.matmul(X, kernel) + biases, name=name)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(X, kernel) + biases, Y, name=name))
        #cost = -tf.reduce_sum(Y*tf.log(softmax1))
        #parameters += [kernel, biases]
        params = [kernel, biases]
        return params, cost, softmax1

def context_net(x, y_, keep_prob, is_train, n_classes):

    global parameters
    x_ = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print "max pool out shape: ", x_.get_shape()

    with tf.variable_scope("conv1"):
        params_conv1, conv1 = _conv(x_, nIn=3, nOut=10, ker=8, st=4, padType='SAME')
        # print "conv1 out shape: ", conv1.get_shape()
    with tf.variable_scope("conv2"):
        params_conv2, conv2 = _conv(conv1,  nIn=10, nOut=10, ker=6, st=2, padType='SAME')
        # print "conv2 out shape: ", conv2.get_shape()
    with tf.variable_scope("conv3"):
        params_conv3, conv3 = _conv(conv2, nIn=10, nOut=20, ker=4, st=2, padType='SAME')
        # print "conv3 out shape: ", conv3.get_shape()
    flat1 = tf.reshape(conv3, [-1, 20 * 8 * 8])
    with tf.variable_scope("fc4"):
        params_fc4, fc4 = _fc(flat1, 20 * 8 * 8, rnn_dim, std=0.005, bias_val = 0.1)
    fc4_drop = tf.nn.dropout(fc4, keep_prob)
    #params_softmax1, cost, out = _softmax(fc1_drop, y_, 128, n_classes)
    #parameters = params_conv1 + params_conv2 + params_conv3 + params_fc1 + params_softmax1
    params = params_conv1 + params_conv2 + params_conv3 + params_fc4
    global var_list_2
    var_list_2 += params
    
    return fc4_drop

def context_net_alex(x, y_, keep_prob, is_train, n_classes):

    global parameters
    global rnn_dim
    #x_ = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print "max pool out shape: ", x_.get_shape()

    #with tf.device('/gpu:1'):
    with tf.variable_scope("conv1"):
        params_conv1, conv1 = _conv(x, nIn=3, nOut=96, ker=11, st=4, padType='VALID')
        # print "conv1 out shape: ", conv1.get_shape()
    lrn1 = _norm(conv1, lsize=5, alpha=0.0001, beta=0.75)
    pool1 = _pool(lrn1, ker=3, st=2, padType='VALID')
    # print "pool1 out shape: ", pool1.get_shape()

    with tf.variable_scope("conv2"):
        params_conv2, conv2 = _conv(pool1, nIn=96, nOut=256, ker=5, st=1, padType='SAME', bias_val = 0.1)
        # print "conv2 out shape: ", conv2.get_shape()
    lrn2 = _norm(conv2, lsize=5, alpha=0.0001, beta=0.75)
    pool2 = _pool(lrn2, ker=3, st=2, padType='VALID')
    # print "pool2 out shape: ", pool2.get_shape()

    with tf.variable_scope("conv3"):
        params_conv3, conv3 = _conv(pool2, nIn=256, nOut=384, ker=3, st=1, padType='SAME')
    # print "conv3 out shape: ", conv3.get_shape()
    with tf.variable_scope("conv4"):
        params_conv4, conv4 = _conv(conv3, nIn=384, nOut=384, ker=3, st=1, padType='SAME', bias_val = 0.1)
    # print "conv4 out shape: ", conv4.get_shape()
    with tf.variable_scope("conv5"):
        params_conv5, conv5 = _conv(conv4, nIn=384, nOut=256, ker=3, st=1, padType='SAME', bias_val = 0.1)
    # print "conv5 out shape: ", conv5.get_shape()
    pool5 = _pool(conv5, ker=3, st=2, padType='VALID')
    # print "pool5 out shape: ", pool5.get_shape()

    flat1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    with tf.variable_scope("fc6"):
        params_fc6, fc6 = _fc(flat1, 256 * 6 * 6, 4096, std=0.005, bias_val = 0.1)
    # fc6_drop = tf.nn.dropout(fc6, keep_prob)
    fc6_drop = _dropout(fc6, keep_prob, is_train)
    with tf.variable_scope("fc7"):
        params_fc7, fc7 = _fc(fc6_drop, 4096, 4096, std=0.005, bias_val = 0.1)
    # fc7_drop = tf.nn.dropout(fc7, keep_prob)
    fc7_drop = _dropout(fc7, keep_prob, is_train)
    with tf.variable_scope("fc8"):
        params_fc8, fc8 = _fc(fc7_drop, 4096, rnn_dim, std=0.005)
    #fc8_drop = tf.nn.dropout(fc8, keep_prob)
    fc8_drop = _dropout(fc8, keep_prob, is_train)
    with tf.variable_scope("sm9"):
        params_softmax9, cost, out = _softmax(fc8_drop, y_, rnn_dim, n_classes)
    # return cost, out
    #parameters = params_conv1 + params_conv2 + params_conv3 + params_fc1
    params = params_conv1 + params_conv2 + params_conv3 + params_conv4 + params_conv5 + params_fc6 + params_fc7 + params_fc8 + params_softmax9
    global var_list_2
    var_list_2 += params
    # print fc8_drop
    return fc8_drop

def glimpse_net(x, locs, keep_prob, is_train, sample_idx): #, glimpse_size=16):
    
    #add random noise around the image x to make it of size 256+glimpse_size*2
    #crop two resolutions patches and stack together
    global n_sample
    global rnn_dim
    
    # print "locs Shape: ", locs.get_shape()#.get_shape()
    loc = tf.reshape(locs[:, sample_idx, :], [-1, 2]) #FIXME: Wait, no slicing is required? !!!
    # loc = tf.reshape(tf.pack([tf.gather(tf.reshape(locs, [-1]), i * tf.shape(locs)[1]), tf.gather(tf.reshape(locs, [-1]), i * tf.shape(locs)[1]+1)]), [1,2])

    x1 = tf.image.extract_glimpse(x, size=[glimpse_size, glimpse_size], offsets=loc)
    im_ex = tf.image.extract_glimpse(x, size=[2*glimpse_size, 2*glimpse_size], offsets=loc)
    x2 = tf.nn.max_pool(im_ex, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    patch = tf.concat(3,[x1,x2])
    #patch = x1

    with tf.variable_scope("conv1"):
        params_conv1, conv1 = _conv(patch, nIn=6, nOut=16, ker=6, st=2, padType='VALID', bias_val = 0.1)
    # print "Glimpse conv1 out shape: ", conv1.get_shape()
    with tf.variable_scope("conv2"):
        params_conv2, conv2 = _conv(conv1, nIn=16, nOut=16, ker=3, st=1, padType='VALID', bias_val = 0.1)
    # print "Glimpse conv2 out shape: ", conv2.get_shape()
    with tf.variable_scope("conv3"):
        params_conv3, conv3 = _conv(conv2,nIn=16, nOut=16, ker=3, st=1, padType='VALID',bias_val = 0.1)
    # print "Glimpse conv3 out shape: ", conv3.get_shape()
    flat1 = tf.reshape(conv3, [-1, 16 * 10 * 10])
    with tf.variable_scope("fc_image"):
        params_fc_image, fc_image = _fc(flat1, 16 * 10 * 10, rnn_dim, std=0.005, bias_val=0.1)
    with tf.variable_scope("fc_loc"):
        params_fc_loc, fc_loc = _fc(loc, 2, rnn_dim, std=0.005, bias_val=0.1)
    
    global var_list_2
    params_glimpse = params_conv1 + params_conv2 + params_conv3 + params_fc_image + params_fc_loc
    var_list_2 += params_glimpse
    return tf.mul(fc_image,fc_loc)
    
def rnn_net(x, cell=None, layer=None, prev_state=None, is_train=1, keep_prob=0.8):

    global lstm_state_size
    global rnn_dim

    if prev_state==None:   
        lstm_cell = rnn_cell.BasicLSTMCell(rnn_dim, forget_bias=0.2)
        if is_train and keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        else:    
            lstm_cell *= keep_prob
    else:
        lstm_cell = cell        

    #batch_dim = n_sample if layer is not None else 1    
    batch_dim = batch_size

    if prev_state==None:   
        _initial_state = lstm_cell.zero_state(batch_dim, tf.float32)
    else:    
        _initial_state = prev_state

    lstm_state_size = lstm_cell.state_size
    outputs = []
    states=[]
    state = _initial_state
    lstm_params=[]
    name = "lstm"+str(layer)
    with tf.variable_scope(name) as vs, tf.device('/gpu:1'):
        #tf.get_variable_scope().reuse_variables()
        (cell_output,state) = lstm_cell(x, state) 
        #outputs.append(cell_output)
        #states.append(state)
        lstm_params = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    #output = tf.reshape(tf.concat(1, outputs), [-1, rnn_dim])
    output = cell_output
    if layer == 2:
        global var_list
        var_list += lstm_params
    elif layer==1:
        global var_list_2
        var_list_2 += lstm_params
    return output, state, lstm_cell

def emission_net(x, keep_params=1):
    global rnn_dim
    with tf.variable_scope("fc_emission"):
        params_emission, output = _fc(x, rnn_dim, 2, std=0.005, activation=False, bias_val = 0.1)
    global var_list
    if keep_params == 1:
        var_list += params_emission
    else:
        global var_list_3
        var_list_3 += params_emission
    #converting coordinates to pixel levels
    #output = tf.floor(out*ratio*image_size/2)+image_size/2
    #output = tf.floor(tf.sigmoid(output)*image_size) # FIXME: tanh or sigmoid?
    output = tf.floor((tf.tanh(output)*(image_size/2)) + (image_size/2))
    return output

def sample_location(n_sample, mean):
    for i in range(n_sample):
        loc = tf.expand_dims(tf.random_normal([batch_size, 2], mean=mean, stddev=stddev, dtype=tf.float32), 1)
        # print "loc shape: ", loc.get_shape()
        locs = loc if i==0 else tf.concat(1, [locs, loc])
        # print "LOCS shape: ", locs.get_shape()
    return locs
    #return tf.random_normal([batch_size, n_sample, 2], mean=mean, stddev=stddev, dtype=tf.float32)
    
def classify_net(x, y_, n_classes):    
    global rnn_dim
    name = 'class_net'
    with tf.name_scope(name) as scope:
        with tf.variable_scope("fc_class"):
            params_fc_out, fc_out = _fc(x, rnn_dim, 1000, std=0.01)
        with tf.variable_scope("sm_class"):
            params_softmax_out, cost, out = _softmax(fc_out, y_, 1000, n_classes)
    #correct = tf.cast(label==y_, tf.float32)
    params_classify = params_fc_out + params_softmax_out
    global var_list_2
    var_list_2 += params_classify

    p_correct = []
    j = tf.cast(tf.argmax(y_, 1), dtype=tf.int32)
    for i in xrange(batch_size):
        p_correct.append(tf.gather(tf.reshape(out, [-1]), i * tf.shape(out)[1] + j))
    p_correct = tf.pack(p_correct)    

    return cost, out, p_correct

def onehot_to_integer(y, n_classes):
    for i in xrange(n_classes):
        if (y[i] == 1):
            return i
    return -1        

def build_net(x, y_, keep_prob, optimizer, learning_rate, batch_cnt, sess):

    rnn1_states = [] 
    rnn2_states = [] 
    rnn1_cells = [] 
    rnn2_cells = [] 
    locs_mean_list = []
    class_reward = []
    train_ops = []
    global baseline
    global reward
    baseline = num_step*[0]
    reward = 0
    tot_reward = 0
    pred_list = []
    with tf.variable_scope("ram_net") as ram_scope:
        for glimpse_idx in xrange(num_step):
            grads_0 = []
            grads_1a = []
            grads_1b_prod = []
            if glimpse_idx > 0: 
                tf.get_variable_scope().reuse_variables()
                dependence = [train_ops[glimpse_idx - 1]]
            if glimpse_idx == 0:
                #with tf.variable_scope("context_net") as context_scope, tf.device('/gpu:1'):
                with tf.variable_scope("context_net") as context_scope:
                    if context_net_type == "alex":
                        context_out = context_net_alex(x=x, y_=y_, keep_prob=keep_prob, n_classes=n_classes, is_train=1)
                        saver.restore(sess, "models/model_130.ckpt")
                    elif context_net_type == "3-layer":    
                        context_out = context_net(x=x, y_=y_, keep_prob=keep_prob, n_classes=n_classes, is_train=1)
                with tf.variable_scope("rnn2_net") as context_scope:
                    rnn2_init_out, rnn2_init_state, rnn2_init_cell = rnn_net(x=context_out, layer=2, is_train=1, keep_prob=0.8)
                with tf.variable_scope("emission_net") as context_scope:
                    emission_out = emission_net(x=rnn2_init_out, keep_params=0)
                dependence = None
            with tf.control_dependencies(dependence):  
                locs_mean = emission_out if glimpse_idx == 0 else tf.floor(tot_emission_out/n_sample)
                # print "LOCS MEAN: ", locs_mean.get_shape()
                locs_mean_list.append(locs_mean)
                locs = sample_location(n_sample, locs_mean) # batch_size x n_sample x 2
                #print "SAMPLE LOCS SHAPE: ", locs.get_shape()
                stop_locs_mean = tf.stop_gradient(locs_mean) # locs_mean will not contribute to the location gradient computation
                #avg_locs_cost = locs_cost/batch_size # location cost average over batch size
                v_er2 = [vv for vv in tf.trainable_variables() if "emission_net" in vv.name or "rnn2_net" in vv.name]
            
 #               locs_grads.append(locs_g)

                for sample_idx in xrange(n_sample):
                    with tf.variable_scope("glimpse_net") as glimpse_scope:
                        if sample_idx > 0: tf.get_variable_scope().reuse_variables()
                        glimpse_out = glimpse_net(x, locs, keep_prob, sample_idx=sample_idx, is_train=1) #, glimpse_size=16)

                    with tf.variable_scope("rnn1_net") as rnn1_scope:
                        if sample_idx > 0: tf.get_variable_scope().reuse_variables()
                        if glimpse_idx == 0:
                            rnn1_state = None # tf.zeros([batch_size, lstm_state_size], dtype=tf.float32) 
                            rnn1_cell = None
                        else:
                            rnn1_state = rnn1_prev_states[sample_idx] 
                            rnn1_cell = rnn1_prev_cells[sample_idx]
                        rnn1_out, rnn1_new_state, rnn1_new_cell = rnn_net(x=glimpse_out, cell=rnn1_cell, layer=1, prev_state=rnn1_state, is_train=1, keep_prob=0.8)
                        if glimpse_idx == 0:
                            rnn1_states.append(rnn1_new_state)
                            rnn1_cells.append(rnn1_new_cell)
                        else:    
                            rnn1_states[sample_idx] = rnn1_new_state
                            rnn1_cells[sample_idx] = rnn1_new_cell

                    with tf.variable_scope("classify_net") as classify_scope:
                        if sample_idx > 0: tf.get_variable_scope().reuse_variables()
                        class_cost, y_pred, p_correct = classify_net(rnn1_out, y_, n_classes)
                        tot_p_correct = tf.log(p_correct) if sample_idx==0 else tot_p_correct+tf.log(p_correct)
                        tot_class_cost = class_cost if sample_idx==0 else tot_class_cost+class_cost

                        pred = tf.argmax(y_pred,1)
                        label = tf.argmax(y_,1)
                        correct = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
                        #print "CORRECT SHAPE: ", correct.get_shape()
                        tot_correct = tf.expand_dims(tf.cast(correct, "float"),1) if sample_idx == 0 else tf.concat(1, [tot_correct, tf.expand_dims(tf.cast(correct, "float"),1)])
                        #print "TOTAL CORRECT SHAPE: ", tot_correct.get_shape()
                        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
                        tot_accuracy = accuracy if sample_idx==0 else tot_accuracy+accuracy
                        # if (glimpse_idx==num_step-1):
                        #     pred = tf.argmax(y_pred,1)
                        #     label = tf.argmax(y_,1)
                        #     correct = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
                        #     accuracy = tf.reduce_mean(tf.cast(correct, "float"))
                        #     tot_accuracy = accuracy if sample_idx==0 else tot_accuracy+accuracy

                    if (glimpse_idx<num_step-1):
                        with tf.variable_scope("rnn2_net") as rnn2_scope:
                            tf.get_variable_scope().reuse_variables()
                            if glimpse_idx == 0:
                                rnn2_state = rnn2_init_state # tf.zeros([batch_size, lstm_state_size], dtype=tf.float32) 
                                rnn2_cell = rnn2_init_cell
                            else:
                                rnn2_state = rnn2_prev_states[sample_idx] 
                                rnn2_cell = rnn2_prev_cells[sample_idx] 
                            rnn2_out, rnn2_new_state, rnn2_new_cell = rnn_net(x=rnn1_out, layer=2, cell=rnn2_cell, prev_state=rnn2_state, is_train=1, keep_prob=0.8)
                            if glimpse_idx == 0:
                                rnn2_states.append(rnn2_new_state)
                                rnn2_cells.append(rnn2_new_cell)
                            else:    
                                rnn2_states[sample_idx] = rnn2_new_state
                                rnn2_cells[sample_idx] = rnn2_new_cell

                        with tf.variable_scope("emission_net") as emission_scope:
                            tf.get_variable_scope().reuse_variables()
                            emission_out = emission_net(x=rnn2_out, keep_params=0)  
                            tot_emission_out = emission_out if sample_idx==0 else tot_emission_out+emission_out
                    if (glimpse_idx == num_step-1):        
                        pred_list.append(pred)
                #print "TOTAL CORRECT SHAPE: ", tot_correct.get_shape()            
                stop_tot_correct = tf.stop_gradient(tot_correct)
                locs_cost = get_locs_cost(locs=locs, mean=stop_locs_mean, reward=stop_tot_correct, base=baseline[glimpse_idx]) # summed up cost
                locs_g = tf.gradients(locs_cost, v_er2)
                #reward = tf.reduce_sum(tot_p_correct) / (n_sample*batch_size)        
                reward = tf.reduce_sum(tot_correct) / (n_sample*batch_size)        
                # tot_reward = dis_reward * tot_reward + (reward-baseline[glimpse_idx])
                # inst_reward = reward-baseline[glimpse_idx]
                class_reward.append(reward) # FIXME: total (discounnted) reward till current glimpse only the reward at current glimpse?
                baseline[glimpse_idx] = 0.9 * baseline[glimpse_idx] + 0.1 * reward
                rnn1_prev_states = rnn1_states
                rnn2_prev_states = rnn2_states
                rnn1_prev_cells = rnn1_cells
                rnn2_prev_cells = rnn2_cells
                avg_class_cost = tf.reduce_sum(tot_class_cost)/(n_sample*batch_size) #classification cost for each glimpse
                v_others = [vo for vo in tf.trainable_variables() if "emission_net" not in vo.name and "rnn2_net" not in vo.name]
                allv = v_others + v_er2
                l2_cost = l2_decay * tf.add_n([tf.nn.l2_loss(var) for var in allv])    
                avg_class_cost = tf.add(avg_class_cost, l2_cost)
                class_g_1 = tf.gradients(avg_class_cost, v_others)
                class_g_2 = tf.gradients(avg_class_cost, v_er2)
                for grad in class_g_1:
                    grads_0.append(grad)
                for grad in class_g_2:
                    grads_1a.append(grad)
                for grad in locs_g:
                    grads_1b_prod.append(lambda_reward * grad)
                grads_1 = [p+q for p,q in zip(grads_1a, grads_1b_prod)]
                allg = grads_0 + grads_1
                train_ops.append(optimizer.apply_gradients(zip(allg, allv), global_step=batch_cnt)) # FIXME: batch_cnt?
#                class_grads_1.append(class_g_1)
#                class_grads_2.append(class_g_2)

    avg_accuracy = tf.reduce_sum(tot_accuracy)/n_sample
    locs_mean_arr = tf.pack(locs_mean_list)
    class_reward_arr = tf.pack(class_reward)
    pred_arr = tf.pack(pred_list)
    return avg_class_cost, locs_cost, class_reward_arr, avg_accuracy, locs_mean_arr, train_ops, pred_arr, grads_1b_prod, baseline

def get_locs_cost(locs, mean, reward, base):
    mean_d = tf.tile(tf.expand_dims(mean, 1), [1, n_sample, 1])
    tiled_base = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(base, 0), 0), [batch_size, n_sample]), tf.float32)
    #print "TILED BASE SHAPE: ", tiled_base.get_shape(), tiled_base.dtype
    red_reward = reward - tiled_base
    reward_cost_arr = tf.mul(tf.reduce_sum(0.5 * tf.square((locs-mean_d)/stddev), 2), red_reward)
    return tf.div(tf.reduce_sum(reward_cost_arr), (n_sample * batch_size))

def print_config():
    print "------------RUN Configuration--------------"
    print "1.  Context Net: ", context_net_type
    print "2.  RNN Size: ", rnn_dim
    print "3.  Glimpse: ", num_step
    print "4.  Sample: ", n_sample
    print "5.  Standard Deviation: ", stddev
    print "6.  Initial Learrning Rate: ", lr
    print "7.  L2 Decay Rate: ", l2_decay
    print "8.  Lambda_reward: ", lambda_reward
    print "9.  Batch Size: ", batch_size
    print "10. Epoch: ", n_epoch
    print "-------------------------------------------"

def train_net(dataset_path = '/raid/mainak/datasets/nabirds/', image_path = 'images',n_classes=555): 

    global lstm_state_size
    global baseline

    global lambda_reward
    global n_sample
    global stddev
    global rnn_dim
    global saver

    print_config()

    lstm_state_size = rnn_dim*2

    #im_mean = find_image_mean(image_path) 
    #np.save('im_mean_256.npy', im_mean)
    #im_mean = np.load('im_mean_256.npy') 
    if context_net_type == "alex":
        im_mean = np.load('im_mean.npy') 
    elif context_net_type == "3-layer":    
        im_mean = np.load('im_mean_256.npy') 

    n_train_data, n_test_data = get_data_length(dataset_path)
    n_train_batches = np.ceil(n_train_data/batch_size)
    n_train_batches = n_train_batches.astype(int)

    x = tf.placeholder(tf.float32, shape=(None,image_size,image_size,3))
    y_ = tf.placeholder(tf.float32, shape=(None,n_classes))
    keep_prob = tf.placeholder("float")
    is_train = tf.placeholder(tf.int32)
    batch_cnt = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(lr, global_step=batch_cnt, decay_steps=5000, decay_rate=0.7, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.98)
    saver = tf.train.Saver()
    sess = tf.Session()
    class_cost, locs_cost, reward, accuracy, loc_mean, train_ops, preds, grads_1b, baseline_ = build_net(x, y_, keep_prob, optimizer, learning_rate, batch_cnt, sess)
    baseline_arr = tf.pack(baseline_)


    # print len(train_ops)
    # print train_ops[-1]

    # Commented Selectively Initializing Variables Part
    # init_vars = [iv for iv in tf.all_variables() if "context_net" not in iv.name]
    # sess.run(tf.initialize_variables(init_vars))
    # uninitialized_vars = []
    # for var in tf.all_variables():
    #     try:
    #         sess.run(var)
    #     except tf.errors.FailedPreconditionError:
    #         uninitialized_vars.append(var)
    # print "\n\n\n\n\n\n\n\n\n\n\n"        
    # for var in uninitialized_vars:        
    #     print "Uninitialized Variables: ", var.name
    # print "\n\n\n\n\n\n\n\n\n\n\n"        
    # saver.restore(sess, "models/model_130.ckpt")

    sess.run(tf.initialize_all_variables()) 
    #tf.assert_variables_initialized(tf.trainable_variables()).eval(session=sess)

    #with sess.as_default():
    with sess.as_default(), tf.device('/gpu:1'):
        for epoch in range(n_epoch):
            batch_cost = []
            batch_accuracy = []
            for minibatch_index in xrange(n_train_batches):
                #print "LR: ", learning_rate.eval()
                batch_x, batch_y_ = load_image_batch(im_mean = im_mean, dataset_path = dataset_path, image_path = image_path, start_idx=(minibatch_index*batch_size), batch_size = batch_size, is_train=is_train, im_size=image_size)
                #print "Batch: ", type(int(batch_y_[0]))
                batch_y = np.zeros((batch_size, n_classes))
                batch_y[np.arange(batch_size), np.int32(batch_y_)-1] = 1
                batch = [batch_x, batch_y]
                feed_dict={x:batch[0], y_:batch[1],is_train:0, keep_prob:0.8}

                class_cost_v, locs_cost_v, reward_v, accuracy_v, loc_mean_v, train_ops_v, preds_v, g1b_v, baseline_v = sess.run(fetches=[class_cost, locs_cost, reward, accuracy, loc_mean, train_ops[-1], preds, grads_1b[0], baseline_arr], feed_dict=feed_dict)
                print ("Epoch: %d, minibatch_idx: %d, class cost: %g, locs_cost: %g, accuracy: %g"%(epoch, minibatch_index, class_cost_v, locs_cost_v, accuracy_v))
                print "Reward: ", np.array_str(reward_v)
                print "Locations: ", np.array_str(loc_mean_v[:,1,:])
                # print "Grads: ", g1b_v
                # print "Baseline: ", baseline_v
                # print "Predicted Classes: ", preds_v
            batch_cost.append(class_cost_v)    
            batch_accuracy.append(accuracy_v)    
            print("Epoch: %d, Train Bach Cost: %g, Train Batch Accuracy: %g"%(epoch, np.mean(batch_cost), np.mean(batch_accuracy)))

            #print("Train Bach Cost: %g, Train Batch Accuracy: %g"%(np.mean(batch_cost), np.mean(batch_accuracy)))
            #with sess.as_default(), tf.device('/cpu:0'):
            #run_test_batch(session=sess, im_mean = im_mean, dataset_path=dataset_path, image_path=image_path, n_test_data=n_test_data, batch_size=batch_size, n_classes=n_classes, accuracy=accuracy, total_cost=total_cost, l2_cost=l2_cost, x=x, y_=y_, keep_prob=keep_prob, is_train=is_train)
            #if (epoch%10 == 0 and epoch>0):
            #    # Save the variables to disk.
            #    save_path = saver.save(sess, "/raid/mainak/Recurrent-Attention-Model/models/model_"+ str(epoch) +".ckpt")
            #    print("Model saved in file: %s" % save_path)

def main(_):
    train_net()
if __name__ == '__main__':
    tf.app.run()
