import random

import numpy as np
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, parameters):
        self.parameters = parameters
        self.network_name = 'qnet'
        self.session = tf.compat.v1.Session()
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder('float', [None, parameters['width'], parameters['height'], 6], name=self.network_name + '_x')
        self.q_t = tf.compat.v1.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.compat.v1.placeholder('float', [None, 4], name = self.network_name + '_actions')
        self.rewards = tf.compat.v1.placeholder('float', [None], name=self.network_name + '_rewards')
        self.terminals = tf.compat.v1.placeholder('float', [None], name=self.network_name+'_terminals')

        #  Convolutional Layer
        layer_name = 'conv_1'; size = 3; channels = 6; filters = 16; stride = 1
        self.weights1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name+"_"+layer_name+"_weights")
        self.biases1 = tf.compat.v1.Variable(tf.constant(0.2, shape=[filters]), name=self.network_name+"_"+layer_name+"_biases")
        self.convs1 = tf.compat.v1.nn.conv2d(self.x, self.weights1, strides=[1, stride, stride, 1], padding='SAME', name=self.network_name+"_" + layer_name +"_convulsions")
        self.relu1 = tf.compat.v1.nn.relu(tf.add(self.convs1, self.biases1), name=self.network_name + "_" + layer_name + "_relu_activations")

        rulu_shape = self.relu1.get_shape().as_list()

        # Fully Connected Layer
        layer = 'full_connect2'; hidden_layers = 256; dim = rulu_shape[1]*rulu_shape[2]*rulu_shape[3]
        self.ru_flat = tf.compat.v1.reshape(self.relu1, [-1, dim], name=self.network_name + "_" + layer_name + "_input_flat")
        self.weights2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([dim, hidden_layers], stddev=0.01), name=self.network_name+"_"+layer_name+"_weights")
        self.biases2 = tf.compat.v1.Variable(tf.constant(0.1, shape=[hidden_layers]), name=self.network_name+"_"+layer_name + "_biases")
        self.ip2 = tf.compat.v1.add(tf.matmul(self.ru_flat, self.weights2), self.biases2, name=self.network_name+"_"+layer_name+"_ips")
        self.relu2 = tf.compat.v1.nn.relu(self.ip2, name=self.network_name+"_"+layer_name+"_relu_activations")

        # Fully Connected Exit Layer
        layer_name = 'full_connect_exit'; hidden_layers = 4; dim = 256
        self.weights3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([dim, hidden_layers], stddev=0.01), name=self.network_name + "_" + layer_name + "_weights")
        self.biases3 = tf.compat.v1.Variable(tf.constant(0.1, shape=[hidden_layers], name=self.network_name+"_"+layer_name+"_biases"))
        self.y = tf.compat.v1.add(tf.compat.v1.matmul(self.relu2, self.weights3), self.biases3, name=self.network_name+"_"+layer_name+"_outputs")

        # Q, Cost, Optimizer
        self.discount = tf.compat.v1.constant(self.parameters['discount'])
        self.yj = tf.compat.v1.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_prediction = tf.compat.v1.reduce_sum(tf.multiply(self.y, self.actions), reduction_indices=1)
        self.cost = tf.compat.v1.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_prediction), 2))

        if self.parameters['load_file'] is not None:
            self.global_step = tf.compat.v1.Variable(int(self.parameters['load_file'].split('_')[-1]), name="global_step", trainable=False)
        else:
            self.global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)

        # Optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.parameters['learning_rate']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

        self.session.run(tf.compat.v1.global_variables_initializer())

        if self.parameters['load_file'] is not None:
            print("Loading...")
            self.saver.restore(self.session, self.parameters['load_file'])


    def train(self, batch_s, batch_a, batch_t, batch_n, batch_r):
        feed_dict={self.x: batch_s, self.q_t: np.zeros(batch_n.shape[0]), self.actions: batch_a, self.terminals: batch_t, self.rewards: batch_r}
        q_t = self.session.run(self.y, feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: batch_s, self.q_t: q_t, self.actions: batch_a, self.terminals: batch_t, self.rewards: batch_r}
        _,step,cost = self.session.run([self.optimizer, self.global_step, self.cost], feed_dict=feed_dict)
        return step, cost

    def save_ckpt(self, filename):
        self.saver.save(self.session, filename)

