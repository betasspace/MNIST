import tensorflow as tf


class Network:
    def __init__(self):
        self.learning_rate = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784], name='x/input_layer')
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        with tf.device('/device:GPU:0'):
            self.w1 = tf.Variable(tf.random_uniform([784, 300], -1, 1), name='w/hidden_layer1')
            self.b1 = tf.Variable(tf.random_uniform([300], -1, 1), name='bias/hidden_layer1')
            self.h1 = tf.nn.sigmoid(tf.matmul(self.x, self.w1) + self.b1)

            self.w2 = tf.Variable(tf.random_uniform([300, 10], -1, 1), name='w/hidden_layer2')
            self.b2 = tf.Variable(tf.random_uniform([10], -1, 1), name='b/hidden_layer2')

            self.y = tf.nn.softmax(tf.matmul(self.h1, self.w2) + self.b2, name='output_layer')

            # # cross-entropy
            # self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))
            # MSE
            self.loss = tf.reduce_mean(tf.square(self.label - self.y))

            self.global_step = tf.Variable(0, trainable=False)
            self.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                      self.global_step)
            error = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
            self.error_rate = tf.reduce_mean(tf.cast(error, 'float'))


class NetworkLayer3:
    def __init__(self):
        self.learning_rate = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784], name='x/input_layer')
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        with tf.device('/device:GPU:0'):
            self.w1 = tf.Variable(tf.random_uniform([784, 500], -1, 1), name='w/hidden_layer1')
            self.b1 = tf.Variable(tf.random_uniform([500], -1, 1), name='bias/hidden_layer1')
            self.h1 = tf.nn.sigmoid(tf.matmul(self.x, self.w1) + self.b1)

            self.w2 = tf.Variable(tf.random_uniform([500, 150], -1, 1), name='w/hidden_layer2')
            self.b2 = tf.Variable(tf.random_uniform([150], -1, 1), name='b/hidden_layer2')
            self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)

            self.w3 = tf.Variable(tf.random_uniform([150, 10], -1, 1), name='w/hidden_layer3')
            self.b3 = tf.Variable(tf.random_uniform([10], -1, 1), name='b/hidden_layer3')

            self.y = tf.nn.softmax(tf.matmul(self.h2, self.w3) + self.b3, name='output_layer')

            # cross-entropy
            self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

            self.global_step = tf.Variable(0, trainable=False)
            self.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                      self.global_step)
            error = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
            self.error_rate = tf.reduce_mean(tf.cast(error, 'float'))


class NetworkLayer3TruncatedNormal:
    def __init__(self):
        self.learning_rate = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784], name='x/input_layer')
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        with tf.device('/device:GPU:0'):
            self.w1 = tf.Variable(tf.truncated_normal([784, 500], 0, 0.1), name='w/hidden_layer1')
            self.b1 = tf.Variable(tf.truncated_normal([500], 0, 0.1), name='bias/hidden_layer1')
            self.h1 = tf.nn.sigmoid(tf.matmul(self.x, self.w1) + self.b1)

            self.w2 = tf.Variable(tf.truncated_normal([500, 150], 0, 0.1), name='w/hidden_layer2')
            self.b2 = tf.Variable(tf.truncated_normal([150], 0, 0.1), name='b/hidden_layer2')
            self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)

            self.w3 = tf.Variable(tf.truncated_normal([150, 10], 0, 0.1), name='w/hidden_layer3')
            self.b3 = tf.Variable(tf.truncated_normal([10], 0, 0.1), name='b/hidden_layer3')

            self.y = tf.nn.softmax(tf.matmul(self.h2, self.w3) + self.b3, name='output_layer')

            # cross-entropy
            self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

            self.global_step = tf.Variable(0, trainable=False)
            self.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                      self.global_step)
            error = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
            self.error_rate = tf.reduce_mean(tf.cast(error, 'float'))


class NetworkLayer3WithReLU:
    def __init__(self):
        self.learning_rate = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784], name='x/input_layer')
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        with tf.device('/device:GPU:0'):
            self.w1 = tf.Variable(tf.truncated_normal([784, 500], 0, 0.1), name='w/hidden_layer1')
            self.b1 = tf.Variable(tf.truncated_normal([500], 0, 0.1), name='bias/hidden_layer1')
            self.h1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)

            self.w2 = tf.Variable(tf.truncated_normal([500, 150], 0, 0.1), name='w/hidden_layer2')
            self.b2 = tf.Variable(tf.truncated_normal([150], 0, 0,1), name='b/hidden_layer2')
            self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2) + self.b2)

            self.w3 = tf.Variable(tf.truncated_normal([150, 10], 0, 0.1), name='w/hidden_layer3')
            self.b3 = tf.Variable(tf.truncated_normal([10], 0, 0.1), name='b/hidden_layer3')

            self.y = tf.nn.softmax(tf.matmul(self.h2, self.w3) + self.b3, name='output_layer')

            # cross-entropy
            self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

            self.global_step = tf.Variable(0, trainable=False)
            self.train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                      self.global_step)
            error = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
            self.error_rate = tf.reduce_mean(tf.cast(error, 'float'))
