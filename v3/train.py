from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from v3.model import LeNet5
import tensorflow as tf
from sklearn.utils import shuffle

BATCH_SIZE = 64


class Train:
    def __init__(self):
        self.CKPT_DIR = './ckpt'
        # Generate data set
        mnist = input_data.read_data_sets("../data/", reshape=False, one_hot=True)
        self.X_train, self.Y_train = mnist.train.images, mnist.train.labels
        self.X_validation, self.Y_validation = mnist.validation.images, mnist.validation.labels
        self.X_test, self.Y_test = mnist.test.images, mnist.test.labels

        print("X_train.shape: ", self.X_train.shape)
        print("X_validation.shape: ", self.X_validation.shape)
        print("X_test.shape: ", self.X_test.shape)

        self.X_train = np.pad(self.X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)
        self.X_validation = np.pad(self.X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)
        self.X_test = np.pad(self.X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=0)

        print("X_train.shape: ", self.X_train.shape)
        print("X_validation.shape: ", self.X_validation.shape)
        print("X_test.shape: ", self.X_test.shape)

        self.net = LeNet5(learning_rate=0.001)
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        epochs = 100
        num_examples = len(self.X_train)

        saver = tf.train.Saver(max_to_keep=5)
        save_interval = 10
        ckpt = tf.train.get_checkpoint_state(self.CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        for i in range(epochs):
            x_train, y_train = shuffle(self.X_train, self.Y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                x, y = x_train[offset:end], y_train[offset:end]
                self.sess.run(self.net.train, feed_dict={
                    self.net.x: x,
                    self.net.label: y,
                })
            if i % save_interval == 0:
                saver.save(self.sess, self.CKPT_DIR + '/model', global_step=i + 1)
            error_rate = self.evaluate(self.X_validation, self.Y_validation)
            print("EPOCH {} ...".format(i + 1))
            print('Validation error rate = {:.3f}\n'.format(error_rate))
        test_error_rate = self.evaluate(self.X_train, self.Y_test)
        print('\n\n-> Test error rate = {:.3f}\n'.format(test_error_rate))

    def evaluate(self, x_data, y_data):
        # sess = tf.get_default_session()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            error_rate = sess.run(self.net.error_rate, feed_dict={
                self.net.x: x_data,
                self.net.label: y_data,
            })
        return error_rate


if __name__ == '__main__':
    app = Train()
    app.train()
