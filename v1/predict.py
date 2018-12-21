import numpy as np
from PIL import Image
from v1.model import Network
import tensorflow as tf
import os


class Predict:
    def __init__(self):
        self.CKPT_DIR = './ckpt'
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未找到保存的模型")

    def predict(self, image_path):
        # 读图片，归一化，并转为黑白的
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, 784) / 255.0
        x = np.array([1 - flatten_img])
        # print(x)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        print(image_path)
        print('     -> predict digit', np.argmax(y))


if __name__ == '__main__':
    print("pwd:", os.getcwd())
    app = Predict()
    app.predict('../images/0_11.png')
    app.predict('../images/1_8.png')
    app.predict('../images/2_1.png')
    app.predict('../images/3_169.png')
    app.predict('../images/4_35.png')
    app.predict('../images/4_94.png')
    app.predict('../images/5_15.png')

# ../images/0_11.png
#      -> predict digit 0
# ../images/1_8.png
#      -> predict digit 1
# ../images/2_1.png
#      -> predict digit 3
# ../images/3_169.png
#      -> predict digit 3
# ../images/4_35.png
#      -> predict digit 0
# ../images/5_15.png
#      -> predict digit 5
