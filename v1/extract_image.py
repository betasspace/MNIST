import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import os

def gen_image(arr, index, label):
    matrix = (np.reshape(1.0 - arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(matrix, 'L')
    img.save('../images/{}_{}.png'.format(label, index))


# print("pwd:", os.getcwd())
data = input_data.read_data_sets('../data/', one_hot=False)
x, y = data.train.next_batch(200)
for i, (arr, label) in enumerate(zip(x, y)):
    print(i, label)
    gen_image(arr, i, label)





