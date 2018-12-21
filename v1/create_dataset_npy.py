import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

x, y = [], []
for i, image_path in enumerate(os.listdir('../images/')):
    label = int(image_path.split('_')[0])
    label_one_hot = [int(i == label) for i in range(10)]
    # print('label_one_hot: ', label_one_hot)
    y.append(label_one_hot)

    image = Image.open('../images/%s' % image_path).convert('L')
    image_arr = 1 - np.reshape(image, 784) / 255.0
    # print('image_arr:', image_arr)
    x.append(image_arr)

np.save('../manual_data_set/X.npy', np.array(x))
np.save('../manual_data_set/Y.npy', np.array(y))


class DataSet:
    def __init__(self):
        x, y = np.load('../manual_data_set/X.npy'), np.load('../manual_data_set/Y.npy')
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(x, y, test_size=0.2, random_state=0)

        self.train_size = len(self.train_x)

    def get_train_batch(self, batch_size=64):
        # size 连续多次随机，可重复
        choice = np.random.randint(self.train_size, size=batch_size)
        print(choice)
        batch_x = self.train_x[choice, :]
        batch_y = self.train_y[choice, :]

        return batch_x, batch_y

    def get_test_set(self):
        return self.test_x, self.test_y


data = DataSet()
x, y = data.get_train_batch()
print(x)
print(y)
