import os
import numpy as np
import h5py
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

    # HDF5 group: 包含0个或多个HDF5对象以及支持元数据（metadata）的一个群组结构。
    # HDF5 dataset: 数据元素的一个多维数组以及支持元数据（metadata）
    with h5py.File('../manual_data_set/data.h5', 'w') as f:
        f.create_dataset('x_data', data=np.array(x))
        f.create_dataset('y_data', data=np.array(y))


class DataSet:
    def __init__(self):
        with h5py.File('../manual_data_set/data.h5', 'r') as f:
            x, y = f['x_data'].value, f['y_data'].value

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
