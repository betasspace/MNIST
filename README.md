# MNIST

参考文章

[1.tensorflow入门-mnist手写数字识别](https://geektutu.com/post/tensorflow-mnist-simplest.html)

##tensorboard
python3 -m tensorboard.main --logdir=./v1/log

##dataset
dataset is downloaded from http://yann.lecun.com/exdb/mnist/

[数据集的格式](https://blog.csdn.net/wspba/article/details/54311566)  
数据集是write('rb') 的file，然后还压缩了一下下，用gzip.GzipFile 打开
文件头几位包含特殊用途  
training set image file:
offset  value  description  
0000    2051   magic number  
0004    60000  number of images  
0008    28     number of rows  
0012    28     number of columns  
0016    ?      pixel  
0017    ?      pixel  
...

training set label file:  
offset  value  description  
0000    2049   magic number  
0004    60000  number of items  
0008    ?      label  
0009    ?      label  
...

##note
首先这是个多分类问题，0-9种数字，所以one-hot + [softmax](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)（可以理解为一个可导的函数，先由h函数归一化，再由J函数放大；用于表示互斥的多分类问题）  
接着是网络结构，第一种尝试是先用一层的网络，即 W * x，其中W是模型参数，x是输入图像向量。




