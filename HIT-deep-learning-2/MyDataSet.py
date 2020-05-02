import pickle
import torch
import numpy as np

# 读取官网下载的数据，其中data部分是# 10000 * 3072(32*32*3=3072)的二维数组
def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

#将对应文件中的数据读入，并重新生成10000*3*32*32的数据以及10000的标签
def load_data(filename):
    data = load_file('cifar-10-batches-py/'+filename)
    X = data['data']
    label = data['labels']
    X = X.reshape((-1, 3, 32, 32))
    return X, label

# 将五个训练数据文件中的数据打包成一个，并封装成tensor对象
# 输出大小
# torch.Size([50000, 3, 32, 32])
# torch.Size([50000])
def get_train_data():
    d = []
    l = []
    for i in range(5):
        X, label = load_data('data_batch_'+str(i+1))
        d.append(X)
        l.append(label)
    train_data = np.vstack((d[0], d[1], d[2], d[3], d[4]))
    train_label = np.hstack((l[0], l[1], l[2], l[3], l[4]))
    return torch.tensor(train_data).float(), torch.tensor(train_label).long()

# 将测试集文件中的数据封装成tensor对象
# 输出大小
# torch.Size([10000, 3, 32, 32])
# torch.Size([10000])
def get_test_data():
    X, label = load_data('test_batch')
    return torch.tensor(X).float(), torch.tensor(label).long()
'''
data, label = get_test_data()
print(data.size())
print(label.size())
'''
