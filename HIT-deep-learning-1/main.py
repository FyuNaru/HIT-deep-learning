import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#一次取出的训练样本数
batch_size = 16
# epoch 的数目
n_epochs = 10

#读取数据
train_data = datasets.MNIST(root="./data", train=True, download=True,transform=transforms.ToTensor())
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
#创建数据加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)

class MLP(nn.Module):
    def __init__(self):
        #继承自父类
        super(MLP, self).__init__()
        #创建一个三层的网络
        #输入的28*28为图片大小，输出的10为数字的类别数
        hidden_first = 512
        hidden_second = 512
        self.first = nn.Linear(in_features=28*28, out_features=hidden_first)
        self.second = nn.Linear(in_features=hidden_first, out_features=hidden_second)
        self.third = nn.Linear(in_features=hidden_second, out_features=10)

    def forward(self, data):
        #先将图片数据转化为1*784的张量
        data = data.view(-1, 28*28)
        data = F.relu(self.first(data))
        data = F.relu((self.second(data)))
        data = F.log_softmax(self.third(data), dim = 1)

        return data

def train():
    # 定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss().cuda()
    #lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            #将数据放至GPU并计算输出
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            #计算误差
            loss = lossfunc(output, target)
            #反向传播
            loss.backward()
            #将参数更新至网络中
            optimizer.step()
            #计算误差
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        test()
    #最后将模型保存
    path = "model.pt"
    torch.save(model, path)

#测试程序
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            #将数据放到GPU上
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return 100.0 * correct / total

model = MLP()
#将模型放到GPU上
model = model.cuda()
train()






