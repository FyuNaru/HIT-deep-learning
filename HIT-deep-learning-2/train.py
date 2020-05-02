import MyDataSet
from torch.utils.data import DataLoader,TensorDataset
import torch
from model import AlexNet
from tensorboardX import SummaryWriter

# 设定batch_size
batch_size = 128
# epoch 的数目
n_epochs = 30
# 模型保存位置
path = "model.pt"

# 从自定义的dataset中将数据读入
data, label = MyDataSet.get_train_data()
train_data = TensorDataset(data, label)
data, label = MyDataSet.get_test_data()
test_data = TensorDataset(data, label)
# 创建数据加载器
train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 0)

#数据可视化
write = SummaryWriter("./result")
# 设定GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 训练函数
def train():
    # 定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    #optimizer = torch.optim.(params=model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 开始训练
    for epoch in range(n_epochs):
        # train
        model.train()
        train_loss = 0.0
        i = -1
        for data, label in train_loader:
            i = i + 1
            optimizer.zero_grad()
            #将数据放至GPU并计算输出
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            #计算误差
            loss = lossfunc(output, label)
            #反向传播
            loss.backward()
            #将参数更新至网络中
            optimizer.step()
            #计算误差
            train_loss += loss.item()
            write.add_scalar('Train', loss, epoch*len(train_loader)+i)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        #当结果较好时将模型保存
        if test() >= 75:
            torch.save(model, path)

#测试函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            #将数据放到GPU上
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return 100.0 * correct / total

try:
    #加载已有的模型
    model = torch.load(path).to(device)
    test()
except FileNotFoundError:
    model = AlexNet().to(device)
    train()

write.close()



