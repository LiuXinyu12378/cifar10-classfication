import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("mps")
# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义MLP模型
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(3*32*32, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积，接收3个通道输入，输出32个通道，卷积核大小为3x3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 第二层卷积，接收32个通道输入，输出64个通道，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 第三层卷积，接收64个通道输入，输出128个通道，卷积核大小为3x3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 最大池化层，使用2x2窗口
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层，输入特征为128*4*4（因为CIFAR-10图像大小为32x32，经过三次池化后变为4x4）
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # 输出层，10个类别
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 通过三层卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 扁平化特征图用于全连接层
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


learning_rates = [0.1, 0.01, 0.001]  # 不同的学习率
colors = ['r', 'g', 'b']  # 对应的绘图颜色

plt.figure(figsize=(10, 8))

# 训练模型并绘制损失曲线
for lr, color in zip(learning_rates, colors):
    # net = MLP().to(device)
    net = CNN().to(device)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20
    losses = []
    
    for epoch in range(num_epochs):
        print(lr,epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        losses.append(epoch_loss)
    
    plt.plot(range(num_epochs), losses, label=f'LR={lr}', color=color)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve for Different Learning Rates')
plt.legend()

# 保存图像到文件系统
plt.savefig('training_loss_comparison.png', format='png', dpi=300)

plt.show()
