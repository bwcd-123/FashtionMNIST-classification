import torch
from  torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import math

from utils import train_one_epoch, evaluate, choose_model
import swanlab

# 设置超参数
# model name: ['ResNet12', 'ResNet4', 'LeNet', 'MLP2', 'MLP5']
net_name = "ResNet12"
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 10000
lr = 0.01
lrf= None

net = choose_model(net_name).to(device)

logger = swanlab.init(
    experiment_name=net._get_name(),
    description="{} without learning rate scheduler".format(net._get_name()),
    config={
        'epochs': epochs,
        'initial_lr': lr,
        "batch_size": batch_size,
        "lrf": lrf,
        },  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)

mnist_train = torchvision.datasets.FashionMNIST(root='../datasets',
                                                train=True,
                                                transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root='../datasets',
                                               train=False,
                                               transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size, shuffle=True)

# train_epochs(net, train_loader, test_loader, run, device)
optimizer = optim.Adam(net.parameters(), lr=logger.config.initial_lr)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    start_time = time.perf_counter()

    net.train()
    optimizer.zero_grad()

    train_loss, train_acc = train_one_epoch(net, train_loader, loss_fn, optimizer, device)

    print("epoch: {}, train_loss: {}, train_acc: {}".format(
            epoch, train_loss, train_acc))
    logger.log({"train_loss": train_loss, "train_acc": train_acc, "lr": optimizer.param_groups[0]["lr"]})
    
    net.eval()
    eval_loss, eval_acc = evaluate(net, test_loader, loss_fn, device)
    print("epoch: {}, eval_loss: {}, eval_acc: {}".format(
            epoch, eval_loss, eval_acc))
    logger.log({"eval_loss": eval_loss, "eval_acc": eval_acc})

    if logger.config.lrf is not None:
        lf = lambda x: ((1 + math.cos(x * math.pi / logger.config.epochs)) / 2) * (1 - logger.config.lrf) + logger.config.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    end_time = time.perf_counter()
    print("epoch time: {}\n".format(end_time - start_time))

# try to save net
try:
    torch.save(net.state_dict(), "./weights/{}.pth".format(net._get_name()))
except Exception as e:
    print(e)
else:
    print("success to save net at: ./weights/{}.pth".format(net._get_name()))