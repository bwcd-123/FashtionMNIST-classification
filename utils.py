import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import math

from models import ResNet12, ResNet4, LeNet, MLP2, MLP5

def get_fashion_mnist_objects(labels):
    """get objects according to labels"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    if isinstance(labels, int):
        return text_labels[labels]
    else:
        return [text_labels[int(i)] for i in labels]


def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    accu_loss = torch.zeros(1).to(device)   # 累计损失
    accu_num = torch.zeros(1).to(device)    # 累计预测正确的样本数
    data_num = torch.zeros(1).to(device)    # 累计计算的样本数
    data_len = len(train_loader)

    for idx, (datas, labels) in enumerate(train_loader):
        datas, labels = datas.to(device), labels.to(device)
        ret = model(datas)

        pred = ret.argmax(dim=1)
        loss = loss_function(ret, labels)
        right = torch.eq(pred, labels).sum()

        accu_num += right
        accu_loss += loss.detach()
        data_num += len(datas)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (idx+1) == data_len:
            return accu_loss.item()/(idx + 1), accu_num.item()/data_num.item()


def train_epochs(model, train_loader, test_loader, sumwriter, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=sumwriter.config.initial_lr)
    if sumwriter.config.lrf is not None:
        lf = lambda x: ((1 + math.cos(x * math.pi / sumwriter.config.epochs)) / 2) * (1 - sumwriter.config.lrf) + sumwriter.config.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    optimizer.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(sumwriter.config.epochs):
        start_time = time.perf_counter()

        accu_loss = torch.zeros(1).to(device)   # 累计损失
        accu_num = torch.zeros(1).to(device)    # 累计预测正确的样本数
        data_num = torch.zeros(1).to(device)    # 累计计算的样本数
        data_len = len(train_loader)

        for idx, (datas, labels) in enumerate(train_loader):
            datas, labels = datas.to(device), labels.to(device)
            ret = model(datas)

            pred = ret.argmax(dim=1)
            loss = loss_fn(ret, labels)
            right = torch.eq(pred, labels).sum()

            accu_num += right
            accu_loss += loss.detach()
            data_num += len(datas)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (idx+1) == data_len:
                print("epoch: {}, train_loss: {}, train_acc: {}".format(
                    epoch, accu_loss.item()/(idx + 1), accu_num.item()/data_num.item()))
                
                sumwriter.log({"train_loss": accu_loss.item() / (idx + 1),
                               "train_acc": accu_num.item() / data_num.item(),
                               "lr": optimizer.param_groups[0]['lr']
                               })
        
        # eval
        eval_loss, eval_acc = evaluate(model, test_loader, device)
        # output data and record them
        print("epoch: {}, test_loss: {}, test_acc: {}, lr: {}".format(
            epoch, eval_loss, eval_acc, optimizer.param_groups[0]['lr']))
        
        sumwriter.log({"test_loss": eval_loss,
                       "test_acc": eval_acc,
                       })
        
        # lr change
        if sumwriter.config.lrf is not None:
            scheduler.step()
        
        end_time = time.perf_counter()
        print("used time: {}\n".format(end_time - start_time))


def evaluate(model, val_loader, loss_fn, device):
    """
    eval step
    """
    model.eval()
    loss = torch.zeros(1).to(device)
    data_num = torch.zeros(1).to(device)
    right = torch.zeros(1).to(device)
    data_len = len(val_loader)

    for datas, labels in val_loader:
        datas, labels = datas.to(device), labels.to(device)
        ret = model(datas)

        loss += loss_fn(ret, labels)
        pred = ret.argmax(dim=1)
        right += torch.eq(pred, labels).sum()
        data_num += len(datas)

    return loss.item() / data_len, right.item() / data_num.item()


def choose_model(model_name:str, **kwargs):
    """
    choose model from ResNet12, ResNet4, LeNet, MLP2, MLP5
    """
    if model_name == "ResNet12":
        return ResNet12(**kwargs)
    elif model_name == "ResNet4":
        return ResNet4(**kwargs)
    elif model_name == "LeNet":
        return LeNet(**kwargs)
    elif model_name == "MLP2":
        return MLP2(**kwargs)
    elif model_name == "MLP5":
        return MLP5(**kwargs)
    else:
        raise ValueError(f"no such model: {model_name}")