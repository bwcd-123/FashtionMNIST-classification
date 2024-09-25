import torch
import torchvision
from torchvision import transforms
import swanlab
from matplotlib import pyplot as plt
import argparse

from utils import get_fashion_mnist_objects, choose_model

def main(opt):
    # 加载数据集
    net = choose_model(opt.model).to(opt.device)
    net.load_state_dict(torch.load(
        "weights/{}.pth".format(net._get_name())))
    
    logger = swanlab.init(
        experiment_name=f"{opt.model}-predict",
        description=f"{opt.model} model predict.",
        config=opt,
        logdir="logs"
    )
    net.eval()
    # 加载数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../datasets',
        train=False,
        )
    # 取前9个图片进行预测
    for i in range(9):
        image_data = transforms.ToTensor()(mnist_test[i][0]).to(opt.device)
        # 推理
        pred = net(image_data.unsqueeze(0)).argmax(-1)
        # 绘图
        plt.subplot(3, 3, i+1)
        plt.imshow(mnist_test[i][0])
        plt.title("true: {},\npredict: {}".format(
            get_fashion_mnist_objects(mnist_test[i][1]),
            get_fashion_mnist_objects(pred)[0]
        ))
    plt.tight_layout()
    logger.log({f"{net._get_name()} predict": swanlab.Image(plt)})
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--device', default="cuda", help='device')
    args.add_argument('--model', type=str, default="LeNet", help='model name: [ResNet12, ResNet4, LeNet, MLP2, MLP5]')
    opt = args.parse_args()
    main(opt)