# Show information from FashtinMNIST.
# Select the first nine data to show images.


import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import get_fashion_mnist_objects

train_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=True)
val_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=False)

print("train_dataset's length:", len(train_dataset))
print("val_dataset's length:", len(val_dataset))

data_image = transforms.ToTensor()(val_dataset[0][0])
print("single image's shape: ", data_image.shape,
      ", label:", val_dataset[0][1],
      ", object: ", get_fashion_mnist_objects([val_dataset[0][1]]))

# show the first nine pictures from val_dataset
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(val_dataset[i][0])
    plt.title("label: {}\nobject: {}".format(
        val_dataset[i][1],
        get_fashion_mnist_objects([val_dataset[i][1]])[0]
    ))

plt.tight_layout()
plt.show()
