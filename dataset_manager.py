import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class DatasetManager:
    def __init__(self, train=True):
        torch.manual_seed(0)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ])
        self.mnist_data = datasets.Caltech256(root='./data', download=True, transform=transform)
        print(len(self.mnist_data))
        filtered = []
        for i in self.mnist_data:
            if i[0].shape[0] == 3:
                filtered.append(i)
        self.mnist_data = filtered
        print(len(self.mnist_data))

    def get_ds(self):
        return self.mnist_data


if __name__ == "__main__":
    dsm = DatasetManager()
    data_loader = torch.utils.data.DataLoader(dataset=dsm.get_ds(),batch_size=64,shuffle=True)
    for data, label in data_loader:
        print(data.shape)
        print(label.shape)
        print(label[0])
        for i in data:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)
