# import image data from data/ folder
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


def load_data(data_dir):
    # data -> list of folders ('data/train/image/CE~', 'data/train/image/CE~', ...)
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                data.append(os.path.join(root, file))
    return data


def set_transforms(size_x=225, size_y=225):
    # set up transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size_x, size_y)),
        ]
    )
    return transform


def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1):
    # split data into train, val, test
    total_size = len(data)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = int(test_size * total_size)

    # shuffle list
    np.random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]
    return train_data, val_data, test_data

def train_val_split(data, train_size=0.8, val_size=0.2):
    # split data into train, val, test
    total_size = len(data)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)

    # shuffle list
    np.random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    return train_data, val_data


class ImageDataset(Dataset):
    def __init__(self, data, data_dir, transform=None):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # check if data and label are the same name after ../(dir)/
        # data_dir = os.path.join(self.data_dir, self.data[idx])
        # label_dir -> change "crop" to "target"
        # label_dir = os.path.join(
        #     self.data_dir, self.data[idx].replace("image", "target")
        # )
        data_dir = self.data[idx]
        label_dir = self.data[idx].replace("image", "target")
        data = cv2.imread(data_dir)

        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        # label to binary
        label[label > 0] = 1
        label = label.astype(np.float32)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        return data, label

    def get_sample_size(self, idx):
        data = cv2.imread(os.path.join(self.data_dir, self.data[idx]))
        return data.shape

    def get_original_image(self, idx):
        data = cv2.imread(os.path.join(self.data_dir, self.data[idx]))
        name = self.data[idx]
        return data, name


class Test_ImageDataset(Dataset):
    def __init__(self, data, data_dir, transform=None):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dir = self.data[idx]
        data = cv2.imread(data_dir)

        if self.transform:
            data = self.transform(data)
        return data

    def get_sample_size(self, idx):
        data = cv2.imread(self.data[idx])
        return data.shape

    def get_original_image(self, idx):
        data = cv2.imread(self.data[idx])
        name = self.data[idx]
        return data, name