# image segmentation with dice + BCE loss
import os
import torch
from torch.utils.data import DataLoader
from loss import DiceBCELoss

from dataloaders import load_data, train_val_test_split, ImageDataset, set_transforms
from vanila import FCNs, VGGNet
from param import CONFIG
from utils import save_image_with_mask, AverageMeter
import cv2
from datetime import datetime
import segmentation_models_pytorch as smp

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_dir = CONFIG["DATA"]["data_dir"]
label_dir = CONFIG["DATA"]["label_dir"]
batch_size = CONFIG["DATA"]["batch_size"]
resize = CONFIG["DATA"]["shape"]

data = load_data(data_dir)
transform = set_transforms(*resize)
train_data, val_data, test_data = train_val_test_split(data)

train_dataset = ImageDataset(train_data, data_dir, label_dir, transform)
val_dataset = ImageDataset(val_data, data_dir, label_dir, transform)
test_dataset = ImageDataset(test_data, data_dir, label_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


def train_one_epoch(model, train_loader, loss, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_loss.update(loss_value.item(), batch_size)
        loss_value.backward()

        # get iou with smp

        optimizer.step()

    return train_loss.avg


def validate(model, val_loader, loss):
    model.eval()
    val_loss = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.cuda(), labels.cuda()
            batch_size = data.size(0)
            outputs = model(data)
            loss_value = loss(outputs, labels)
            val_loss.update(loss_value.item(), batch_size)

    return val_loss.avg


def test(model, test_loader, loss):
    model.eval()
    test_loss = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.cuda(), labels.cuda()
            batch_size = data.size(0)
            outputs = model(data)
            loss_value = loss(outputs, labels)
            test_loss.update(loss_value.item(), batch_size)

    return test_loss.avg


def train(model, train_loader, val_loader, loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss, optimizer)
        val_loss = validate(model, val_loader, loss)
        test_loss = test(model, test_loader, loss)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
        )


def save_image(model, test_loader, test_data=test_dataset):
    output_dir = "./output/" + str(datetime.now().date()) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outputs = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.cuda()
        output = model(data)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0)

    for i in range(len(outputs)):
        data, data_name = test_data.get_original_image(i)
        output = outputs[i]
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        output = output.squeeze()
        # resize output to original size
        output = cv2.resize(output, (data.shape[1], data.shape[0]))
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        save_image_with_mask(data, output, data_name, output_dir)


def main():
    vgg_model = VGGNet()
    model = FCNs(pretrained_net=vgg_model, n_class=1)
    model = model.cuda()
    loss = DiceBCELoss(weight=CONFIG["TRAIN"]["loss_weight"])
    loss = loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)

    num_epochs = 10
    train(model, train_loader, val_loader, loss, optimizer, num_epochs)
    save_image(model, test_loader)


if __name__ == "__main__":
    main()
