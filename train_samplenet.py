from utils import AverageMeter, get_iou_score
from time import time
import torch
from torchviz import make_dot
import os


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
        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        optimizer.step()

    return train_loss.avg, train_iou.avg


def make_dot_graph(model, image_name, image_dir):
    x = torch.randn(1, 3, 224, 224).requires_grad_(True)
    y = model(x)
    g = make_dot(y, params=dict(list(model.named_parameters())))
    image_path = os.path.join(image_dir, image_name)
    g.render(image_path, view=False, format="jpg")


def test(model, test_loader, loss):
    model.eval()
    test_loss = AverageMeter()
    test_iou = AverageMeter()

    # for batch_idx, (data, labels) in enumerate(test_loader):
    for batch_idx, data in enumerate(test_loader):
        # data, labels = data.cuda(), labels.cuda()
        data = data.cuda()
        batch_size = data.size(0)
        outputs = model(data)
        # loss_value = loss(outputs, labels)
        # test_loss.update(loss_value.item(), batch_size)
        # iou_score = get_iou_score(outputs, labels)
        # test_iou.update(iou_score, batch_size)

    # return test_loss.avg, test_iou.avg


def train_samplenet(model, train_loader, test_loader, loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, loss, optimizer)
        # test_loss, test_iou = test(model, test_loader, loss)
        test(model, test_loader, loss)
        # print(
        #     f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}"
        # )


def check_gpu_latency(model, height, width, repeat=100):
    model.eval()
    model = model.cuda()
    gpu_lat = AverageMeter()

    with torch.no_grad():
        for _ in range(repeat):
            data = torch.randn(1, 3, height, width).cuda()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(data)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            gpu_lat.update(elapsed_time)

    return gpu_lat.avg


def check_cpu_latency(model, height, width, repeat=100):
    model.cpu()
    model.eval()
    cpu_lat = AverageMeter()

    with torch.no_grad():
        for _ in range(repeat):
            data = torch.randn(1, 3, height, width)
            data.cpu()
            start_time = time()
            model(data)
            end_time = time()

            elapsed_time = (end_time - start_time) * 1000
            cpu_lat.update(elapsed_time)

    return cpu_lat.avg
