import torch
from utils import AverageMeter, get_iou_score
import torch.nn as nn
from param import CONFIG

def train_one_epoch_weight(model, train_loader, loss, optimizer_weight):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer_weight.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_loss.update(loss_value.item(), batch_size)
        loss_value.backward()

        # get iou with smp
        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        optimizer_weight.step()

    return train_loss.avg, train_iou.avg


def train_one_epoch_weight_alpha(
    model,
    train_loader,
    val_loader,
    loss,
    optimizer_weight,
    optimizer_alpha,
    clip_grad=5,
):
    model.train()
    train_w_loss = AverageMeter()
    train_w_iou = AverageMeter()
    train_a_loss = AverageMeter()
    train_a_iou = AverageMeter()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer_weight.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_w_loss.update(loss_value.item(), batch_size)
        train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        loss_value.backward()
        optimizer_weight.step()

    for batch_idx, (data, labels) in enumerate(val_loader):
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_a_loss.update(loss_value.item(), batch_size)
        train_a_iou.update(get_iou_score(outputs, labels), batch_size)
        loss_value.backward()
        optimizer_alpha.step()
        # nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if len(CONFIG["GPU"]) >= 2:
            model.module.clip_alphas()
        else:
            model.clip_alphas()

    return train_w_loss.avg, train_w_iou.avg, train_a_loss.avg, train_a_iou.avg


def test(model, test_loader, loss):
    model.eval()
    test_loss = AverageMeter()
    test_iou = AverageMeter()

    with torch.no_grad():
        # for batch_idx, (data, labels) in enumerate(test_loader):
        for batch_idx, data in enumerate(test_loader):
            # data, labels = data.cuda(), labels.cuda()
            data = data.cuda()
            batch_size = data.size(0)
            outputs = model(data)
            # loss_value = loss(outputs, labels)
            # test_loss.update(loss_value.item(), batch_size)
            # test_iou.update(get_iou_score(outputs, labels), batch_size)

    # return test_loss.avg, test_iou.avg


def train_architecture(
    model,
    train_loader,
    val_loader,
    test_loader,
    loss,
    optimizer_weight,
    optimizer_alpha,
    num_epochs,
    warmup_epochs=10,
    clip_grad=5,
):
    for epoch in range(warmup_epochs):
        train_loss, train_iou = train_one_epoch_weight(
            model, train_loader, loss, optimizer_weight
        )
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}"
        )

    for epoch in range(warmup_epochs, num_epochs):
        train_w_loss, train_w_iou, train_a_loss, train_a_iou = (
            train_one_epoch_weight_alpha(
                model,
                train_loader,
                val_loader,
                loss,
                optimizer_weight,
                optimizer_alpha,
                clip_grad,
            )
        )
        # test_loss, test_iou = test(model, test_loader, loss)
        test(model, test_loader, loss)
        if len(CONFIG["GPU"]) >= 2:
            alphas = model.module.get_alphas()
        else:
            alphas = model.get_alphas()
        print(f"Alphas: {alphas}")
        print(
            f"[Train W] Epoch {epoch+1}/{num_epochs}, Train Weight Loss: {train_w_loss:.4f}, Train Weight IOU: {train_w_iou:.4f}"
        )
        print(
            f"[Train A] Epoch {epoch+1}/{num_epochs}, Train Alpha Loss: {train_a_loss:.4f}, Train Alpha IOU: {train_a_iou:.4f}"
        )
        # print(
        #     f"[Test] Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test IOU: {test_iou:.4f}"
        # )
