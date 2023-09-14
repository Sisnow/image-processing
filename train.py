import gc
import os
import datetime
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from Vgg16 import Vgg16
from transfer import load_image
from transfer_net import TransformerNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 格拉姆矩阵
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# 规范样本
def normal_batch(batch):
    mean = batch.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = batch.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def train():
    device = torch.device('cuda:0')
    np.random.seed(42)
    torch.manual_seed(42)
    # 模型配置
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(r"D:\images", transform)
    train_loader = DataLoader(train_dataset, batch_size=8)
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), 1e-3)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    # 风格配置
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image("./style/style3.jpg", size=256)
    style = style_transform(style)
    style = style.repeat(8, 1, 1, 1).to(device)

    feature_style = vgg(normal_batch(style))
    gram_style = [gram_matrix(y) for y in feature_style]
    # 训练
    for e in range(2):
        agg_content_loss = 0
        agg_style_loss = 0
        count = 0
        transformer.train()
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normal_batch(y)
            x = normal_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)
            # 计算内容损失
            content_loss = 1e5 * mse_loss(features_y.relu2_2, features_x.relu2_2)
            # 计算风格损失
            style_loss = 0
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= 1e10

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            # 显示进度
            if (batch_id + 1) % 5 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

    # 保存模型
    transformer.eval().cuda()
    save_model_path = "style2.pth"
    torch.save(transformer.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


train()
