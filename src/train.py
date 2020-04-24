import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from vgg import vgg19
from transformNet import transformNet
from utils import loadImage, gram, normalizeBatch
import time


def train(args):
    # 是否使用GPU
    device = torch.device("cuda" if args.cuda else "cpu")
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 数据载入及预处理
    transform = transforms.Compose([transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    dataSet = datasets.ImageFolder(args.dataset, transform)
    data = DataLoader(dataSet, batch_size=args.batch_size)
    # 初始化训练模型
    transformer = transformNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    # 预训练
    vgg = vgg19(requires_grad=False).to(device)
    styleTransform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    style = loadImage(args.style_image, size=args.style_size)
    style = styleTransform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    features_style = vgg(normalizeBatch(style))
    gram_style = [gram(y) for y in features_style]
    # 训练
    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batchId, (x, _) in enumerate(data):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            # 数据部署到GPU或CPU
            x = x.to(device)
            y = transformer(x)
            # 归一化
            y = normalizeBatch(y)
            x = normalizeBatch(x)
            # 提取特征
            features_y = vgg(y)
            features_x = vgg(x)
            # 计算 content loss
            content_loss = args.content_weight * mse_loss(
                features_y.relu3_3, features_x.relu3_3)
            # 计算 style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight
            # 计算 total loss
            total_loss = content_loss + style_loss
            # 反向传播
            total_loss.backward()
            # 更新模型
            optimizer.step()
            # 计算 aggregate loss
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            # 输出日志
            if (batchId + 1) % args.log_interval == 0:
                msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {}\tstyle: {}\ttotal: {}".format(time.ctime(), epoch + 1, count, len(dataSet), agg_content_loss / (batchId + 1), agg_style_loss / (batchId + 1), (agg_content_loss + agg_style_loss) / (batchId + 1))
                print(msg)
            # 保存检查点
            if args.checkpoint_model_dir is not None and (batchId + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batchId + 1) + ".pth"
                ckpt_model_path = args.checkpoint_model_dir + '/' + args.save_model_name
                ckpt_model_path += '/' + ckpt_model_filename
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
    # 保存模型
    transformer.eval().cpu()
    save_model_path = args.save_model_dir + '/' + args.save_model_name + '.pth'
    torch.save(transformer.state_dict(), save_model_path)

    print("model saved at", save_model_path)
