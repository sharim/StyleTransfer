# StyleTransfer

BUAA课程“python编程与智能车技术”大作业存档。

## 一、程序简述

&emsp;&emsp;这是一个用 Python3 编写的快速图像风格迁移程序，原理基于感知损失，算法来自 Johoson 等人在 Perc-eptual Losses for Real-Time Style Transfer and Super-Resolution 一文中提出的基于生成模型迭代的风格迁移算法，在本程序中该深度学习算法用 PyTorch 实现。

&emsp;&emsp;程序实现功能为:

1. 针对输入的特定图片，提取其风格特征并利用现有数据集训练对应风格迁移模型；
2. 利用训练所得模型将任意输入图片快速转换为对应风格。

## 二、功能演示

### （一）Train mode

1. 程序运行指令
   >python src/main.py train --style-image <style_image> --dataset <path_to_dataset> --save-model-dir <path_to_save_model> --cuda <1_for_cuda_and_0_for_cpu>

2. 参数说明
   + train: 设置程序为Train mode以训练模型；
   + -\-style-image: 用于提取训练风格的图片；
   + -\-dataset:  数据集路径；
   + -\-save-model-name: 模型保存文件名；
   + -\-cuda: 设置为 0 时使用 CPU 训练，否则使用 GPU 训练；
   + 其它非必需参数: 见程序完整参数表。

3. 运行演示

    3.1. 选择 Style image
    ![wm](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/wm.jpg)

    3.2. 运行程序
    ![train_out](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/train_out.jpg)

    3.3. 保存模型
    ![out_pth](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/out_pth.jpg)

### （二）Evalute mode

1. 程序运行指令
    > python src/main.py eval --content-image <content_image> --model <saved_model> --generate-image <generate_image> --cuda <1_for_cuda_and_0_for_cpu>

2. 参数说明
   + eval: 设置程序为Evalute mode以生成图片；
   + -\-content-image: 用于提取内容以赋予新风格的图片；
   + -\-model:  用于生成图片的模型；
   + -\-generate-image: 生成图片保存路径及文件名；
   + -\-cuda: 设置为 0 时使用 CPU 训练，否则使用 GPU 训练；
   + 其它非必需参数: 见附件二。

3. 运行演示

    3.1. 选择 Content image
    ![Darksoul](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/Darksoul.jpg)

    3.2. 运行程序
    ![eval_out](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/eval_out.jpg)

    3.3. 查看 Generate image
    ![Darksoul](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/Darksoul_wm.jpg)

## 三、实现思路

### &emsp;&emsp;（一）风格迁移原理

&emsp;&emsp;本程序使用算法为Johoson等人在 Perceptual Losses for Real-Time Style Transfer and Super-Resolution 一文中提出的基于生成模型迭代的风格迁移算法。Johnson使用了两个网络：一个是VGG（也称损失网络），另一个是图像生成网络。网络结构如下：
![Johnson](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/Johnson.jpg)
&emsp;&emsp;本程序即是使用上图神经网络实现。

### &emsp;&emsp;（二）神经网络搭建

#### &emsp;&emsp;&emsp;&emsp; 1. vgg

&emsp;&emsp; 与Johnson的原始方法略有不同的地方在于，本程序使用的提取特征网络为VGG19而非VGG16，对应层结构需要细微调整：

```python
class vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        # 初始化参数
        super().__init__()
        pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        h = self.slice4(h)
        relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
```

#### &emsp;&emsp;&emsp;&emsp; 2. transformNet

##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 2.1. 网络结构图

![transformNet](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/transformNet.jpg)

##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 2.2. 程序实现

&emsp;&emsp;按照网络结构可用代码构建如下：

```python
class transformNet(nn.Module):
    # 按Johnson转换结构构建模型
    def __init__(self):
        super().__init__()
        # 初始化下采样模块
        self.down1 = conv(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.down2 = conv(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.down3 = conv(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # 初始化残差模块
        self.res1 = residual(128)
        self.res2 = residual(128)
        self.res3 = residual(128)
        self.res4 = residual(128)
        self.res5 = residual(128)
        # 初始化上采样模块
        self.up1 = conv(128, 64, 3, 1, 2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.up2 = conv(64, 32, 3, 1, 2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.up3 = conv(32, 3, 9, 1)
        # 初始化ReLU模块
        self.relu = nn.ReLU()

    def forward(self, x):
        # 下采样
        y = self.relu(self.in1(self.down1(x)))
        y = self.relu(self.in2(self.down2(y)))
        y = self.relu(self.in3(self.down3(y)))
        # 残差
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        # 上采样
        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        y = self.up3(y)
        return y
```

&emsp;&emsp;搭建好网络后, 还需要搭建对应上下采样模块和残差模块。
&emsp;&emsp;上下采样模块可合并到一个卷积模块：

```python
class conv(nn.Module):
    # 卷积模块，分上采样和下采样
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.up = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        y = x
        if self.up is not None:
            y = nn.functional.interpolate(y, mode='nearest', scale_factor=self.up)
        y = self.conv(self.reflection_pad(y))
        return y
```

&emsp;&emsp;残差模块：

```python
class residual(nn.Module):
    # 残差模块
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv(in_channels, in_channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = conv(in_channels, in_channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        resi = x
        y = self.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        y = y + resi
        return y
```

### &emsp;&emsp;（三）自顶向下设计思路

#### &emsp;&emsp;&emsp;&emsp; 1. 程序IPO模式

+ 输入1：待提取风格图片及训练集；
+ 处理1：提取图片风格并利用训练集训练对应风格迁移模型；
+ 输出1：保存风格迁移模型；
+ 输入2：待转换风格图片和风格迁移模型；
+ 处理2：利用风格迁移模型转换图片风格；
+ 输出2：保存生成图片。

#### &emsp;&emsp;&emsp;&emsp; 2. 程序自顶向下设计图

![design](https://github.com/sharim/StyleTransfer/raw/master/readmeImages/design.jpg)

### &emsp;&emsp;（四）模块化设计

#### &emsp;&emsp;&emsp;&emsp; 1. main

&emsp;&emsp;主程序入口，获取并分析指令，执行相应处理：

```python
import sys
import torch
from utils import getArgument
from stylize import stylize
from train import train


def main():
    args = getArgument()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    train(args) if args.subcommand == "train" else stylize(args)


if __name__ == "__main__":
    main()
```

#### &emsp;&emsp;&emsp;&emsp; 2. train

&emsp;&emsp;训练程序，依托transformNet和vgg19网络训练模型：

```python
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
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    dataSet = datasets.ImageFolder(args.dataset, transform)
    data = DataLoader(dataSet, batch_size=args.batch_size)
    # 初始化训练模型
    transformer = transformNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    # 预训练
    vgg = vgg19(requires_grad=False).to(device)
    styleTransform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.mul(255))])
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
                msg="{}\tEpoch {}:\t[{}/{}]\tcontent: {}\tstyle: {}\ttotal: {}".format(
                    time.ctime(), epoch + 1, count, len(dataSet),
                    agg_content_loss / (batchId + 1),
                    agg_style_loss / (batchId + 1),
                    (agg_content_loss + agg_style_loss) / (batchId + 1))
                print(msg)
            # 保存检查点
            if args.checkpoint_model_dir is not None and (
                    batchId + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(
                    epoch) + "_batch_id_" + str(batchId + 1) + ".pth"
                ckpt_model_path = args.checkpoint_model_dir + '/' + args.save_model_name
                ckpt_model_path += '/' + ckpt_model_filename
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
    # 保存模型
    transformer.eval().cpu()
    save_model_path = args.save_model_dir + '/' + args.save_model_name + '.pth'
    torch.save(transformer.state_dict(), save_model_path)

    print("model saved at", save_model_path)

```

#### &emsp;&emsp;&emsp;&emsp; 3. stylize

&emsp;&emsp;图片风格转换程序，利用保存的模型在transformNet生成转换风格后的图片：

```python
import torch
from torchvision import transforms
from transformNet import transformNet
from utils import loadImage, saveImage


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = loadImage(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.mul(255))])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = transformNet()
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        generate = style_model(content_image).cpu()
    saveImage(args.generate_image, generate[0])
    print("generate image saved as", args.generate_image)
```

#### &emsp;&emsp;&emsp;&emsp; 4. utils

&emsp;&emsp;utils中包含其他程序中所需用到的函数：
&emsp;&emsp;加载图片：

```python
def loadImage(image_name, size=None, scale=None):
    # 载入图片
    image = Image.open(image_name)
    image = image.resize(
        (size, size), Image.ANTIALIAS) if size is not None else image
    image = image.resize(
        (int(image.size[0] / scale), int(image.size[1] / scale)),
        Image.ANTIALIAS) if scale is not None else image
    return image
```

&emsp;&emsp;保存图片：

```python
def saveImage(image_name, data):
    # 保存图片
    Image.fromarray(data.clone().clamp(0, 255).numpy().transpose(
        1, 2, 0).astype("uint8")).save(image_name)

```

&emsp;&emsp;获取gram矩阵：

```python
def gram(m):
    (b, ch, h, w) = m.size()
    features = m.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
```

&emsp;&emsp;归一化：

```python
def normalizeBatch(batch):
    # 归一化
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
```

&emsp;&emsp;获取指令：

```python
def getArgument():
    main_arg_parser = argparse.ArgumentParser(
        description="parser for fast-style-transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands",
                                                dest="subcommand")

    # Train mode（训练模型指令）
    train_arg_parser = subparsers.add_parser(
        "train", help="parser for training arguments")
    # 必要
    train_arg_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=
        "path to the folder containing another folder with training images"
    )
    train_arg_parser.add_argument("--style-image",
                                  type=str,
                                  required=True,
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-name",
                                  type=str,
                                  required=True,
                                  help="give a name to trained model")
    train_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU")
    # 可选
    train_arg_parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size",
                                  type=int,
                                  default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument(
        "--save-model-dir",
        type=str,
        default="models/models",
        help=
        "path to folder where trained model will be saved, default is models")
    train_arg_parser.add_argument(
        "--checkpoint-model-dir",
        type=str,
        default=None,
        help="path to folder where checkpoints of trained models will be saved"
    )
    train_arg_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument(
        "--style-size",
        type=int,
        default=None,
        help="size of style-image, default is the original size of style image"
    )
    train_arg_parser.add_argument("--seed",
                                  type=int,
                                  default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument(
        "--content-weight",
        type=float,
        default=1e5,
        help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument(
        "--style-weight",
        type=float,
        default=1e10,
        help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr",
                                  type=float,
                                  default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help=
        "number of images after which the training loss is logged, default is 500"
    )
    train_arg_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help=
        "number of batches after which a checkpoint of the trained model will be created"
    )

    # Evalute mode（生成图片指令）
    eval_arg_parser = subparsers.add_parser(
        "eval", help="parser for evaluation/stylizing arguments")
    # 必要
    eval_arg_parser.add_argument(
        "--content-image",
        type=str,
        required=True,
        help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--generate-image",
                                 type=str,
                                 required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="saved model to be used for stylizing the image.")
    eval_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU")
    # 可选
    eval_arg_parser.add_argument(
        "--content-scale",
        type=float,
        default=None,
        help="factor for scaling down the content image")

    return main_arg_parser.parse_args()
```

## 四、程序完整参数表

参数|作用
:-:|:-:
Train mode|
train|设置程序为Evalute mode以生成图片
-\-style-image|用于提取训练风格的图片
-\-dataset|数据集路径
-\-save-model-name|模型保存文件名
-\-cuda|设置为 0 时使用 CPU 训练，否则使用 GPU 训练
-\-epochs|数据集经过神经网络的次数
-\-batch-size|单次传向神经网络的数据数
-\-save-model-dir|模型保存路径
-\-checkpoint-model-dir|检查点保存路径
-\-image-size|调整数据集中图片的尺寸
-\-style-size|调整提取风格图片的尺寸
-\-seed|设置随机种子
-\-content-weight|内容权重
-\-style-weight|风格权重
-\-lr|学习率
-\-log-interval|输出日志间隔
-\-checkpoint-interval|保存点间隔
Evalute mode|
eval|设置程序为Evalute mode以生成图片
-\-content-image|用于提取内容以赋予新风格的图片
-\-model|用于生成图片的模型
-\-generate-image|生成图片保存路径及文件名
-\-cuda|设置为 0 时使用 CPU 训练，否则使用 GPU 训练
