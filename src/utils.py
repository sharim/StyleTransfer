import argparse
from PIL import Image


def loadImage(image_name, size=None, scale=None):
    # 载入图片
    image = Image.open(image_name)
    image = image.resize((size, size), Image.ANTIALIAS) if size is not None else image
    image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS) if scale is not None else image
    return image


def saveImage(image_name, data):
    # 保存图片
    Image.fromarray(data.clone().clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")).save(image_name)


def gram(m):
    (b, ch, h, w) = m.size()
    features = m.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalizeBatch(batch):
    # 归一化
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def getArgument():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-style-transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    # Train mode（训练模型指令）
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    # 必要
    train_arg_parser.add_argument("--dataset", type=str, required=True, help="path to the folder containing another folder with training images")
    train_arg_parser.add_argument("--style-image", type=str, required=True, help="path to style-image")
    train_arg_parser.add_argument("--save-model-name", type=str, required=True, help="give a name to trained model")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    # 可选
    train_arg_parser.add_argument("--epochs", type=int, default=2, help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4, help="batch size for training, default is 4")
    train_arg_parser.add_argument("--save-model-dir", type=str, default="models/models", help="path to folder where trained model will be saved, default is models")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None, help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256, help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None, help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5, help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10, help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500, help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000, help="number of batches after which a checkpoint of the trained model will be created")

    # Evalute mode（生成图片指令）
    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    # 必要
    eval_arg_parser.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--generate-image", type=str, required=True, help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image.")
    eval_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    # 可选
    eval_arg_parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")

    return main_arg_parser.parse_args()
