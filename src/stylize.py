import torch
from torchvision import transforms
from transformNet import transformNet
from utils import loadImage, saveImage


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = loadImage(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
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
