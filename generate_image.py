import torch
import torch.nn as nn
import torch.optim as optimization
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm  # Use the non-notebook version of tqdm
from PIL import Image

def get_content_loss(target, content):
    return torch.mean((target-content)**2)


def gram_matrix(input, c, h, w):
    input = input.view(c, h * w)
    G = torch.mm(input, input.t())
    return G


def get_style_loss(target, style):
    _, c, h, w = target.size()
    G = gram_matrix(target, c, h, w)
    S = gram_matrix(style, c, h, w)
    return torch.mean((G - S) ** 2) / (c * h * w)


def getStyledImage(content_img, style_img, vgg, device, progress_bar=None):
    def load_img(image):
        img = Image.open(image)
        img = loader(img).unsqueeze(0)
        return img.to(device)

    img_size = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    content_img = load_img(content_img)
    style_img = load_img(style_img)

    steps = 1000
    alpha = 1
    beta = 10000

    target_img = content_img.clone().requires_grad_(True)
    optimizer = optimization.Adam([target_img], lr=0.001)

    for step in tqdm(range(steps)):
        target_feature = vgg(target_img)
        content_feature = vgg(content_img)
        style_feature = vgg(style_img)

        style_loss = 0
        content_loss = 0

        for target, content, style in zip(target_feature, content_feature, style_feature):
            content_loss += get_content_loss(target, content)
            style_loss += get_style_loss(target, style)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if progress_bar:
            progress_bar.progress((step + 1) / steps)

    return target_img
