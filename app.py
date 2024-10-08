import torch
import torchvision.transforms as transforms
import streamlit as st
import torch.nn as nn
import torchvision.models as models
import generate_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.select_features = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, output):
        features = []
        for name, layer in self.vgg._modules.items():
            output = layer(output)
            if name in self.select_features:
                features.append(output)
        return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG().to(device).eval()


def show_image(target):
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denormalization(img).clamp(0, 1)
    img = img.permute(1, 2, 0).detach().cpu().numpy()

    st.image(img, use_column_width=True)

uploaded_context = st.file_uploader("Choose content image...", type=["jpg", "png", "jpeg"])
uploaded_style = st.file_uploader("Choose style image...", type=["jpg", "png", "jpeg"])

progress_bar = st.progress(0)

if st.button("Process Image"):
    target_img = generate_image.getStyledImage(content_img=uploaded_context, style_img=uploaded_style, vgg=vgg, device=device, progress_bar=progress_bar)
    show_image(target_img)
