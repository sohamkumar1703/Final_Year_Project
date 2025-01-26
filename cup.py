import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        
    def forward(self, x):
        skip_connections = []
        for i in range(8):
            x = list(self.resnet.children())[i](x)
            if i in [2, 4, 5, 6, 7]:
                skip_connections.append(x)
        encoder_outputs = skip_connections.pop(-1)
        skip_connections = skip_connections[::-1]
        
        return encoder_outputs, skip_connections
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return  self.dconv(x)
    
class Multi_CAM_Block(nn.Module):
    def __init__(self, f_g, f_int):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.downsize = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, f_g,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(f_g),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.avgpool(g)
        g2 = self.maxpool(g)
        g1 = self.downsize(g1)
        g2 = self.downsize(g2)
        decoder =  g1 + g2
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x1 = self.downsize(x1)
        x2 = self.downsize(x2)
        encoder =  x1 + x2
        psi = self.relu(encoder+decoder)
        psi = self.psi(psi)
        
        return psi

class Multi_PAM_Block(nn.Module):
    def __init__(self, f_g, f_int):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(f_g, f_int, kernel_size=1)
        self.final_conv = nn.Conv2d(3*f_int, f_int, kernel_size=7, padding=3)
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.conv1x1(self.avgpool(g))
        g2 = self.conv1x1(self.maxpool(g))
        g3 = self.conv1x1(g)
        g3 = torch.cat((g1,g2,g3), dim=1)
        decoder =  self.final_conv(g3)
        x1 = self.conv1x1(self.avgpool(x))
        x2 = self.conv1x1(self.maxpool(x))
        x3 = self.conv1x1(x)
        x3 = torch.cat((x1,x2,x3), dim=1)
        encoder =  self.final_conv(x3)
        psi = self.relu(encoder+decoder)
        psi = self.psi(psi)
        
        return psi
    
class Attention_Block(nn.Module):
    def __init__(self, f_g, f_int):
        super().__init__()
        
        self.pam = Multi_PAM_Block(f_g, f_int)
        self.cam = Multi_CAM_Block(f_g, f_int)
        
    def forward(self, g, x):
        return x * self.pam(g, x) * self.cam(g, x)
    
class UnetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.dconv = DoubleConv(out_channels*2, out_channels)
        self.att = Attention_Block(out_channels, out_channels//2)
    def forward(self, layer_input, skip_input):
        u = self.convt(layer_input)
        u = self.norm1(u)
        u = self.act1(u)
        skip_input = self.att(u, skip_input)
        u = torch.cat((u, skip_input), dim=1)
        u = self.dconv(u)
        return u

class DSC_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSC_Block, self).__init__()
        
        # Depthwise separable convolutions with different dilation rates
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1), #, groups=in_channels
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(3*in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(4*in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(5*in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out1 = self.conv1(x)
        cat1 = torch.cat((x, out1), dim=1)
        cat1 = self.conv5(cat1)
        out2 = self.conv2(cat1)
        cat2 = torch.cat((x, out1, out2), dim=1)
        cat2 = self.conv6(cat2)
        out3 = self.conv3(cat2)
        cat3 = torch.cat((x, out1, out2, out3), dim=1)
        cat3 = self.conv7(cat3)
        out4 = self.conv4(cat3)
        cat4 = torch.cat((x, out1, out2, out3, out4), dim=1)
        cat4 = self.conv8(cat4)
        
        return cat4
    
class AttentionUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.encoder = FeatureExtractor()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upsample1 = UnetUpSample(512, 256)
        self.upsample2 = UnetUpSample(256, 128)
        self.upsample3 = UnetUpSample(128, 64)
        self.upsample4 = UnetUpSample(64, 64)
        
        self.final_convt = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.spp1 = DSC_Block(512,512)
        self.spp2 = DSC_Block(256,256)
        self.spp3 = DSC_Block(128,128)
        self.spp4 = DSC_Block(64,64)
        self.spp5 = DSC_Block(64,64)
        #self.final_conv = DSC_Block(64, out_channels)
        
    def forward(self, x):
        x1, skip_connections = self.encoder(x)
        x2 = self.upsample1(self.spp1(x1), self.spp2(skip_connections[0]))
        x3 = self.upsample2(x2, self.spp3(skip_connections[1]))
        x4 = self.upsample3(x3, self.spp4(skip_connections[2]))
        x5 = self.upsample4(x4, self.spp5(skip_connections[3]))
        x6 = self.final_convt(x5)
        
        return self.final_conv(x6)
    
model_path = '/home/soham/Downloads/Final_Year_Project/model-Modified-Unet base-resnet-18 dim-256x256-fold-3.bin'

model = AttentionUNet().to("cpu") #cuda:0
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask >= 0.5,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

import process
from process import enhance_image
def process_and_visualize(img_path, model):

    #img = cv2.imread(img_path)
    img = enhance_image(img_path)
    # Apply transformations
    transforms = A.Compose([
        A.Resize(height=256, width=256, interpolation=3),
        ToTensorV2()
    ], p=1.0)
    transformed = transforms(image=img)
    image_tensor = torch.tensor(transformed['image'], dtype=torch.float32, device="cpu")  # cuda:0
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    output_np = output.squeeze().cpu().numpy()
    output_np = output_np > 0.5

    # Resize the original image
    img_resized = cv2.resize(img, (256, 256))

    # Display side by side
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title('Original Image with Preprocessing')

    # Apply mask and visualize
    overlay = apply_mask(img_resized, output_np, color=(1.0, 0.0, 0.0))

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Predicted  Disc Mask')
    plt.show()

# Example usage
img_path = '/home/soham/Downloads/Final_Year_Project/glaucoma.jpg'
process_and_visualize(img_path, model)


