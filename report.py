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
import process
from process import enhance_image, extract_largest_contour_roi
from PIL import Image
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

cup_model_path = 'model-Modified-Unet base-resnet-18 dim-256x256-fold-3.bin'

cup_model = AttentionUNet().to("cpu") #cuda:0
cup_model.load_state_dict(torch.load(cup_model_path, map_location=torch.device('cpu')))

disc_model_path = 'model-ModifiedUnet5 base-resnet dim-256x256-fold-2.bin'

disc_model = AttentionUNet().to("cpu") #cuda:0
disc_model.load_state_dict(torch.load(disc_model_path, map_location=torch.device('cpu')))

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask >= 0.5,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def resize_image_to_square(image, target_size):
    # Get the original height and width
    height, width = image.shape[:2]

    # Determine the new size (assuming you want a square image)
    new_size = max(height, width)

    # Calculate the cropping box
    top = (new_size - height) // 2
    bottom = new_size - height - top
    left = (new_size - width) // 2
    right = new_size - width - left

    # Pad the image with black color
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize the image to the target size
    image = cv2.resize(image, (target_size, target_size))

    return image

def calculate_dimensions(binary_mask):
    # Convert binary mask to numpy array
    binary_mask_np = np.array(binary_mask, dtype=np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming there is only one contour (for simplicity)
    if len(contours) > 0:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contours[0], closed=True)

        # Calculate the area enclosed by the contour
        area = cv2.contourArea(contours[0])

        # Calculate other dimensions and features
        bounding_box = cv2.boundingRect(contours[0])
        diameter_vertical = bounding_box[3]  # Vertical diameter of the optic cup
        diameter_horizontal = bounding_box[2]  # Horizontal diameter of the optic cup

        return {
            'perimeter': perimeter,
            'area': area,
            'diameter_vertical': diameter_vertical,
            'diameter_horizontal': diameter_horizontal
            # Add more dimensions as needed
        }
    else:
        return {
            'perimeter': 0,
            'area': 0,
            'diameter_vertical': 0,
            'diameter_horizontal': 0
            # Initialize other dimensions as needed
        }

def process_and_visualize(img_path, cup_model, disc_model):

    # Original Image
    img = np.array(Image.open(img_path))
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Enhanced Image
    enhanced_img = enhance_image(img_path)

    # Resize the original image
    img_resized = resize_image_to_square(extract_largest_contour_roi(img), 256)
    #enhanced_img = enhance_image(img_resized)

    # Apply transformations
    transforms = A.Compose([
        A.Resize(height=256, width=256, interpolation=3),
        ToTensorV2()
    ], p=1.0)

    # Transform for disc_model
    transformed_disc = transforms(image=enhanced_img)
    image_tensor_disc = torch.tensor(transformed_disc['image'], dtype=torch.float32, device="cpu")  # cuda:0
    image_tensor_disc = image_tensor_disc / 255.0
    image_tensor_disc = image_tensor_disc.unsqueeze(0)

    # Transform for cup_model
    transformed_cup = transforms(image=enhanced_img)
    image_tensor_cup = torch.tensor(transformed_cup['image'], dtype=torch.float32, device="cpu")  # cuda:0
    image_tensor_cup = image_tensor_cup / 255.0
    image_tensor_cup = image_tensor_cup.unsqueeze(0)

    # Perform inference for disc_model
    disc_model.eval()
    with torch.no_grad():
        output_disc = disc_model(image_tensor_disc)
    output_np_disc = output_disc.squeeze().cpu().numpy()
    output_np_disc = output_np_disc > 0.5

    # Perform inference for cup_model
    cup_model.eval()
    with torch.no_grad():
        output_cup = cup_model(image_tensor_cup)
    output_np_cup = output_cup.squeeze().cpu().numpy()
    output_np_cup = output_np_cup > 0.5

    cup_dimensions = calculate_dimensions(output_np_cup)
    disc_dimensions = calculate_dimensions(output_np_disc)
    cup_perimeter, cup_area, cup_diameter_vertical, cup_diameter_horizontal = cup_dimensions['perimeter'], cup_dimensions['area'], cup_dimensions['diameter_vertical'], cup_dimensions['diameter_horizontal']
    disc_perimeter, disc_area, disc_diameter_vertical, disc_diameter_horizontal = disc_dimensions['perimeter'], disc_dimensions['area'], disc_dimensions['diameter_vertical'], disc_dimensions['diameter_horizontal']

    if disc_diameter_horizontal != 0 and cup_diameter_horizontal != 0:
        horizontal_cdr = cup_diameter_horizontal / disc_diameter_horizontal
        horizontal_error = (0.1355 / cup_diameter_horizontal) + (0.038 / disc_diameter_horizontal)
        horizontal_low = horizontal_cdr - horizontal_error
        horizontal_high = horizontal_cdr + horizontal_error
    else:
        horizontal_cdr, horizontal_low, horizontal_high = 0, 0, 0

    if disc_diameter_vertical != 0 and cup_diameter_vertical != 0:
        vertical_cdr = cup_diameter_vertical / disc_diameter_vertical
        vertical_error = (0.1355 / cup_diameter_vertical) + (0.038 / disc_diameter_vertical)
        vertical_low = vertical_cdr - vertical_error
        vertical_high = vertical_cdr + vertical_error
    else:
        vertical_cdr, vertical_low, vertical_high = 0, 0, 0

    if horizontal_cdr == 0 and vertical_cdr == 0:
        text = 'Cup or Disc not detected. Please enter a more clear picture.'
        color = 'blue'
    elif horizontal_high >= 0.6:
        text = f'CDR = ({horizontal_low:.3f}, {horizontal_high:.3f}) Glaucoma Detected'
        color = 'red'
    elif vertical_high >= 0.6:
        text = f'CDR = ({vertical_low:.3f}, {vertical_high:.3f}) Glaucoma Detected'
        color = 'red'
    else:
        cdr_low = min(horizontal_low, vertical_low)
        cdr_high = max(horizontal_high, vertical_high)
        text = f'(CDR = {cdr_low:.3f}, {cdr_high:.3f}) Healthy Eye'
        color = 'green'
    fig = plt.figure()
    # Display side by side
    plt.subplot(2, 2, 1)
    #plt.imshow(cv2.resize(img,(256,256)))
    plt.imshow(img)
    plt.title('Uploaded Image')

    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_img)
    plt.title('Pre-processed Image')

    # Apply mask and visualize for disc_model
    overlay_disc = apply_mask(img_resized, output_np_disc, color=(1.0, 0.0, 0.0))
    plt.subplot(2, 2, 3)
    plt.imshow(overlay_disc)
    plt.title(f'Disc Segmentation P={disc_perimeter:.3f}, A={disc_area:.3f}',fontsize=10)

    # Apply mask and visualize for cup_model
    overlay_cup = apply_mask(img_resized, output_np_cup, color=(0.0, 1.0, 0.0))
    plt.subplot(2, 2, 4)
    plt.imshow(overlay_cup)
    plt.title(f'Cup Segmentation P={cup_perimeter:.3f}, A={cup_area:.3f}',fontsize=10) #cdr:.3f
    
    #fig.suptitle('Patient Glaucoma Detection Report', fontsize=32)

    # Add a label to the entire grid
    fig.text(0.5, 0.04, text, ha='center', va='center', fontsize=20, color=color)

    # Set the figure size to 1920x1080
    fig.set_size_inches(1920 / 100, 1080 / 100)

    # Show the plot
    plt.show()


# Example usage
#img_path = 'REFUGE.jpg'
"""process_and_visualize(img_path, cup_model, disc_model)"""


