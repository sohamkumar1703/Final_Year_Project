import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def vessel_inpainting(image, mask):
    # Convert the image to grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask
    binary_mask = np.uint8(mask)

    # Apply the inpainting algorithm
    inpainted = cv2.inpaint(image, binary_mask, 3, cv2.INPAINT_NS)

    # Convert the inpainted image back to color
    #inpainted = cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR)

    return inpainted

def enhance_illuminated_region(img):
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    equalized_img = clahe.apply(img)
    
    return equalized_img

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

def extract_largest_contour_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # Find contours and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = image[y:y+h, x:x+w]
        return roi

def enhance_image(image_path):
    # Read the image using PIL
    image = np.array(Image.open(image_path))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image  = extract_largest_contour_roi(image)
    image = resize_image_to_square(image, 512)
    # Split the image into channels (assuming it is in RGB format)
    r, g, b = cv2.split(image)

    # Apply Gaussian blur to the red channel
    r_blurred = cv2.GaussianBlur(r, (5, 5), 0)
    g_blurred = cv2.GaussianBlur(g, (5, 5), 0)
    b_blurred = cv2.GaussianBlur(b, (5, 5), 0)
    print('Gaussian blur applied')

    # Perform top-hat transform on the blurred red channel
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    r_blackhat = cv2.morphologyEx(r_blurred, cv2.MORPH_BLACKHAT, structuring_element)
    g_blackhat = cv2.morphologyEx(g_blurred, cv2.MORPH_BLACKHAT, structuring_element)
    b_blackhat = cv2.morphologyEx(b_blurred, cv2.MORPH_BLACKHAT, structuring_element)
    print('Blackhat performed')

    # Threshold the top-hat image
    _, r_thresholded = cv2.threshold(r_blackhat, 1, 5, cv2.THRESH_BINARY)
    _, g_thresholded = cv2.threshold(g_blackhat, 1, 5, cv2.THRESH_BINARY)
    _, b_thresholded = cv2.threshold(b_blackhat, 1, 5, cv2.THRESH_BINARY)

    # Perform vessel inpainting on the red channel using the thresholded image
    r_inpainted = vessel_inpainting(r, r_thresholded)
    g_inpainted = vessel_inpainting(g, g_thresholded)
    b_inpainted = vessel_inpainting(b, b_thresholded)
    print('Inpainting done.')

    # Apply enhancement on the inpainted image
    #r_inpainted = enhance_illuminated_region(r_inpainted)
    #g_inpainted = enhance_illuminated_region(g_inpainted)
    #b_inpainted = enhance_illuminated_region(b_inpainted)
    

    enhanced_image = cv2.merge((r_inpainted, g_inpainted, b_inpainted))


    return enhanced_image
