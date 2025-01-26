import os
import torch
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Import the necessary functions and classes from your existing code
from report import process_and_visualize, AttentionUNet

app = Flask(__name__, template_folder='templates')

# Load the models
cup_model_path = 'model-Modified-Unet base-resnet-18 dim-256x256-fold-3.bin'
cup_model = AttentionUNet().to("cpu")
cup_model.load_state_dict(torch.load(cup_model_path, map_location=torch.device('cpu')))

disc_model_path = 'model-ModifiedUnet5 base-resnet dim-256x256-fold-2.bin'
disc_model = AttentionUNet().to("cpu")
disc_model.load_state_dict(torch.load(disc_model_path, map_location=torch.device('cpu')))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Save the image file to a temporary location
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)

        # Call your process_and_visualize function to generate the report
        process_and_visualize(image_path, cup_model, disc_model)

        # Set the figure size and dpi
        fig = plt.gcf()  # Get the current figure
        fig.set_size_inches(12, 6)  # Set the desired size in inches
        fig.set_dpi(100)  # Set the desired dpi

        # Get the base64-encoded image data from the matplotlib figure
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')

        # Render the template with the base64-encoded image data
        return render_template('index.html', img_base64=img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)