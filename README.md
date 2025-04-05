# A5
GenAI

To create a chatbot that can describe a short story based on an uploaded image using Python, you can follow these steps:

Set up a Flask application to handle the web interface and image uploads.
Integrate a pre-trained image captioning model to generate descriptions of the uploaded images.
Generate a short story based on the image description using a text generation model like GPT-3 or GPT-4.
Here's a basic implementation to get you started:

Step 1: Set up Flask Application
First, you need to install Flask and some other dependencies. You can do this using pip:

bash
pip install flask torch torchvision transformers
Step 2: Create the Flask Application
Create a file named app.py and add the following code:

Python
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize the image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate caption and story
        caption = generate_caption(filepath)
        story = f"Once upon a time, {caption}."

        return render_template('result.html', caption=caption, story=story)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
Step 3: Create HTML Templates
Create a folder named templates and add the following HTML files:

index.html
HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Upload</title>
</head>
<body>
  <h1>Upload an image</h1>
  <form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
  </form>
</body>
</html>
result.html
HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Result</title>
</head>
<body>
  <h1>Generated Story</h1>
  <p><strong>Caption:</strong> {{ caption }}</p>
  <p><strong>Story:</strong> {{ story }}</p>
  <a href="/">Upload another image</a>
</body>
</html>
Step 4: Run the Application
Run your Flask application using the following command:

bash
python app.py
This will start a local web server. You can open your browser and navigate to http://127.0.0.1:5000/ to upload an image and generate a short story based on it.

This is a basic implementation, and you can enhance it by adding more sophisticated story generation techniques, better error handling, and deploying it to a cloud platform for broader accessibility.

