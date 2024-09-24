import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import time
import tensorflow_hub as hub
from flask import Flask, request, render_template

# Configurations
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['axes.grid'] = False

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load TensorFlow Hub model for Fast Style Transfer
def load_models():
    print("Loading pre-trained models...")
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    style_content_model = StyleContentModel(style_layers, content_layers)
    print("Models loaded successfully!")
    return hub_model, style_content_model

# Define style and content layers
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Function to extract VGG19 layers
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

# Model for extracting style and content
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = [gram_matrix(output) for output in outputs[:self.num_style_layers]]
        content_outputs = outputs[self.num_style_layers:]

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        return {'style': style_dict, 'content': content_dict}

# Helper Functions
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(input_tensor.shape[1] * input_tensor.shape[2], tf.float32)
    return result / num_locations

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)

# Load images for stylization
def load_img(path_to_img, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    img = tf.image.resize(img, tf.cast(shape * scale, tf.int32))
    return img[tf.newaxis, :]

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def run_style_transfer(content_image, style_image):
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)

# Define the index route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle content image upload
        content_image_file = request.files.get('content_image')
        if content_image_file:
            content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image_file.filename)
            content_image_file.save(content_image_path)

            selected_style = request.form.get('style_image')
            style_image_path = os.path.join('static/styles', selected_style)

            # Load and process images
            content_image = load_img(content_image_path)
            style_image = load_img(style_image_path)
            stylized_image = run_style_transfer(content_image, style_image)

            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'stylized_output.png')
            stylized_image.save(output_image_path)

            return render_template("index.html", result_image='output/stylized_output.png')

    return render_template("index.html")


# Load models at startup
hub_model, style_content_model = load_models()

if __name__ == "__main__":
    app.run(debug=True)
