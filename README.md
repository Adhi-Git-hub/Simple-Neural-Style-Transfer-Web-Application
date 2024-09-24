# Simple Neural Style Transfer Web Application

## Overview

This project is a **Simple Neural Style Transfer Web Application** built using Flask and TensorFlow Hub. The application allows users to upload a content image, choose a style image, and apply neural style transfer to generate an artistic, stylized version of the content image. The result can be previewed on the webpage and downloaded for personal use.

## Features

- Upload your own content image.
- Choose from multiple predefined style images.
- Apply Neural Style Transfer using a TensorFlow Hub model.
- Preview the stylized output directly on the web page.
- Download the output image for personal use.

## Technologies Used

- **Flask**: Backend framework to handle file uploads and manage the web server.
- **HTML/CSS/JavaScript**: Frontend for designing the user interface.
- **Bootstrap**: Responsive styling and layout for the web page.
- **TensorFlow Hub**: Pre-trained neural style transfer model.
- **Neural Style Transfer**: Deep learning technique for combining a content image with a style image to produce a stylized output.

## Neural Style Transfer

Neural Style Transfer is a deep learning technique that uses convolutional neural networks (CNNs) to apply the artistic style of one image to the content of another image. This involves the following steps:

1. **Content Image**: The image you want to stylize (e.g., a landscape, portrait).
2. **Style Image**: The artistic style you want to apply (e.g., Van Gogh's painting, a texture).
3. **Output Image**: The result of blending the content and style images.

The algorithm utilizes **convolutional layers** from a pre-trained model (such as VGG19) to separate and recombine the **content** and **style** from the respective images. The goal is to minimize the content loss (difference from the original content image) and style loss (difference from the style image’s artistic texture) using optimization techniques.

## TensorFlow Hub Model

TensorFlow Hub provides access to a variety of pre-trained models, including the neural style transfer model used in this project. This makes it easy to integrate advanced deep learning functionalities without having to train a model from scratch.

The specific TensorFlow Hub model used for this project is a pre-trained neural style transfer model that allows us to transform content images by applying different artistic styles with minimal setup.

Model URL: [TensorFlow Hub - Neural Style Transfer](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)

## How It Works

1. **Upload Content Image**: The user uploads an image they want to stylize.
2. **Select a Style Image**: Choose one of the predefined style images to apply.
3. **Apply Style Transfer**: The backend processes the images using the TensorFlow Hub model, combining the content and style.
4. **Preview & Download**: The stylized image is displayed, and the user can download the result.

