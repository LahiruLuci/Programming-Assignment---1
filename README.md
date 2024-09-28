# Programming-Assignment---1


# Image Classifier with ResNet-50 and Streamlit Dashboard

This repository contains a web application built using **Streamlit** that classifies images using the **ResNet-50** deep learning model. The application also provides visual interpretations of model predictions using **Integrated Gradients** from the **Captum** library.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)

---

## Overview

This project allows users to upload an image, classify it using the **ResNet-50** model pretrained on **ImageNet**, and visualize the model's prediction probabilities. It also interprets predictions using **Integrated Gradients**, which highlights the regions of the image most important to the prediction. The web interface is built using **Streamlit**, making it easy for users to interact with the model.

## Features

- **Upload images** in `png`, `jpg`, or `jpeg` format.
- **ResNet-50 Model** for image classification with top-5 predictions.
- **Integrated Gradients** for visual interpretation of the model’s decision.
- **Probability Visualization** showing the top 5 class probabilities.
- **Simple UI** for easy interaction built with **Streamlit**.

## Installation

### Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.7+
- pip (Python package manager)

### Install dependencies

Clone the repository and install the required dependencies using the following commands:

```bash
git clone https://github.com/your_username/image_classifier_with_streamlit.git
cd image_classifier_with_streamlit

# Install required Python packages
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, you can install the necessary libraries individually:

```bash
pip install streamlit torch torchvision captum pillow matplotlib
```

### Running the App

Once dependencies are installed, you can run the Streamlit app with the following command:

```bash
streamlit run streamlit_dashboard.py
```

This will start a local server, and you can access the app in your browser at `http://localhost:8501`.

## Usage

1. After starting the Streamlit app, upload an image by clicking on the "Upload Image" button.
2. The app will preprocess the image and use **ResNet-50** to make predictions.
3. The top 5 predictions will be displayed as a bar chart.
4. The uploaded image and a heatmap visualization of the **Integrated Gradients** interpretation will be displayed side-by-side.

## Project Structure

```
.
├── image_classifier.ipynb         # Jupyter notebook for experimentation (details not provided)
├── streamlit_dashboard.py         # Main Streamlit application file
├── requirements.txt               # Python package dependencies (optional)
└── README.md                      # Project README file
```

### Explanation of Key Files:

- **image_classifier.ipynb**: A Jupyter Notebook (details not provided) likely used for experimentation with the model.
- **streamlit_dashboard.py**: The main file responsible for running the Streamlit application. It loads the ResNet-50 model, processes the uploaded image, makes predictions, and visualizes the results using Matplotlib and Captum.
  - Key components include:
    - **load_model()**: Loads the ResNet-50 model with pretrained ImageNet weights.
    - **make_prediction()**: Makes predictions on the processed image and returns the top 5 classes.
    - **interpret_prediction()**: Uses Integrated Gradients to visualize feature importance for the prediction.
  
## Technologies

- **Streamlit**: Used for building the web application interface.
- **PyTorch & Torchvision**: For loading the pre-trained ResNet-50 model and making predictions.
- **Captum**: For generating the feature importance visualizations using Integrated Gradients.
- **Matplotlib**: For plotting the class probabilities and visualizing the images.
- **Pillow**: For image processing and handling.

