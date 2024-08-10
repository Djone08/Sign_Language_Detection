# Sign Language Detection

## Overview

This project is centered around the development of a machine learning model that detects and interprets American Sign Language (ASL) gestures. Using a deep learning approach, specifically Convolutional Neural Networks (CNNs), the model is trained on image data to classify various hand signs with the aim of facilitating communication for individuals with hearing or speech impairments.

## Features

- **Hand Gesture Recognition**: The model identifies and classifies ASL signs from images.
- **Deep Learning Model**: Utilizes CNNs for robust and accurate gesture recognition.
- **Data Augmentation**: Enhances training by applying transformations such as rotation, scaling, and flipping to images.
- **Real-time Inference**: Potential for integrating into applications that require real-time sign language detection.

## Project Structure

- `data/`:
  - `sign_mnist_test.csv`: Test dataset for evaluating the model.
  - `sign_mnist_train.csv`: Training dataset for model training.
- `Sign_Language_Detection.ipynb`: Jupyter notebook for data exploration, model training, and evaluation.
- `sign_language_mnist_cnn.keras`: Saved model file.
- `snip.jpg`: Sample test image from an external source.

## Setup Instructions

### Prerequisites

- Python 3.7+
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

### Installation
>[!TIP]
You Can Directly Download [`Sign_Language_Detection.ipynb`](https://github.com/Djone08/Sign_Language_Detection/blob/main/Sign_Language_Detection.ipynb) and Run The Code in Google Colab

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Djone08/Sign_Language_Detection.git
   cd Sign_Language_Detection
   ```
 2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
 3. **Download and Prepare Dataset:**

    Acquire the ASL dataset and place it in the data/ directory. Ensure the dataset is structured correctly for training.

## Usage

 - **Data Preprocessing:** Use the scripts in the `scripts/` folder or Jupyter notebooks to preprocess the data (e.g., resizing images, applying data augmentation).

 - **Model Training:** Run the training script or use a Jupyter notebook to train the CNN model on the prepared dataset.

 - **Evaluation:** Evaluate the model’s performance on test data using metrics like accuracy and loss.

 - **Inference:** Use the trained model to make predictions on new ASL images.

## Example Commands

 - **Train the Model:**
   ```bash
   python scripts/train_model.py
   ```
 - **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
 - **Download and Prepare Dataset:**
   The dataset files (sign_mnist_test.csv and sign_mnist_train.csv) are already included in the data/ directory.

## Usage

 1. **Data Preprocessing:** Use the Jupyter notebook to preprocess data, including resizing images and applying data augmentation.

 2. **Model Training:** Train the CNN model using the provided Jupyter notebook.

 3. **Evaluation:** Evaluate model performance on test data as described in the notebook.

 4.**Inference:** Use the trained model to make predictions on new ASL images, such as snip.jpg.

## Example Commands

 - **Train the Model:**
   Follow instructions in the Jupyter notebook for training

 - **Evaluate the Model:*
   Follow instructions in the Jupyter notebook for evaluation
 
 - **Run Inference:**
   Follow instructions in the Jupyter notebook to predict using 'snip.jpg'

## Results

The model achieves high accuracy in classifying ASL signs, showing promise for real-world applications in communication aids and educational tools.

## Contributing

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
