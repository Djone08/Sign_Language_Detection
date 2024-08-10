# Sign Language Detection

## Overview

This project is centered around the development of a machine learning model that detects and interprets American Sign Language (ASL) gestures. Using a deep learning approach, specifically Convolutional Neural Networks (CNNs), the model is trained on image data to classify various hand signs with the aim of facilitating communication for individuals with hearing or speech impairments.

## Features

- **Hand Gesture Recognition**: The model identifies and classifies ASL signs from images.
- **Deep Learning Model**: Utilizes CNNs for robust and accurate gesture recognition.
- **Data Augmentation**: Enhances training by applying transformations such as rotation, scaling, and flipping to images.
- **Real-time Inference**: Potential for integrating into applications that require real-time sign language detection.

## Project Structure

- `data/`: Directory where datasets are stored.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `models/`: Directory for saving trained models and checkpoints.
- `scripts/`: Python scripts for data preprocessing, training the model, and running predictions.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies needed to run the project.

## Setup Instructions

### Prerequisites

- Python 3.7+
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

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

 - **Data Preprocessing:** Use the scripts in the scripts/ folder or Jupyter notebooks to preprocess the data (e.g., resizing images, applying data augmentation).

 - **Model Training:** Run the training script or use a Jupyter notebook to train the CNN model on the prepared dataset.

 - **Evaluation:** Evaluate the model’s performance on test data using metrics like accuracy and loss.

 - **Inference:** Use the trained model to make predictions on new ASL images.

## Example Commands

 - **Train the Model:**
   ```bash
   python scripts/train_model.py
   ```
 - **Evaluate the Model:**
   ```bash
   python scripts/evaluate_model.py
   ```
 - **Run Inference:**
   ```bash
   python scripts/predict.py --image_path "path_to_image"
   ```

## Results

The model achieves significant accuracy in classifying ASL signs, demonstrating potential for real-world applications in communication aids and educational tools.

## Contributing

Contributions are encouraged! Feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
