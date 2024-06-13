# CIFAR-10 Image Classification

This project demonstrates image classification using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset using PyTorch. It includes training scripts, a GUI for predicting image classes, and a pre-trained model.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [GUI](#gui)
- [Requirements](#requirements)

## Project Structure

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Loma-AIResearcher/CNN.git
    ```
2. Change to the project directory:
    ```bash
    cd CNN
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
#There is already trained model in Models folder with Accuracy :71.69%
1. Download the CIFAR-10 dataset:
    ```bash
    python src/download.py
    ```
2. Train the model:
    ```bash
    python src/train.py
    ```


    


### Prediction

Run the prediction script with a test image:
```bash
python src/predict.py
```



### Run GUI
#IF WANTED#

```bash
python src/gui.py

