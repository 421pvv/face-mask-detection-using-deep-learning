# Face Mask Detection using Deep Learning

This project is an updated version of a team project I completed in 2021 (published in IEEE conference proceedings of ICACRS 2022), which previously only classified two categories: "with mask" and "without mask". This enhanced implementation now distinguishes a third category - "mask worn incorrectly", making it more comprehensive for real-world applications.

## Features

- Real-time face mask detection through webcam
- Image upload for mask detection
- Responsive web interface
- High accuracy classification (97.83% on test data)

## Model Performance

The model achieves excellent performance across all three classes:

```
Classification Report:
                        precision    recall  f1-score   support

mask_weared_incorrect       0.99      0.99      0.99       598
with_mask                   0.96      0.98      0.97       598
without_mask                0.99      0.96      0.98       598

accuracy                                        0.98      1794
macro avg                   0.98      0.98      0.98      1794
weighted avg                0.98      0.98      0.98      1794
```

Final validation accuracy: **97.83%**
Final validation loss: **0.0613**

![Training History](static/images/training_history.png)
![Confusion Matrix](static/images/confusion_matrix.png)

## Technology Stack

- **Python 3.9**
- **TensorFlow 2.12** with Metal support for Mac
- **MobileNetV2** for the base convolutional neural network
- **OpenCV** for computer vision and face detection
- **Flask** for web application framework
- **Bootstrap 5** for frontend styling
- **JavaScript** for interactive frontend functionality

## System Requirements

- **Recommended**: macOS with Apple Silicon (M1/M2/M3)
- **Python**: 3.9.x
- At least 4GB RAM
- Webcam (for live detection)

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/421pvv/face-mask-detection-using-deep-learning.git
cd face-mask-detection-using-deep-learning
```

### 2. Create a Conda environment

```bash
# Create a new environment with Python 3.9
conda create -n mask-detection python=3.9
conda activate mask-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for non-Mac users**: If you're not using a Mac with Apple Silicon, you should replace `tensorflow-macos` and `tensorflow-metal` in the requirements.txt with standard `tensorflow`:
>
> ```bash
> pip uninstall tensorflow-macos tensorflow-metal
> pip install tensorflow==2.12.0
> ```

## Project Structure

```
face-mask-detection/
├── data/
│   └── processed/           # Place your dataset here
│       └── mask_dataset/    # Dataset directory
│           ├── with_mask/
│           ├── without_mask/
│           └── mask_weared_incorrect/
├── models/
│   ├── face_detector/       # Face detection model files
│   └── face_mask_detection_model/ # Trained model for inference
├── notebooks/
│   └── model_training.ipynb # Model training notebook
├── static/
│   ├── css/
│   ├── js/
│   └── images/              # Store visualization images here
│       ├── training_history.png
│       └── confusion_matrix.png
├── templates/               # HTML templates
├── app.py                   # Flask application
├── detection_utils.py       # Utility functions for detection
└── requirements.txt
```

## Dataset

The dataset used for training is from Kaggle: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection/data)

### Dataset Preparation

1. Download the dataset from the Kaggle link
2. Extract the dataset and organize it into three folders:
   - `data/processed/mask_dataset/with_mask/`
   - `data/processed/mask_dataset/without_mask/`
   - `data/processed/mask_dataset/mask_weared_incorrect/`

## Model Training

If you want to train the model yourself:

1. Ensure you have the dataset organized as described above
2. Open the notebook `notebooks/model_training.ipynb`
3. Run the cells in sequence
4. The trained model will be saved to the `models/` directory

The model uses MobileNetV2 architecture with transfer learning, which provides an excellent balance between accuracy and performance for deployment.

## Running the Web Application

To start the web application:

```bash
python app.py
```

This will start the Flask development server, and you can access the application at `http://localhost:3000`.

### Using the Application

1. **Home Page**: Provides options for live detection or image upload
2. **Live Detection**: Accesses your webcam for real-time mask detection
3. **Image Upload**: Allows you to upload an image for mask detection

## Troubleshooting

### Common Issues on macOS

- If you encounter webcam access issues, make sure to grant camera permissions to your terminal/IDE
- For M1/M2/M3 Macs, the application uses a Haar Cascade face detector as a fallback if the DNN detector fails

### Common Issues on Other Platforms

- If you encounter CUDA/GPU errors on Windows/Linux, try disabling GPU acceleration by setting this environment variable:
  ```bash
  export CUDA_VISIBLE_DEVICES=-1  # Linux/Mac
  set CUDA_VISIBLE_DEVICES=-1     # Windows
  ```

## Acknowledgements

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Face Mask Detection Dataset](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection/data)
- [Our Previous Work (2022)](https://ieeexplore.ieee.org/abstract/document/10029096) - Conference paper from ICACRS 2022
