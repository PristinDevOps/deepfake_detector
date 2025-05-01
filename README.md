# Deepfake Detector and Verification System

A machine learning system to detect and verify deepfake images and videos using PyTorch, with a user-friendly web interface.

## Overview

This project provides tools to:
1. Download and explore a deepfake dataset from Kaggle
2. Train a deepfake detection model using ResNet50
3. Verify images and videos as real or fake with confidence scores
4. Web interface for easy uploading and analyzing of media

## Requirements

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The system uses the "Deepfake and Real Images" dataset from Kaggle:
- Dataset creator: manjilkarki
- Dataset contains 190,000+ images divided into Train, Test, and Validation splits
- Each split contains both Real and Fake image folders

## Project Structure

- `requirements.txt`: Required dependencies
- `download_dataset.py`: Script to download and explore the dataset
- `deepfake_detector.py`: Main model training script
- `deepfake_verifier.py`: Command-line tool to verify new images or videos as real or fake
- `app.py`: Flask web application for the user interface
- `templates/`: HTML templates for web interface
- `static/`: CSS, JavaScript and result files for web interface
- `deepfake_detector_model.pth`: Trained model file (created after training)

## Usage

### 1. Download and Explore Dataset

```bash
python download_dataset.py
```

This will download the dataset and display statistics about its structure.

### 2. Train the Model

```bash
python deepfake_detector.py
```

This script will:
- Load the dataset from Kaggle
- Train a ResNet50-based model to distinguish between real and fake images
- Save the trained model as `deepfake_detector_model.pth`
- Generate training history plots

### 3. Command-line Verification Tool

To verify a single image:

```bash
python deepfake_verifier.py --input path/to/image.jpg
```

To verify all images in a directory:

```bash
python deepfake_verifier.py --input path/to/images/ --output path/to/output/
```

To analyze a video for deepfakes:

```bash
python deepfake_verifier.py --input path/to/video.mp4 --video --sample_rate 5
```

### 4. Web Interface

Run the web application:

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/`

With the web interface, you can:
- Upload images or videos for verification
- See real-time results with confidence scores
- View frame-by-frame analysis statistics for videos
- Get visual feedback with color-coded results

## Model Details

- Architecture: ResNet50 (pretrained on ImageNet)
- Fine-tuned on the deepfake dataset
- Input: 224x224 RGB images
- Output: Binary classification (Real/Fake) with confidence score

## Performance

After training on the full dataset, the model typically achieves:
- ~95-98% accuracy on the test set
- Good generalization to unseen images

## Limitations

- Performance may vary on different types of deepfakes not present in the training data
- Video analysis is based on frame-by-frame detection and may miss temporal inconsistencies
- Some sophisticated deepfakes might still be difficult to detect
- Web processing of large videos may be slow and memory-intensive 