# Deepfake Detector - Setup Guide

This guide provides detailed installation and setup instructions for the Deepfake Detector and Verification System.

## System Requirements

- Python 3.8+ (3.9 or 3.10 recommended)
- 8GB+ RAM (16GB recommended for video processing)
- CUDA-capable GPU (optional, but highly recommended for faster processing)
- 5GB+ free disk space (for the application, model, and dependencies)

## Installation Steps

### 1. Clone or download the repository

```bash
git clone https://github.com/your-username/deepfake_detector.git
cd deepfake_detector
```

Or download the project and extract to a folder of your choice.

### 2. Create a virtual environment (recommended)

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, you may want to install PyTorch with CUDA support. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions.

### 4. Download the dataset (optional if you only want to use the verification system)

```bash
python download_dataset.py
```

Wait for the download to complete. This will download the dataset to your Kaggle cache directory.

### 5. Train the model (or use a pre-trained model)

If you want to train your own model:

```bash
python deepfake_detector.py
```

This will take several hours depending on your hardware. If you have a pre-trained model file (`deepfake_detector_model.pth`), place it in the project root directory.

## Quick Start Guide

### Using the Web Interface

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Use the web interface to upload images or videos for verification.

### Using the Command Line Tool

For quick verification of a single image:
```bash
python deepfake_verifier.py --input path/to/your/image.jpg
```

For video verification:
```bash
python deepfake_verifier.py --input path/to/your/video.mp4 --video
```

## Troubleshooting

### Common Issues

1. **Model not found error**: Ensure you have the model file `deepfake_detector_model.pth` in the project directory. If not, train your model first or download a pre-trained model.

2. **Memory issues with large videos**: Reduce the frame resolution or increase the sample rate (e.g., `--sample_rate 20`) to process fewer frames.

3. **CUDA out of memory**: Reduce the batch size in `deepfake_detector.py` or process on CPU if your GPU has limited memory.

4. **Dependencies installation problems**: Try installing dependencies one by one or check for specific error messages.

### Getting Help

If you encounter issues not covered here, please refer to the project repository or create an issue on GitHub. 