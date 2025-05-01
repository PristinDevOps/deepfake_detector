import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import glob
import cv2
from tqdm import tqdm

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model architecture (must match the training architecture)
def get_model(model_path, num_classes=2):
    # Load the same architecture as used during training
    model = models.resnet50(weights=None)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Define image transformation
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Predict single image
def predict_image(image_path, model, transform):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_class = torch.max(outputs, 1)
    
    # Get prediction and confidence
    is_real = bool(predicted_class.item())
    confidence = probabilities[predicted_class.item()].item()
    
    return is_real, confidence

# Add prediction text to image
def add_prediction_to_image(image_path, is_real, confidence):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Prepare text
    result_text = f"{'REAL' if is_real else 'FAKE'}: {confidence:.2%}"
    text_color = (0, 255, 0) if is_real else (255, 0, 0)  # Green for real, red for fake
    
    # Use a default font (may need adjustment on different systems)
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fall back to default
        font = ImageFont.load_default()
    
    # Draw text with outline for visibility
    # Black outline
    for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        draw.text((10 + offset[0], 10 + offset[1]), result_text, fill=(0, 0, 0), font=font)
    # Colored text
    draw.text((10, 10), result_text, fill=text_color, font=font)
    
    # Save the annotated image
    output_path = str(Path(image_path).with_stem(f"{Path(image_path).stem}_verified"))
    image.save(output_path)
    
    return output_path

# Process a directory of images
def process_directory(input_dir, model, transform, output_dir=None):
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    results = []
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Make prediction
            is_real, confidence = predict_image(image_path, model, transform)
            
            # Add prediction to image
            if output_dir:
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                image = Image.open(image_path).convert('RGB')
                
                # Add text to image
                draw = ImageDraw.Draw(image)
                result_text = f"{'REAL' if is_real else 'FAKE'}: {confidence:.2%}"
                text_color = (0, 255, 0) if is_real else (255, 0, 0)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Draw text with outline
                for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    draw.text((10 + offset[0], 10 + offset[1]), result_text, fill=(0, 0, 0), font=font)
                draw.text((10, 10), result_text, fill=text_color, font=font)
                
                # Save the annotated image
                image.save(output_path)
                
            # Record result
            results.append({
                'file': image_path,
                'is_real': is_real,
                'confidence': confidence,
            })
            
            print(f"{image_path}: {'REAL' if is_real else 'FAKE'} (Confidence: {confidence:.2%})")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Summarize results
    real_count = sum(1 for r in results if r['is_real'])
    fake_count = len(results) - real_count
    
    print(f"\nSummary:")
    print(f"Total images: {len(results)}")
    print(f"Real images: {real_count} ({real_count/len(results):.2%})")
    print(f"Fake images: {fake_count} ({fake_count/len(results):.2%})")
    
    return results

# Process video frames
def process_video(video_path, model, transform, output_path=None, sample_rate=1):
    """
    Process video file by analyzing frames.
    
    Args:
        video_path: Path to input video
        model: Deepfake detection model
        transform: Image transformation pipeline
        output_path: Optional path for output video with annotations
        sample_rate: Process every Nth frame (default: 1 = all frames)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video if requested
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    results = []
    frame_count = 0
    processed_frames = 0
    
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            
            # Only process every Nth frame
            if frame_count % sample_rate != 0:
                if output_path:
                    out.write(frame)  # Write original frame
                continue
            
            processed_frames += 1
            
            # Convert frame to PIL Image for model
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(pil_frame).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted_class = torch.max(outputs, 1)
            
            is_real = bool(predicted_class.item())
            confidence = probabilities[predicted_class.item()].item()
            
            # Add to results
            results.append({
                'frame': frame_count,
                'is_real': is_real,
                'confidence': confidence
            })
            
            # Annotate frame if output requested
            if output_path:
                # Add prediction text to frame
                text = f"{'REAL' if is_real else 'FAKE'}: {confidence:.2%}"
                text_color = (0, 255, 0) if is_real else (0, 0, 255)  # BGR format: Green for real, Red for fake
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Outline
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                
                # Write annotated frame
                out.write(frame)
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    # Calculate summary
    frames_analyzed = len(results)
    real_frames = sum(1 for r in results if r['is_real'])
    fake_frames = frames_analyzed - real_frames
    
    # Overall verdict based on majority
    is_real_video = real_frames > fake_frames
    
    # Confidence of verdict (how many frames agree)
    majority_pct = max(real_frames, fake_frames) / frames_analyzed
    
    print(f"\nVideo Analysis Summary:")
    print(f"Total frames analyzed: {frames_analyzed}")
    print(f"Real frames: {real_frames} ({real_frames/frames_analyzed:.2%})")
    print(f"Fake frames: {fake_frames} ({fake_frames/frames_analyzed:.2%})")
    print(f"Verdict: This video is {'REAL' if is_real_video else 'FAKE'} with {majority_pct:.2%} confidence")
    
    return {
        'total_frames': frames_analyzed,
        'real_frames': real_frames,
        'fake_frames': fake_frames,
        'is_real': is_real_video,
        'confidence': majority_pct
    }

# Main function
def main():
    parser = argparse.ArgumentParser(description='Deepfake Verification Tool')
    parser.add_argument('--model', type=str, default='deepfake_detector_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory for processed images')
    parser.add_argument('--video', action='store_true',
                        help='Flag to indicate input is a video file')
    parser.add_argument('--sample_rate', type=int, default=5,
                        help='Sample every Nth frame for video processing (default: 5)')
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = get_model(args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get transformation
    transform = get_transform()
    
    # Process input
    if args.video:
        # Process video
        if os.path.isfile(args.input):
            output_path = args.output if args.output else args.input.replace('.mp4', '_verified.mp4')
            result = process_video(args.input, model, transform, output_path, args.sample_rate)
        else:
            print(f"Error: {args.input} is not a valid video file.")
    else:
        # Process image or directory
        if os.path.isfile(args.input):
            # Single image
            try:
                is_real, confidence = predict_image(args.input, model, transform)
                if args.output:
                    output_path = args.output
                else:
                    output_path = add_prediction_to_image(args.input, is_real, confidence)
                
                print(f"Result: {'REAL' if is_real else 'FAKE'}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Annotated image saved to: {output_path}")
            except Exception as e:
                print(f"Error processing image: {e}")
        
        elif os.path.isdir(args.input):
            # Directory of images
            results = process_directory(args.input, model, transform, args.output)
        
        else:
            print(f"Error: {args.input} is not a valid file or directory.")

if __name__ == "__main__":
    main() 