# American Sign Language and Facial Expression Recognition

## Overview
This project implements a deep learning-based system for real-time American Sign Language (ASL) and facial expression recognition. It utilizes state-of-the-art object detection models, including YOLOv9, YOLOv8, Detectron2 (Faster R-CNN), and DETR, to accurately detect and interpret ASL gestures and facial expressions in video streams.

## Authors
- Ravi Teja Vempati

## Abstract
Our study proposes a novel approach for ASL and facial expression detection using deep learning models. We evaluated the performance of YOLOv9, YOLOv8, Detectron2 (Faster R-CNN), and DETR on a manually annotated dataset. The results showed that YOLOv9 performed better in accurately detecting ASL signs and facial expressions in real-time video streams.

## Dataset
- 4429 training images
- 551 validation images
- 195 test images
- Ratio: 22:3:1
- Augmentation techniques applied:
  - Rotations between -15° and +15°
  - Noise addition up to 3% of pixels
  - Brightness adjustment between -20% and +20%
  - Exposure adjustment between -10% and +10%

## Requirements
- Python 3.x
- Google Colab
- Roboflow
- Ultralytics
- Supervision package
- PyTorch
- Detectron2
- DETR

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/asl-facial-expression-recognition.git
   cd asl-facial-expression-recognition
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the dataset:
   - Organize your dataset into train, validation, and test sets
   - Use Roboflow for data annotation and augmentation

## Usage
1. To train the models:
   ```
   python train.py --model [yolov9|yolov8|detectron2|detr]
   ```

2. To evaluate the models:
   ```
   python evaluate.py --model [yolov9|yolov8|detectron2|detr]
   ```

3. To run real-time detection:
   ```
   python detect.py --model yolov9 --source [video_file|0 for webcam]
   ```

## Results
Our experiments showed that YOLOv9 outperformed other models in detecting ASL signs and facial expressions. Detailed results, including training times, test results, and performance metrics, can be found in the full research paper.

## Future Work
- Implement multimodal systems integrating various sensory inputs
- Enhance recognition of non-manual features
- Develop a user-friendly interface for real-world applications

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
