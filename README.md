# PPE Detection with YOLO

A computer vision system for detecting Personal Protective Equipment (PPE) in video streams.
This project uses a custom-trained YOLO model to identify helmet, gloves, boots, vest, and human classes in real time.

The repository contains the full inference pipeline, sample code, and instructions for running the model locally.

## Features

- Fine-tuned YOLO model trained on PPE dataset
- Real-time detection on videos or webcam
- Bounding boxes, labels, and confidence scores
- Simple and modular code structure

## Tech Stack

- Python
- Ultralytics YOLO
- OpenCV

## Installation

### Create a virtual environment (recommended):

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows


### Install dependencies:

pip install -r requirements.txt



## Model Training

This project uses a custom fine-tuned YOLO model.
The model was trained on a PPE dataset containing the following classes:

0: boost
1: gloves
2: helmet
3: human
4: vest

Training was done using Ultralytics YOLO with transfer learning (fine-tuning) on the custom dataset.
The model weights are included in model/fine_tuned_model.pt.

## Dataset

The training dataset is sourced from Roboflow, licensed under CC BY 4.0.
Attribution required by the license:

Dataset: PPE Detection

Source: Roboflow Universe

License: CC BY 4.0


## Notes

Full-HD videos should be resized internally for stable performance.
The system runs smoothly on mid-range GPUs; CPU inference is possible but slower.
This repo is intended for research, learning, and prototyping purposes.

## License

MIT License
