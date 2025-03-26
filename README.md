# CSE 527 - Computer Vision Assignments

This repository contains assignments for the CSE 527 Computer Vision course at Stony Brook University by Prof. Haibin Ling. The assignments focus on implementing and experimenting with various computer vision and deep learning techniques.

## Repository Structure
```
.
├── README.md
├── env.yml
├── hw1
│   ├── hw1_part1.ipynb
│   ├── hw1_part2.ipynb
│   ├── hw1_part3.ipynb
│   └── source_images/
├── hw2
│   ├── data/
│   └── hw2_part1.ipynb
├── hw3
│   ├── coco.zip
│   ├── part1
│   │   ├── CryoEM
│   │   │   ├── data.yaml
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── particle_coordinates/
│   │   ├── calibri.ttf
│   │   ├── data
│   │   │   └── MNIST/
│   │   ├── hw3_part1.ipynb
│   │   └── yolo11m.pt
│   ├── part2
│   │   ├── detr
│   │   │   ├── README.md
│   │   │   ├── d2
│   │   │   │   ├── README.md
│   │   │   │   ├── configs/
│   │   │   │   ├── converter.py
│   │   │   │   ├── detr
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── config.py
│   │   │   │   │   ├── dataset_mapper.py
│   │   │   │   │   └── detr.py
│   │   │   │   └── train_net.py
│   │   │   ├── datasets
│   │   │   │   ├── __init__.py
│   │   │   │   ├── coco.py
│   │   │   │   ├── coco_eval.py
│   │   │   │   ├── coco_panoptic.py
│   │   │   │   ├── panoptic_eval.py
│   │   │   │   └── transforms.py
│   │   │   ├── engine.py
│   │   │   ├── hubconf.py
│   │   │   ├── instances_train2017.json
│   │   │   ├── main.py
│   │   │   ├── models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── backbone.py
│   │   │   │   ├── detr.py
│   │   │   │   ├── matcher.py
│   │   │   │   ├── position_encoding.py
│   │   │   │   ├── segmentation.py
│   │   │   │   └── transformer.py
│   │   │   ├── requirements.txt
│   │   │   └── util/
│   │   └── hw3_part2.ipynb
│   └── part3
│       ├── hw3_part3.ipynb
│       └── unet
│           ├── coco.py
│           ├── main.py
│           ├── transforms.py
│           ├── unet.py
│           └── util/
└── hw4
    ├── part1
    │   ├── Config.py
    │   ├── RGB-T234.zip
    │   ├── SiamFCMModel.pth
    │   ├── SiamFCVideo.zip
    │   ├── SiamNet.py
    │   └── hw4_part1.ipynb
    └── part2
        ├── hw4_part2.ipynb
        ├── resources/
        └── source_images/
```

## Setup Instructions

### Environment Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd cse527-assignments
```

2. Create and activate the Conda environment:

```bash
conda env create -f env.yml
conda activate cse527
```

## Assignments

### Assignment 1: Image Processing Fundamentals
Introduction to fundamental image processing techniques implemented across three parts, working with various image transformations and manipulations using the provided source images.

### Assignment 2: Deep Learning for Scene Recognition
Focuses on implementing deep learning techniques for computer vision:
- Training convolutional neural networks from scratch for scene recognition
- Fine-tuning pre-trained networks (AlexNet) for improved accuracy
- Implementing Vision Transformers (ViT) for image recognition tasks

### Assignment 3: Object Detection and Segmentation
Multi-part assignment working with the COCO dataset, exploring advanced computer vision concepts and deep learning applications for object detection and image understanding.

### Assignment 4: Object Tracking and Keypoints
Focuses on object tracking using siamese network and concepts related to image keypoints and stitching.

## Dependencies

The project uses Python 3.10 and includes the following main dependencies:
- PyTorch 2.5.1
- TorchVision 0.20.1
- NumPy 1.26.4
- Matplotlib 3.10.0
- scikit-learn 1.6.0
- OpenCV Python 4.10.0
- Pandas 2.2.3
- Ultralytics 8.3.55
- Seaborn 0.13.2
- Jupyter Notebook

All dependencies are specified in the `env.yml` file.

## License

All rights reserved. This project is for educational purposes only and part of the CSE 527 coursework.