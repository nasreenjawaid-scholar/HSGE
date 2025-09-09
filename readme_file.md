# H-SGE: Hybrid Scene Graph Enrichment for Small Handgun Detection

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the official implementation of **H-SGE (Hybrid Scene Graph Enrichment)**, a novel framework for enhanced handgun detection in surveillance systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

H-SGE addresses the critical challenge of small handgun detection in CCTV surveillance by combining:

- **GAN-based Enhancement**: Improves visibility of small/occluded objects
- **Scene Graph Generation**: Captures spatial-semantic relationships  
- **Knowledge Graph Integration**: Provides contextual validation
- **Multi-YOLO Detection**: Leverages complementary strengths of YOLOv5, YOLOv7, YOLO10, and YOLO11
- **Weighted Fusion**: Intelligently combines multi-model predictions

## ✨ Key Features

- **Significant Performance Improvements**: F1-scores improved from 58-64% to 80-87%
- **Real-time Performance**: Maintains 47-51 FPS across all YOLO variants
- **Robust in Challenging Conditions**: 
  - Low-light scenarios: +16.3-20.1% improvement
  - Occlusion handling: +19.2-20.7% improvement  
  - Small objects (16×16 to 32×32 pixels): +28.7% improvement
- **Statistical Validation**: Large effect sizes (Cohen's d: 0.8-1.2) confirmed

## 🏗️ Architecture

```
Input Image → GAN Enhancement → Scene Graph Generation → Knowledge Graph Integration
                                        ↓
Final Detections ← Weighted Fusion ← Multi-YOLO Detection (v5, v7, v10, v11)
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for optimal performance

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/H-SGE.git
cd H-SGE
```

2. **Create virtual environment**:
```bash
python -m venv hsge_env
source hsge_env/bin/activate  # On Windows: hsge_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download YOLO models** (optional - will auto-download if not present):
```bash
mkdir models
# Models will be automatically downloaded during first run
```

## 🎯 Quick Start

### Basic Usage

```python
from hsge_framework import HSGEFramework, HSGEConfig
import cv2

# Initialize H-SGE framework
config = HSGEConfig()
hsge = HSGEFramework(config)

# Load and process image
image = cv2.imread('path/to/surveillance/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = hsge.detect_handgun(image)

# Display results
print(f"Detected {len(results['detections'])} handguns")
for i, detection in enumerate(results['detections']):
    print(f"Detection {i+1}: confidence={detection['final_confidence']:.3f}")
```

### Command Line Interface

```bash
# Run detection on single image
python detect.py --image path/to/image.jpg --config config.yaml

# Run on video
python detect.py --video path/to/video.mp4 --config config.yaml

# Batch processing
python detect.py --input_dir path/to/images/ --output_dir results/ --config config.yaml
```

## 📊 Dataset Preparation

### Directory Structure

```
data/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

### Annotation Format

```json
[
  {
    "image_file": "image001.jpg",
    "image_id": 1,
    "annotations": [
      {
        "bbox": [x, y, width, height],
        "class": 0,
        "class_name": "handgun"
      }
    ]
  }
]
```

### Supported Datasets

- Custom handgun datasets
- COCO format (filtered for relevant classes)
- Pascal VOC format (with conversion script)

## 🎓 Training

### Complete Training Pipeline

```bash
# Train all components
python train.py --config config.yaml --mode full

# Train individual components
python train.py --config config.yaml --mode gan    # GAN enhancer only
python train.py --config config.yaml --mode yolo   # YOLO fine-tuning only
```

### Configuration

Modify `config.yaml` to adjust:
- Model paths and hyperparameters
- Training parameters
- Data paths
- Hardware settings

### Hardware Requirements

| Configuration | GPU Memory | Performance | Use Case |
|---------------|------------|-------------|----------|
| Minimum | 6GB (RTX 3060) | 2-3 streams | Standard surveillance |
| Recommended | 12GB (RTX 3080) | 4-6 streams | High-accuracy deployment |
| Enterprise | 16GB+ (A4000/A5000) | 8+ streams | Large-scale deployment |

## 📈 Evaluation

### Run Evaluation

```bash
# Evaluate on test dataset
python train.py --config config.yaml --mode eval

# Custom evaluation
python evaluate.py --model_path models/ --test_data data/test/ --output results/
```

### Metrics

The framework reports:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **FPS**: Frames per second for real-time performance

## 📊 Results

### Performance Comparison

| Model | Baseline F1 | H-SGE F1 | Improvement | FPS |
|-------|-------------|----------|-------------|-----|
| YOLOv5 | 58% | 80% | +22% | 51 |
| YOLOv7 | 56% | 82% | +26% | 48 |
| YOLO10 | 62% | 85% | +23% | 49 |
| YOLO11 | 64% | 87% | +23% | 47 |

### Challenging Scenarios

| Scenario | YOLOv5 | YOLOv7 | YOLO10 | YOLO11 |
|----------|--------|--------|--------|--------|
| Low-light | +20.1% | +19.4% | +17.4% | +16.3% |
| Occlusion | +20.4% | +20.4% | +19.2% | +20.7% |
| Multi-scale | +28.7% | +28.7% | +28.7% | +28.7% |

### Statistical Validation

All improvements confirmed with:
- **p-values < 0.001** (highly significant)
- **Cohen's d: 0.8-1.2** (large effect sizes)
- **5-fold cross-validation**

## 🔧 Optimization Strategies

### Real-time Optimization

1. **Selective Processing**: Apply full H-SGE only when YOLO confidence < 0.7
2. **Adaptive Enhancement**: Skip GAN for high-quality images (brightness > 120)
3. **Dynamic Model Selection**: Use fewer models for simple scenarios
4. **Multi-threading**: Parallel processing pipeline

Expected performance gains: **20-30%** with minimal accuracy loss.

## 🚀 Deployment Configurations

### Configuration A: Maximum Accuracy
- **Setup**: All YOLO variants + full H-SGE
- **Performance**: 97.1% precision, 47-51 FPS
- **Use Case**: Forensic analysis, post-incident investigation

### Configuration B: Balanced Performance  
- **Setup**: YOLOv7 + YOLO11 + selective H-SGE
- **Performance**: 95.8% precision, 65-70 FPS
- **Use Case**: Real-time monitoring with high accuracy

### Configuration C: High-Speed Screening
- **Setup**: YOLO11 + adaptive H-SGE
- **Performance**: 93.5% precision, 85-90 FPS  
- **Use Case**: Multiple camera feeds, initial screening

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{jawaid2025hsge,
  title={H-SGE: A Hybrid Model Based on Scene Graph Enrichment},
  author={Jawaid, Nasreen and Ali, Najma Imtiaz and Korejo, Imtiaz Ali and Brohi, Imtiaz Ali and Hassan, Noor Hafeizah Binti},
  journal={Signal, Image and Video Processing},
  volume={19},
  pages={830},
  year={2025},
  publisher={Springer}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO implementations
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- Contributors and researchers in the computer vision community

## 📧 Contact

For questions or collaborations:

- **Najma Imtiaz Ali**: najma@utem.edu.my  
- **Nasreen Jawaid**: nasreen.jawaid@usindh.edu.pk

## 📊 Repository Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/H-SGE)
![GitHub forks](https://img.shields.io/github/forks/yourusername/H-SGE)
![GitHub issues](https://img.shields.io/github/issues/yourusername/H-SGE)

---

**Note**: This implementation is for research and educational purposes. Please ensure compliance with local laws and regulations when deploying surveillance systems.
