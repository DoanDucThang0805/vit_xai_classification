# Plant Disease Classification with Vision Transformers and Explainable AI (XAI)

A comprehensive deep learning project for plant disease classification using various neural network architectures with integrated explainable AI techniques. The project includes model training, evaluation, optimization, and interpretability analysis for tomato and pepper disease detection.

## 📋 Project Overview

This project aims to classify plant diseases (specifically tomato and pepper) using multiple deep learning models. It combines state-of-the-art architectures with explainable AI methods to understand model predictions, making it useful for agricultural applications.

### Key Features
- **Multiple Model Architectures**: MobilePlantViT, ResNet50, VGG16, DenseNet121, MobileNetV3, ShuffleNetV2, SqueezeNetV2
- **Explainable AI Integration**: GradCAM, LIME, SHAP for model interpretability
- **Dataset Support**: PlantVillage and PlantDoc datasets
- **Model Export**: ONNX format for deployment
- **Performance Benchmarking**: FLOPs, latency, and accuracy metrics
- **Comprehensive Evaluation**: Per-class F1-scores, confusion matrices, and detailed reports

## 📁 Project Structure

```
vit_xai/
├── data/                          # Dataset directories
│   ├── PlantVillage/             # PlantVillage dataset (13 classes)
│   ├── PlantDoc-Dataset/         # PlantDoc dataset
│   ├── cocoplantdoc/             # COCO-format PlantDoc annotations
│   └── cropped_data/             # Preprocessed cropped images
│
├── checkpoints/                   # Trained model weights
│   ├── plantdoc/                 # Models trained on PlantDoc
│   └── plantvillage/             # Models trained on PlantVillage
│
├── onnx_model/                   # Exported ONNX models
│   ├── plantdoc/
│   └── plant_village/
│
├── reports/                       # Training reports & metrics
│   ├── plant_village/
│   └── plantdoc/
│
├── src/                          # Source code
│   ├── dataset/                  # Data loading and preprocessing
│   │   ├── crop_dataset.py      # Crop images from COCO annotations
│   │   ├── dataset.py           # Generic dataset loader
│   │   └── plantdoc_dataset.py  # PlantDoc-specific loader
│   │
│   ├── model/                   # Model architectures
│   │   ├── mobilenetv3_small.py
│   │   ├── mobileplantvit.py
│   │   └── vgg16.py
│   │
│   ├── trainning/               # Training scripts
│   │   ├── trainer.py           # Core Trainer class
│   │   ├── mobileplantvit_train.py
│   │   ├── resnet50_train.py
│   │   └── ...other_model_trains
│   │
│   ├── inference/               # Inference & evaluation
│   │   ├── inference.py         # Model inference
│   │   ├── f1score.py           # F1-score calculation
│   │   └── params.py            # Configuration parameters
│   │
│   ├── benchmark/               # Performance evaluation
│   │   ├── benchmark.py         # Benchmarking utilities
│   │   ├── caculate_flop.py     # FLOPs calculation
│   │   └── caculate_latency.py  # Latency measurement
│   │
│   ├── xai/                     # Explainable AI techniques
│   │   ├── gradcam.py           # Gradient-weighted Class Activation Maps
│   │   ├── lime.py              # Local Interpretable Model-agnostic Explanations
│   │   ├── shap.py              # SHAP values for interpretability
│   │   ├── visualize.py         # Visualization utilities
│   │   └── test.py              # XAI testing
│   │
│   ├── metric/                  # Evaluation metrics
│   │   └── metric.py            # Accuracy and other metrics
│   │
│   ├── utils/                   # Utility functions
│   │   └── utils.py             # Helper functions (LoadDataset classes)
│   │
│   └── export/                  # Model export
│       └── export_onnx.py       # Export to ONNX format
│
├── notebook/                    # Jupyter notebooks
│   ├── prams.ipynb             # Parameter experiments
│   └── test.ipynb              # Testing & exploration
│
├── images/                     # Project images & visualizations
│
├── trainning.sh               # Main training script
└── README.md                  # This file
```

## 🚀 Getting Started

### Requirements
- Python 3.11
- PyTorch 1.9+
- TorchVision, timm
- NumPy, Pandas, Matplotlib
- Scikit-learn
- OpenCV
- LIME, SHAP, GradCAM libraries
- ONNX, ONNX Runtime

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd vit_xai

# Install dependencies (if using pip)
pip install torch torchvision torchinfo timm # Or appropriate CUDA version
pip install numpy pandas scikit-learn opencv-python pillow matplotlib
pip install lime shap
pip install onnx onnxruntime
```

## 🏋️ Training Models

### Train Single Model

```bash
cd src/

# Train MobilePlantViT on PlantVillage
python -m trainning.mobileplantvit_train

# Train ResNet50
python -m trainning.resnet50_train

# Train VGG16
python -m trainning.vgg16_train

# Train DenseNet121
python -m trainning.densnet_train

# Train MobileNetV3
python -m trainning.mobilenetv3_train

# Train ShuffleNetV2
python -m trainning.shuffelnetv2_train

# Train SqueezeNetV2
python -m trainning.squezzenet_train
```

## 📊 Evaluation & Inference

### Run Inference

```bash
python -m inference.inference
```

## 📈 Explainable AI (XAI)
```bash
python xai_rp.py  # Run all XAI methods together
```

## 🔄 Model Export to ONNX

Export trained models to ONNX format for deployment:

```bash
cd src/
python -m export.export_onnx.py
```

ONNX models are saved in `onnx_model/` directory.

## 📁 Dataset Details

### PlantVillage
- **Total Classes**: 13
- **Format**: Image files organized in class folders
- **Size**: ~16,600 images
- **Resolution**: 224*224 pixels
- **Location**: `data/PlantVillage/`

### PlantDoc
- **Total Classes**: 8 (tomato diseases)
- **Location**: `data/tomato_only/`

### Data Splitting
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%


## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

Specify your license here (MIT, Apache 2.0, etc.)

## 📧 Contact

For questions or issues, please contact the project maintainer or create an issue on GitHub.

## 📚 References

- Vision Transformers: https://arxiv.org/abs/2010.11929
- GradCAM: https://arxiv.org/abs/1610.02055
- LIME: https://arxiv.org/abs/1602.04938
- SHAP: https://arxiv.org/abs/1705.07874
- PlantVillage Dataset: https://plantvillage.org/
- PlantDoc Dataset: https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
