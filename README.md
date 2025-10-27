# Satellite_Image-Visual_Search
Training a ML model that performs General Visual Search and Retrieval from Satellite Images given sample image chip of the target.

Kaggle Notebook -> [click here](https://www.kaggle.com/code/ashok205/siamese2)

# Model Training Pipeline

# Siamese Network Model for Satellite Object Detection

## Overview
Few-shot learning approach using Siamese networks to detect objects in satellite imagery with minimal training examples (3-5 per class). Achieved 3156 detections through metric learning rather than traditional object detection.

## Architecture

### Model Structure
- **Input**: 64×64 pixel chips with 4 spectral bands (Blue, Green, Red, Near-Infrared)
- **Backbone**: ResNet50 pre-trained on ImageNet
- **Embedding Dimension**: 128-dimensional normalized vectors
- **Output**: L2-normalized embeddings for similarity comparison

### Key Components
1. **4→3 Channel Projection**: 1×1 convolution converts satellite bands to RGB-compatible format
2. **ResNet50 Backbone**: Transfer learning from ImageNet for feature extraction
3. **Embedding Head**: Dense layers (2048→512→128) with unit normalization

## Training Process

### Data Preparation
- **Chip Extraction**: 64×64 patches with 15% padding around labeled objects
- **Background Class**: Random sampling from non-object regions for negative learning
- **Normalization**: Per-band mean/std normalization with auto-detected scale factor
- **Augmentation**: Random horizontal/vertical flips and 90° rotations

### P×K Batch Sampling
- **P_CLASSES = 8**: 8 different classes per batch
- **K_SAMPLES = 3**: 3 examples per class
- **Batch Size = 24**: Balanced positive/negative pairs for triplet learning

### Training Strategy
1. **Phase 1 - Warmup (5 epochs)**: Freeze ResNet50, train embedding head only
2. **Phase 2 - Fine-tuning (145 epochs)**: Unfreeze conv5 block + embedding head
3. **Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4
4. **Early Stopping**: Patience of 10 epochs on validation loss

### Loss Function: Batch-Hard Triplet Loss
- **Concept**: For each anchor, find hardest positive (furthest same-class) and hardest negative (closest different-class)
- **Margin**: 0.5 safety buffer between positive and negative distances
- **Formula**: `loss = softplus(d(A,P) - d(A,N) + margin)`
- **Mining**: Automatic hard pair selection within each batch

## Performance Optimization

### Mixed Precision Training
- **Policy**: float16 for most operations, float32 for critical computations
- **Benefit**: 2x faster training without accuracy loss

### Calibration
- **Purpose**: Convert cosine similarities to detection probabilities
- **Method**: Logistic regression on validation pairs (same-class vs different-class)
- **Threshold**: 0.3 probability for detection confidence

## Inference Pipeline

### Prototype Generation
- Average embeddings from 3-5 exemplar chips per class
- Creates representative "fingerprint" for each object type

### Sliding Window Search
- **Stride**: 32 pixels (50% overlap for comprehensive coverage)
- **Batch Processing**: 64 patches simultaneously for efficiency
- **Similarity Computation**: Dot product between patch embeddings and class prototype
- **Non-Maximum Suppression**: Remove overlapping detections with IoU threshold

## Key Advantages
- **Data Efficient**: Works with 3-5 examples per class vs 1000+ for traditional methods
- **Flexible**: Add new classes without retraining the entire model
- **Generalizable**: Learns similarity rather than class-specific features
- **Fast Training**: 2-3 hours vs 3-5 days for conventional object detection

## Technical Specifications
- **Framework**: TensorFlow with Keras API
- **Hardware**: P100 GPU in Kaggle
- **Output Format**: Bounding boxes with confidence scores for submission in a csv format
