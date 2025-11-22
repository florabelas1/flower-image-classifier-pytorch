# Flower Image Classifier with PyTorch

**AWS AI Programming with Python Nanodegree - Final Project**

Deep learning image classifier for flower species recognition using transfer learning 
and PyTorch. Command-line application with training and inference capabilities.

---

## 🎯 Project Objective

Build an image classifier that:
1. Trains on 102 flower species dataset
2. Uses transfer learning with pre-trained CNN models
3. Provides command-line interface for training and prediction
4. Achieves high accuracy through data augmentation and fine-tuning

---

## 🌸 Dataset

**102 Category Flower Dataset**
- 102 different flower species
- ~8,000 images total
- Split: 70% training, 15% validation, 15% testing
- Variable image sizes and backgrounds

---

## 🧠 Model Architectures Supported

| Architecture | Layers | Parameters | Performance |
|--------------|--------|------------|-------------|
| **VGG16** | 16 | 138M | Highest accuracy |
| **AlexNet** | 8 | 61M | Fastest training |
| **ResNet18** | 18 | 11M | Best balance |

All models use **transfer learning** - pre-trained on ImageNet, fine-tuned on flowers.

---

## 🛠️ Technologies

- **PyTorch** (torchvision)
- **Python 3.x**
- **PIL** (Image processing)
- **NumPy**
- **argparse** (CLI)
- **JSON** (category mapping)

---

## 📁 Project Structure

```
├── train.py                           # Training script
├── predict.py                         # Prediction script
├── cat_to_name.json                   # Category ID to flower name mapping
├── Image_Classifier_Project.html     # Full notebook with analysis
└── training_log.txt                   # Training metrics log
```

---

## 🚀 Usage

### Training a Model

```bash
# Basic training (VGG16, 5 epochs)
python train.py --data_path flowers/

# Advanced training with custom parameters
python train.py --arch resnet18 \
                --data_path flowers/ \
                --lr 0.001 \
                --hidden 512 \
                --epochs 10 \
                --gpu
```

#### Training Arguments

- `--arch`: Model architecture (`vgg16`, `alexnet`, `resnet18`)
- `--data_path`: Path to dataset directory
- `--lr`: Learning rate (default: 0.001)
- `--hidden`: Hidden layer size (default: 4096)
- `--epochs`: Number of training epochs (default: 5)
- `--gpu`: Enable GPU training (if available)

### Making Predictions

```bash
# Predict flower species from image
python predict.py path/to/image checkpoint.pth

# With top-K predictions and category names
python predict.py path/to/image checkpoint.pth \
                  --top_k 5 \
                  --category_names cat_to_name.json \
                  --gpu
```

#### Prediction Arguments

- `image_path`: Path to image file
- `checkpoint`: Path to saved model checkpoint
- `--top_k`: Return top K predictions (default: 5)
- `--category_names`: JSON file mapping categories to names
- `--gpu`: Use GPU for inference

---

## 📊 Results

### Model Performance

- **Training Accuracy**: 95%+
- **Validation Accuracy**: 85-90%
- **Test Accuracy**: 80-85%

### Data Augmentation Applied

✅ Random rotation (±30°)  
✅ Random resized crop (224x224)  
✅ Random horizontal flip  
✅ Normalization (ImageNet statistics)

---

## 🔬 Transfer Learning Approach

1. **Load Pre-trained Model**: Models trained on ImageNet (1000 classes)
2. **Freeze Feature Extraction**: Keep convolutional layers unchanged
3. **Replace Classifier**: New fully-connected layers for 102 flower classes
4. **Fine-tune**: Train only the new classifier layers

### Custom Classifier Architecture

```
Input (from CNN features)
    ↓
Fully Connected (hidden_units)
    ↓
ReLU + Dropout(0.2)
    ↓
Fully Connected (102 classes)
    ↓
LogSoftmax
```

---

## 📈 Training Process

### What Happens During Training:

1. **Data Loading**: Load images with augmentation
2. **Forward Pass**: Compute predictions
3. **Loss Calculation**: Negative Log Likelihood Loss
4. **Backpropagation**: Update classifier weights
5. **Validation**: Check performance on validation set
6. **Checkpoint Saving**: Save best model

### Sample Training Output

```
Epoch 1/5.. Train loss: 3.892.. Valid loss: 2.145.. Valid accuracy: 0.452
Epoch 2/5.. Train loss: 2.234.. Valid loss: 1.456.. Valid accuracy: 0.623
Epoch 3/5.. Train loss: 1.789.. Valid loss: 1.123.. Valid accuracy: 0.724
Epoch 4/5.. Train loss: 1.456.. Valid loss: 0.967.. Valid accuracy: 0.789
Epoch 5/5.. Train loss: 1.234.. Valid loss: 0.878.. Valid accuracy: 0.821

Training complete! Model saved as checkpoint.pth
```

---

## 🎓 Project Context

**Program**: AWS AI Programming with Python Nanodegree  
**Provider**: Udacity  
**Focus**: Deep Learning, Transfer Learning, PyTorch, CLI Applications  
**Year**: 2024

### Learning Objectives Achieved

✅ Implement deep learning models in PyTorch  
✅ Apply transfer learning for image classification  
✅ Build command-line applications with argparse  
✅ Handle data augmentation and preprocessing  
✅ Save and load model checkpoints  
✅ Optimize training with GPU acceleration

---

## 🔍 Key Technical Concepts

### Transfer Learning Benefits
- **Faster Training**: Leverage pre-learned features
- **Better Accuracy**: Start from proven architectures
- **Less Data Required**: Works well with limited datasets

### Data Augmentation
- **Prevents Overfitting**: Model learns invariant features
- **Increases Diversity**: Artificially expands training data
- **Improves Generalization**: Better performance on new images

### Checkpoint System
- **Resume Training**: Continue from last saved state
- **Model Deployment**: Load trained model for predictions
- **Experiment Tracking**: Compare different configurations

---

## 💡 Example Predictions

```bash
$ python predict.py test_images/rose.jpg checkpoint.pth --top_k 3

Top 3 Predictions:
1. Rose (98.5%)
2. Wild Rose (1.2%)
3. Pink Primrose (0.3%)
```

---

## 🎯 Future Enhancements

- [ ] Add web interface for image upload
- [ ] Implement model ensemble (combine multiple architectures)
- [ ] Add confusion matrix visualization
- [ ] Support for additional datasets
- [ ] Mobile app deployment
- [ ] Real-time video classification

---

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

*AWS AI Programming with Python Nanodegree - 2024*
