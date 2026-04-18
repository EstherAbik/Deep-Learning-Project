# Predicting Object Categories from Limited Labelled Data: Comparing MLP, CNN, and Transfer Learning Approaches on STL-10

*A comparative study of neural network architectures for image classification, examining the effects of regularisation, pretrained feature transfer, and model ensembling on a data-scarce object recognition task.*

---

## 1. Introduction

This report presents a comparative study of deep learning models for image classification on the STL-10 dataset. The central question is: when labelled data is scarce, which combination of architecture, regularisation, and pretrained features produces the best classifier?

Three families of approaches is compared, each addressing the data scarcity problem differently:

**From-scratch fully-connected (MLP):** A Multi-Layer Perceptron that flattens each 96×96 image into a 27,648-dimensional vector. With 14.3 million parameters and only 5,000 training images, this model has roughly 2,860 parameters per training example 

**From-scratch convolutional (CNN):** A 2-block Convolutional Neural Network with only 198,000 parameters. By exploiting local connectivity, weight sharing, and spatial pooling, it achieves far better generalisation than the MLP despite having 72× fewer parameters.

**Pretrained transfer learning (AlexNet, ResNet18, and their ensemble):** Models whose convolutional backbones were pre-trained on 1.2 million ImageNet photos. Because STL-10 and ImageNet share the same visual domain (natural object photographs), the pretrained features transfer directly, enabling high accuracy from just 5,000 examples. Combining both models into an ensemble further improves accuracy by exploiting the diversity of their learned representations.

Each from-scratch model is trained in two variants — with and without regularisation — to isolate the effect of Dropout and BatchNorm on overfitting behaviour. Including the ensemble, this yields seven model configurations in total.

---

## 2. Dataset: STL-10

STL-10 is a benchmark image classification dataset from Stanford containing 96×96 colour photographs across 10 object classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, and truck. The dataset is perfectly balanced (500 images per class in training, 800 per class in test).

The defining characteristic of STL-10 is its small labelled training set: only 5,000 images. Models that generalise well on STL-10 must either have strong inductive biases (like convolutions) or borrow knowledge from external data (like ImageNet).


| Property | Value |
|----------|-------|
| Classes | 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) |
| Image size | 96 × 96 pixels, RGB colour |
| Training images | 5,000 (500 per class) |
| Test images | 8,000 (800 per class) |
| Class balance | Perfectly balanced |

---

## 3. Methodology

### 3.1 Data Preprocessing

Two separate transform pipelines are used because scratch models and pretrained models require different input formats.

**Pipeline A (scratch models):** Images are kept at their native 96×96 resolution and normalised using STL-10 channel statistics (mean = 0.447, 0.440, 0.407; std = 0.260, 0.257, 0.271). Training augmentation includes RandomCrop with 4-pixel padding and random horizontal flipping.

**Pipeline B (pretrained models):** Images are resized to 128×128 using Resize(144) followed by RandomCrop(128) on training and CenterCrop(128) on test. Normalisation uses ImageNet channel statistics (mean = 0.485, 0.456, 0.406; std = 0.229, 0.224, 0.225) because the pretrained backbones expect inputs normalised with the statistics they were trained on. Using different normalisation would shift the activations away from the learned range and degrade feature quality.

### 3.2 Model Architectures

| Model | Architecture | Total Params | Trainable |
|-------|-------------|-------------|-----------|
| MLP | Flatten → 512 → 256 → 128 → 10 (ReLU) | 14,321,802 | 14,321,802 |
| MLP + Reg | Same + Dropout(0.2) between layers | 14,321,802 | 14,321,802 |
| CNN | 2×[Conv→Conv→MaxPool] + AdaptiveAvgPool(4,4) → 128 → 10 | 198,058 | 198,058 |
| CNN + Reg | Same + BatchNorm2d + Dropout(0.25) | 198,698 | 198,698 |
| AlexNet | Pretrained features (frozen) + classifier (unfrozen) | 57,044,810 | 54,575,114 |
| ResNet18 | Pretrained backbone (frozen) + .fc replaced | 11,177,538 | 5,130 |
| Ensemble | Average of AlexNet + ResNet18 softmax probabilities | — | — |

**MLP:** The input (96×96×3 = 27,648 values) is flattened into a single vector. The first Linear layer alone has 14.16 million weights. This architecture has no notion of spatial proximity.

**CNN:** Two convolutional blocks extract local features. AdaptiveAvgPool2d(4,4) reduces the spatial dimensions to a fixed 4×4 grid, feeding a compact 1,024-dimensional vector into the classifier. The CNN has 72× fewer parameters than the MLP.

**AlexNet:** The convolutional features are frozen (pretrained ImageNet weights). The full classifier is unfrozen because AlexNet's classifier contains Dropout(0.5) layers — when the Linear layers underneath are frozen, the Dropout corrupts the forward signal without the model being able to adapt. Training uses lr=1e-4 to protect the pretrained weights.

**ResNet18:** The entire backbone is frozen and only the final fully-connected layer (.fc, 5,130 parameters) is replaced. ResNet18 has no Dropout in its architecture, so the frozen-backbone approach works cleanly.

**Ensemble:** The softmax probability vectors from AlexNet and ResNet18 are averaged element-wise, and the class with the highest average probability is selected. This requires no additional training.

### 3.3 Training Configuration

| Hyperparameter | Scratch models | Pretrained models |
|---------------|---------------|-------------------|
| Epochs | 30 | 10 |
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 (AlexNet) / 1e-3 (ResNet18) |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR |
| Batch size | 64 | 64 |
| Weight decay | 1e-4 (reg versions) / 0 | 5e-4 |

### 3.4 Data Leakage Prevention

The test set is evaluated every epoch for monitoring purposes, but no training decisions are based on test metrics. The epoch count is fixed in advance. Final-epoch accuracy is reported, not best-epoch accuracy, because selecting the best epoch retroactively based on test performance would constitute data leakage.

---

## 4. Understanding Test Loss vs Test Accuracy

Before interpreting the results, it is important to understand the difference between test loss and test accuracy, as they can tell different stories about model behaviour.

**Test accuracy** is the percentage of test images correctly classified. It is a hard, binary metric: each prediction is either right or wrong, regardless of the model's confidence.

**Test loss** (cross-entropy) measures how confident and well-calibrated the model's predictions are. It penalises not just incorrect predictions but also correct predictions made with low confidence. A model that predicts the right class with 51% probability is penalised more than one that predicts it with 99% probability.

These two metrics can diverge in important ways:

**Rising test loss with flat accuracy:** The model maintains the same number of correct predictions but its confidence is degrading — an early warning sign of overfitting. The model is becoming overconfident on training data at the expense of generalisation.

**Falling test loss with flat accuracy:** The model isn't flipping more predictions from wrong to right, but its confidence in correct predictions is improving — a sign of healthy learning.

In our experiments, the no-reg MLP demonstrates the first pattern: test accuracy plateaus around 45% while test loss rises from 1.67 → 1.95 between epochs 10 and 30. The regularised MLP's test loss keeps falling (1.86 to 1.53) throughout training, indicating that Dropout prevents the confidence degradation. This makes test loss a more sensitive diagnostic for overfitting than accuracy alone.

---

## 5. Results and Analysis

### 5.1 MLP vs CNN — Architecture Matters

The CNN outperforms the MLP despite having 72× fewer parameters. With 96×96 images and only 5,000 training samples, the MLP's 14.3 million parameters create a massive over-parameterisation problem: approximately 2,860 parameters per training image.

The CNN succeeds because it encodes three prior assumptions about images: (1) local connectivity — each neuron sees only a small patch, matching the spatial structure of natural images; (2) weight sharing — the same filter is applied everywhere, so a feature detector works at all locations; (3) pooling — AdaptiveAvgPool2d reduces spatial dimensions and provides translation invariance.

### 5.2 Regularisation — Before vs After

**MLP:** Without regularisation, the MLP reaches ~80% training accuracy but only ~45% test accuracy — a 35-percentage-point gap with rising test loss. Adding Dropout(0.2) reduces the gap to ~12 points and keeps test loss falling. However, both versions achieve similar test accuracy (~44–45%), indicating that the MLP is architecturally limited on this task. Regularisation prevents degradation but cannot overcome the architectural bottleneck.

**CNN:** BatchNorm stabilises gradient flow (accelerating convergence) and acts as a mild regulariser. The regularised CNN typically starts at higher accuracy in epoch 1 and reaches a higher final accuracy. Dropout(0.25) provides additional protection against feature co-adaptation.

### 5.3 Transfer Learning 

STL-10 contains natural object photos — the same domain as ImageNet. The pretrained backbone's features (edges, textures, shapes, object parts learned from 1.2 million images) are directly relevant to STL-10's classification task.

With only 5,000 training images, from-scratch models cannot learn rich visual features. The pretrained backbone brings external knowledge that transforms the problem from "learn everything from 5,000 images" to "learn a classifier on top of already-excellent features."

### 5.4 Ensemble — Combining Diverse Models

By averaging the softmax probabilities of AlexNet and ResNet18, the ensemble achieves higher accuracy than either model alone. This works because the two architectures make partially uncorrelated errors — AlexNet's large-filter features (11×11, 5×5) and ResNet18's small-filter features with residual connections (3×3) capture different aspects of the images.

The ensemble requires no additional training. It is a post-hoc combination at inference time. Even a modest accuracy gain demonstrates the principle that diverse models combined outperform any individual model.

---


## 6. Conclusions

Four lessons emerge from this experiment, each visible because of STL-10's small training set:

**1. Architecture matters.** A CNN with 198K parameters outperforms an MLP with 14.3M parameters because it encodes the right inductive biases for image data. More parameters do not compensate for the wrong assumptions about data structure.

**2. Regularisation matters when data is scarce.** Dropout and BatchNorm prevent the train-test gap from exploding. The key diagnostic is test loss, not just test accuracy — rising test loss signals overfitting even when accuracy appears stable.

**3. Pretrained features matter most — when the domain matches.** ImageNet features transfer to STL-10 because both contain natural object photographs. Pretrained models achieve the highest accuracy with orders of magnitude fewer trainable parameters.

**4. Model diversity adds value.** Ensembling two architecturally different pretrained models improves accuracy beyond either alone, at zero additional training cost. The principle extends to any setting where diverse models make uncorrelated errors.

Taken together, these results illustrate a hierarchy of importance for data-scarce image classification: the right architecture provides a necessary foundation, regularisation prevents that foundation from cracking under limited data, pretrained features provide external knowledge that small datasets cannot supply, and ensembling extracts the last bit of performance from diverse models.

---

## 8. Reproducibility

All random seeds are fixed (torch.manual_seed(42), np.random.seed(42)). CPU runs are exactly reproducible; GPU runs may vary by ±1% due to non-deterministic GPU kernels.

To reproduce: install dependencies from requirements.txt, open Deep_Learning_Project.ipynb in Jupyter, and select Kernel → Restart & Run All. The first run downloads ~2.6 GB of data. No API keys or external services are required.
