# Facial Emotion Recognition using CNN and Transfer Learning

## Project Overview

This project builds a deep learning image classification system for **facial emotion recognition**. The goal is to classify facial images into one of four emotion categories:

* Happy
* Neutral
* Sad
* Surprise

The project compares multiple computer vision approaches, including custom Convolutional Neural Networks (CNNs), grayscale vs RGB image pipelines, and transfer learning using pretrained CNN architectures.

This project demonstrates an end-to-end deep learning workflow for image classification, including data preparation, image preprocessing, model development, transfer learning, fine-tuning, model evaluation, and business interpretation.

---

## Business Problem

Facial emotion recognition is part of the broader field of **Affective Computing**, where systems attempt to recognize and respond to human emotional states.

Potential use cases include:

* Customer sentiment analysis
* Human-computer interaction
* User experience research
* Healthcare support tools
* Driver monitoring systems
* Online learning engagement analysis

The objective is to develop a model that can accurately recognize facial expressions while also understanding the limitations and risks of emotion-based AI systems.

---

## Dataset

The dataset contains facial images organized into three folders:

```text
train/
validation/
test/
```

Each folder contains four emotion-class subfolders:

```text
happy/
neutral/
sad/
surprise/
```

### Class Distribution

| Split      | Happy | Neutral |   Sad | Surprise |
| ---------- | ----: | ------: | ----: | -------: |
| Train      | 3,976 |   3,978 | 3,982 |    3,173 |
| Validation | 1,825 |   1,216 | 1,139 |      797 |
| Test       |    32 |      32 |    32 |       32 |

The training and validation sets show some class imbalance, especially for the `surprise` class. The test set is balanced with 32 images per class.

---

## Project Workflow

### 1. Data Loading

Images were loaded using TensorFlow/Keras image utilities with separate pipelines for:

* RGB images
* Grayscale images

This allowed comparison between color-based and intensity-based image representations.

### 2. Exploratory Data Analysis

The analysis included:

* Visualizing sample images by class
* Checking class distributions
* Identifying class imbalance
* Comparing RGB and grayscale image inputs
* Reviewing image shape and preprocessing requirements

### 3. Image Preprocessing

Preprocessing steps included:

* Resizing images
* Normalizing pixel values
* Batching images
* Caching and prefetching datasets for faster training
* Preparing RGB inputs for transfer learning models

### 4. Modeling

Several modeling approaches were tested.

#### Custom CNN Models

The project first trained CNN models from scratch using convolutional layers, pooling layers, dropout, and batch normalization.

Experiments included:

* RGB CNN pipeline
* Grayscale CNN pipeline
* Larger CNN architectures
* Callback-based training using early stopping and model checkpointing

#### Transfer Learning

Transfer learning was used to improve performance by leveraging pretrained convolutional feature extractors.

Models evaluated included:

* VGG16 baseline transfer learning
* Fine-tuned VGG16
* ResNetV2 baseline transfer learning

The best overall result came from the fine-tuned VGG16 model.

---

## Model Evaluation

Models were evaluated using:

* Test accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion matrix

Because this is a multi-class classification problem, the classification report and confusion matrix were used to understand per-class performance.

---

## Results Summary

### Custom CNN Results

| Model         | Input Type | Test Accuracy | Notes                                                    |
| ------------- | ---------- | ------------: | -------------------------------------------------------- |
| CNN v1        | RGB        |        0.7656 | Strong baseline CNN performance                          |
| CNN v1        | Grayscale  |        0.7656 | Similar performance to RGB                               |
| Larger CNN v2 | RGB        |        0.7266 | Lower performance, likely due to overfitting/instability |
| Larger CNN v2 | Grayscale  |        0.7344 | Slightly better than RGB v2 but still below CNN v1       |
| CNN v2b       | RGB        |        0.7578 | Improved revised CNN with callbacks                      |

### Transfer Learning Results

| Model                      | Test Accuracy | Notes                                        |
| -------------------------- | ------------: | -------------------------------------------- |
| VGG16 Transfer Learning    |        0.7422 | Frozen pretrained feature extractor          |
| Fine-tuned VGG16           |        0.7891 | Best overall model                           |
| ResNetV2 Transfer Learning |        0.6797 | Lower performance than VGG16 in this project |

---

## Best Model

The best-performing model was the **fine-tuned VGG16 transfer learning model**, achieving approximately:

* **Test Accuracy:** 0.7891
* **Best overall class balance among tested models**
* Strong performance on `happy` and `surprise`
* More difficulty distinguishing `neutral` and `sad`

### Fine-tuned VGG16 Classification Report Summary

| Class    | Precision |   Recall | F1-score |
| -------- | --------: | -------: | -------: |
| Happy    |    Strong |   Strong |    ~0.88 |
| Neutral  |  Moderate | Moderate |    ~0.72 |
| Sad      |  Moderate | Moderate |    ~0.69 |
| Surprise |    Strong |   Strong |    ~0.89 |

The model performed best on more visually distinct expressions such as `happy` and `surprise`, while `neutral` and `sad` were more difficult to separate because they can share subtle facial patterns.

---

## Key Insights

### CNNs are well-suited for facial emotion recognition

CNN models outperformed simpler image classification approaches because they can learn spatial features such as facial contours, eyes, mouth shape, and expression-related patterns.

### RGB and grayscale performed similarly

For the custom CNN experiments, RGB and grayscale pipelines produced similar results. Since many facial emotion cues are based on shape and intensity rather than color, grayscale can be a strong and efficient option.

### Larger models do not always perform better

The larger CNN architecture did not outperform the simpler CNN baseline. This highlights the importance of model regularization, validation monitoring, and avoiding unnecessary complexity.

### Transfer learning improved performance

Fine-tuning VGG16 produced the best result, showing that pretrained image features can improve performance even when the target dataset is relatively small.

### Neutral and sad were the most challenging classes

The model had more difficulty separating `neutral` and `sad`, likely because these expressions can be visually subtle and overlapping.

---

## Business Impact

Facial emotion recognition can help organizations better understand user sentiment, engagement, and experience when used responsibly. A model like this could support applications such as customer feedback analysis, online learning engagement monitoring, or human-computer interaction systems.

However, emotion recognition models should be treated as decision-support tools rather than definitive measures of a person's emotional state. Facial expressions are influenced by context, culture, lighting, pose, and individual differences. For real-world use, this type of system should include human oversight, fairness testing, privacy safeguards, and clear communication about model limitations.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Modeling Techniques

* Convolutional Neural Networks
* Transfer Learning
* Fine-tuning
* Batch Normalization
* Dropout
* Early Stopping
* Model Checkpointing

### Data Processing

* NumPy
* TensorFlow Dataset API

### Visualization and Evaluation

* Matplotlib
* Seaborn
* scikit-learn

### Development Environment

* Google Colab
* GPU runtime using NVIDIA T4 when available

---

## Repository Structure

```text
facial-emotion-recognition/
│
├── notebooks/
│   └── Facial_Emotion_Recognition.ipynb
│
├── images/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── training_curves.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_vgg16.png
│   └── model_comparison.png
│
├── data/
│   └── README.md
│
├── requirements.txt
│
└── README.md
```

---

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Upload the dataset or mount Google Drive.
3. Set the runtime to GPU.
4. Run all cells from top to bottom.

### Option 2: Local Environment

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Suggested Requirements

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Future Improvements

Potential next steps include:

* Add more training data to improve generalization
* Apply stronger data augmentation
* Tune learning rates and fine-tuning depth
* Evaluate EfficientNet or MobileNet for lightweight deployment
* Add Grad-CAM visualizations to explain model attention
* Create a small demo app using Streamlit or Gradio
* Perform fairness and bias analysis across demographic groups if metadata is available

---

## Conclusion

This project demonstrates a complete deep learning workflow for facial emotion recognition. It compares custom CNN models, grayscale vs RGB input pipelines, and transfer learning approaches.

The final results show that fine-tuned transfer learning with VGG16 achieved the best overall performance, while the custom CNN models provided a strong baseline. The project also highlights important real-world considerations such as class imbalance, subtle class overlap, model interpretability, and responsible use of emotion recognition systems.

---

## Author: Jerome Jabson
