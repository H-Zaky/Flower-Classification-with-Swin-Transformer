# Flower Classification with Swin Transformer on TPU
![image](https://github.com/user-attachments/assets/c3162c03-fb91-4e17-b068-3066cd994f02)

This project implements a flower classification model using a Swin Transformer architecture, fine-tuned on a TPU (Tensor Processing Unit). The dataset includes various flower species, and the model is trained to classify images into these species categories. The project also integrates with Weights & Biases (W&B) for experiment tracking.

## Project Overview

- **Dataset**: Flower species images dataset in TFRecord format.
- **Model Architecture**: Swin Transformer (a Vision Transformer model) is used as the backbone for classification.
- **Training Setup**: The model is trained on TPU using TensorFlow, with various augmentations and a custom learning rate scheduler.
- **Evaluation**: The model is evaluated using F1-score, precision, and recall metrics. A confusion matrix is plotted for visualization.
- **Deployment**: Final predictions are saved in a CSV file for submission.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Addons
- Weights & Biases (W&B)
- Swin Transformer for TensorFlow
- Matplotlib
- Scikit-learn

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download and Extract the Dataset:**

    - Ensure that the dataset is available in the `/kaggle/input` directory in TFRecord format.

4. **Install W&B:**

    ```sh
    pip install -qU wandb
    ```

    > Note: Ensure compatibility with other packages like `elegy` if using a virtual environment.

5. **Clone the Swin Transformer TensorFlow Implementation:**

    ```sh
    git clone https://github.com/rishigami/Swin-Transformer-TF
    ```

6. **Add the Swin Transformer to Python Path:**

    ```python
    import sys
    sys.path.append('/kaggle/working/Swin-Transformer-TF')
    ```

## Running the Project

### TPU Setup

1. **Initialize TPU Strategy:**

    The TPU strategy is initialized to leverage TPU cores for distributed training. If a TPU is not available, the default strategy is used.

    ```python
    def get_strategy():
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            strategy = tf.distribute.get_strategy()
        return strategy
    ```

### Data Preparation

1. **Data Augmentation:**

    Random erasing (blockout) is applied to the images as part of data augmentation.

2. **Dataset Loading:**

    The training, validation, and test datasets are loaded from TFRecord files with appropriate preprocessing steps.

    ```python
    def get_training_dataset(do_onehot=False):
        # Code to load and augment training data
    ```

3. **Dataset Visualization:**

    Visualization functions are provided to inspect training, validation, and test datasets.

    ```python
    display_batch_of_images(next(train_batch))
    ```

### Model Training

1. **Custom Learning Rate Scheduler:**

    A custom learning rate scheduler is defined to control the learning rate during training.

    ```python
    def get_lr_callback(plot_schedule=False):
        # Code for learning rate scheduler
    ```

2. **Model Definition and Training:**

    The Swin Transformer model is loaded and fine-tuned. The model is trained with checkpointing and integrated with W&B for experiment tracking.

    ```python
    def load_and_fit_model(print_summary=False):
        # Code to define, compile, and train the model
    ```

    Example training output:

    ```
    Epoch 1: val_f1_score improved from -inf to 0.85973, saving model to checkpoints/swin_large_best
    ```

### Model Evaluation

1. **Prediction and Confusion Matrix:**

    The model is evaluated on the validation set, and a confusion matrix is plotted to visualize the model's performance.

    ```python
    def display_confusion_matrix(cmat, score, precision, recall):
        # Code to plot confusion matrix
    ```

2. **Submission:**

    The model's predictions on the test dataset are saved to a CSV file for submission.

    ```python
    sub_df.to_csv('submission.csv', index=False)
    ```

## Example Output

- **Confusion Matrix:**
  
    ![image](https://github.com/user-attachments/assets/f5fa4cbd-d335-4d43-855e-19b26c7df99b)

- **Submission File:**

    ```csv
    id,label
    287200,automobile
    33557,cat
    281872,deer
    ...
    ```
